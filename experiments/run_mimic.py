"""
MIMIC-IV clinical data experiment.

Reproduces Section 5.3 from Benac et al. (2024):
  - Sepsis patient trajectories from MIMIC-IV
  - Discretized vitals -> states, treatments -> actions
  - Learn hospital transition dynamics via ITL
  - Outlier detection and Bayesian Regret for transfer

PREREQUISITES:
  1. PhysioNet credentialed access (https://physionet.org/content/mimiciv/)
  2. Download MIMIC-IV tables to data/mimic-iv/
  3. pip install scikit-learn (for k-means clustering)

Following Komorowski et al. (2018) "Artificial Intelligence Clinician":
  - Sepsis-3 criteria: suspected infection + SOFA >= 2
  - 4-hour time windows for state/action aggregation
  - K-means clustering for state discretization (750 clusters)
  - 5x5 vasopressor/fluid grid for action discretization (25 actions)
  - Reward: +15 survival at 90 days, -15 mortality (sparse terminal reward)
"""

import numpy as np
import pandas as pd
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mdp import TabularMDP, deterministic_policy
from src.expert import trajectories_to_counts
from src.itl_solver import solve_itl
from src.utils import transition_mse_visited_vs_unvisited, print_results_table


# =============================================================================
# CONFIGURATION
# =============================================================================

MIMIC_PATH = "data/mimic-iv"
RESULTS_PATH = "results/mimic"

# Komorowski et al. (2018) parameters
N_STATE_CLUSTERS = 750
N_VASO_BINS = 5
N_FLUID_BINS = 5
N_ACTIONS = N_VASO_BINS * N_FLUID_BINS  # 25
TIME_WINDOW_HOURS = 4
GAMMA = 0.99  # high discount for clinical decisions over days
REWARD_SURVIVE = +15
REWARD_DEATH = -15

# Vital sign features for state discretization
STATE_FEATURES = [
    "heart_rate", "systolic_bp", "mean_arterial_pressure", "diastolic_bp",
    "respiratory_rate", "spo2", "temperature", "fio2",
    "gcs_total",
    "urine_output_4h",
    "lactate", "creatinine", "bilirubin_total", "platelets", "wbc",
    "potassium", "sodium", "glucose",
    "pao2", "paco2", "ph", "bun",
]


# =============================================================================
# Step 1: COHORT EXTRACTION
# =============================================================================

def extract_sepsis_cohort(mimic_path: str) -> pd.DataFrame:
    """
    Extract sepsis cohort from MIMIC-IV following Komorowski et al. (2018).

    Inclusion criteria:
      - Age >= 18
      - ICU stay >= 24 hours
      - Suspected infection (antibiotics + cultures within 24h)
      - SOFA score >= 2 at some point during ICU stay

    Exclusion criteria:
      - Readmissions (only first ICU stay per hospital admission)
      - ICU stay < 24 hours (insufficient data)
      - Missing key vitals in first 24 hours

    Returns:
        DataFrame with columns:
            subject_id, hadm_id, stay_id, intime, outtime,
            los_hours, mortality_90d, age, gender
    """
    mp = Path(mimic_path)

    # Load core tables
    patients = pd.read_csv(mp / "hosp" / "patients.csv.gz",
                           usecols=["subject_id", "gender", "anchor_age"])
    admissions = pd.read_csv(mp / "hosp" / "admissions.csv.gz",
                             usecols=["subject_id", "hadm_id", "admittime",
                                      "dischtime", "deathtime", "hospital_expire_flag"])
    icustays = pd.read_csv(mp / "icu" / "icustays.csv.gz",
                           usecols=["subject_id", "hadm_id", "stay_id",
                                    "intime", "outtime", "los"])

    # Parse dates
    for col in ["admittime", "dischtime", "deathtime"]:
        admissions[col] = pd.to_datetime(admissions[col])
    for col in ["intime", "outtime"]:
        icustays[col] = pd.to_datetime(icustays[col])

    # Merge
    cohort = icustays.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
    cohort = cohort.merge(patients, on="subject_id", how="inner")

    # Age filter
    cohort = cohort[cohort["anchor_age"] >= 18]

    # LOS filter (>= 24 hours)
    cohort["los_hours"] = (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600
    cohort = cohort[cohort["los_hours"] >= 24]

    # Keep only first ICU stay per admission
    cohort = cohort.sort_values("intime").groupby("hadm_id").first().reset_index()

    # 90-day mortality
    cohort["mortality_90d"] = 0
    death_mask = cohort["deathtime"].notna()
    days_to_death = (cohort.loc[death_mask, "deathtime"] -
                     cohort.loc[death_mask, "outtime"]).dt.days
    cohort.loc[death_mask & (days_to_death <= 90), "mortality_90d"] = 1
    # In-hospital deaths also count
    cohort.loc[cohort["hospital_expire_flag"] == 1, "mortality_90d"] = 1

    # Sepsis filter: look for Sepsis-3 ICD codes
    diagnoses = pd.read_csv(mp / "hosp" / "diagnoses_icd.csv.gz",
                            usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])

    # Sepsis ICD-10 codes (A40, A41, R65.2x) and ICD-9 (995.91, 995.92, 785.52)
    sepsis_icd10 = diagnoses[
        (diagnoses["icd_version"] == 10) &
        (diagnoses["icd_code"].str.startswith(("A40", "A41", "R652")))
    ]["hadm_id"].unique()

    sepsis_icd9 = diagnoses[
        (diagnoses["icd_version"] == 9) &
        (diagnoses["icd_code"].isin(["99591", "99592", "78552"]))
    ]["hadm_id"].unique()

    sepsis_hadm_ids = set(sepsis_icd10) | set(sepsis_icd9)
    cohort = cohort[cohort["hadm_id"].isin(sepsis_hadm_ids)]

    print(f"  Sepsis cohort: {len(cohort)} ICU stays")
    print(f"  90-day mortality rate: {cohort['mortality_90d'].mean():.1%}")
    print(f"  Median LOS: {cohort['los_hours'].median():.0f} hours")

    return cohort[["subject_id", "hadm_id", "stay_id", "intime", "outtime",
                   "los_hours", "mortality_90d", "anchor_age", "gender"]]


# =============================================================================
# Step 2: VITAL SIGN EXTRACTION (4-hour windows)
# =============================================================================

def extract_vitals_timeseries(mimic_path: str, cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Extract vital signs and labs in 4-hour windows for each ICU stay.

    For each stay, creates time windows [0h, 4h), [4h, 8h), ...
    and aggregates vital signs within each window (median).

    Missing values are forward-filled within each stay, then
    imputed with population median.

    Returns:
        DataFrame with columns: stay_id, window_idx, [features...]
    """
    mp = Path(mimic_path)

    # MIMIC-IV chartevents ItemIDs for key vitals
    vital_itemids = {
        "heart_rate": [220045],
        "systolic_bp": [220050, 220179],
        "mean_arterial_pressure": [220052, 220181],
        "diastolic_bp": [220051, 220180],
        "respiratory_rate": [220210, 224690],
        "spo2": [220277],
        "temperature": [223761, 223762],  # F and C
        "fio2": [223835, 227009],
        "gcs_total": [220739, 223900, 223901],  # eye + verbal + motor
    }

    # Lab ItemIDs
    lab_itemids = {
        "lactate": [50813],
        "creatinine": [50912],
        "bilirubin_total": [50885],
        "platelets": [51265],
        "wbc": [51300, 51301],
        "potassium": [50971, 50822],
        "sodium": [50983, 50824],
        "glucose": [50931, 50809],
        "pao2": [50821],
        "paco2": [50818],
        "ph": [50820],
        "bun": [51006],
    }

    stay_ids = cohort["stay_id"].values
    stay_times = cohort.set_index("stay_id")[["intime", "outtime"]].to_dict("index")

    # Read chartevents in chunks (large file)
    all_itemids = []
    for ids in vital_itemids.values():
        all_itemids.extend(ids)
    for ids in lab_itemids.values():
        all_itemids.extend(ids)

    print(f"  Extracting vitals for {len(stay_ids)} stays...")

    records = []
    chartevents_path = mp / "icu" / "chartevents.csv.gz"
    labevents_path = mp / "hosp" / "labevents.csv.gz"

    # Process chartevents
    if chartevents_path.exists():
        for chunk in pd.read_csv(chartevents_path, chunksize=500000,
                                 usecols=["stay_id", "charttime", "itemid", "valuenum"]):
            chunk = chunk[chunk["stay_id"].isin(stay_ids) &
                         chunk["itemid"].isin(all_itemids)]
            if len(chunk) > 0:
                records.append(chunk)

    # Process labevents
    if labevents_path.exists():
        lab_all_ids = [i for ids in lab_itemids.values() for i in ids]
        for chunk in pd.read_csv(labevents_path, chunksize=500000,
                                 usecols=["subject_id", "hadm_id", "charttime",
                                          "itemid", "valuenum"]):
            chunk = chunk[chunk["hadm_id"].isin(cohort["hadm_id"].values) &
                         chunk["itemid"].isin(lab_all_ids)]
            if len(chunk) > 0:
                # Map hadm_id to stay_id
                hadm_to_stay = cohort.set_index("hadm_id")["stay_id"].to_dict()
                chunk["stay_id"] = chunk["hadm_id"].map(hadm_to_stay)
                chunk = chunk.dropna(subset=["stay_id"])
                records.append(chunk[["stay_id", "charttime", "itemid", "valuenum"]])

    if not records:
        raise FileNotFoundError("No vital sign data found. Check MIMIC-IV file paths.")

    events = pd.concat(records, ignore_index=True)
    events["charttime"] = pd.to_datetime(events["charttime"])
    events["stay_id"] = events["stay_id"].astype(int)

    # Create item-to-feature mapping
    itemid_to_feature = {}
    for feat, ids in {**vital_itemids, **lab_itemids}.items():
        for item_id in ids:
            itemid_to_feature[item_id] = feat

    events["feature"] = events["itemid"].map(itemid_to_feature)
    events = events.dropna(subset=["feature", "valuenum"])

    # Compute time windows
    result_rows = []
    for stay_id in stay_ids:
        if stay_id not in stay_times:
            continue
        info = stay_times[stay_id]
        intime = info["intime"]
        outtime = info["outtime"]

        stay_events = events[events["stay_id"] == stay_id].copy()
        if len(stay_events) == 0:
            continue

        stay_events["hours_since_admit"] = (
            stay_events["charttime"] - intime
        ).dt.total_seconds() / 3600
        stay_events["window_idx"] = (stay_events["hours_since_admit"] // TIME_WINDOW_HOURS).astype(int)
        stay_events = stay_events[stay_events["window_idx"] >= 0]

        # Aggregate: median per (window, feature)
        grouped = stay_events.groupby(["window_idx", "feature"])["valuenum"].median().unstack()

        # Forward fill, then impute with column median
        grouped = grouped.ffill()

        for window_idx in grouped.index:
            row = {"stay_id": stay_id, "window_idx": int(window_idx)}
            for feat in STATE_FEATURES:
                row[feat] = grouped.loc[window_idx, feat] if feat in grouped.columns else np.nan
            result_rows.append(row)

    vitals_df = pd.DataFrame(result_rows)

    # Impute remaining NaN with population median
    for feat in STATE_FEATURES:
        if feat in vitals_df.columns:
            median_val = vitals_df[feat].median()
            vitals_df[feat] = vitals_df[feat].fillna(median_val)

    print(f"  Extracted {len(vitals_df)} time windows across {vitals_df['stay_id'].nunique()} stays")
    return vitals_df


# =============================================================================
# Step 3: TREATMENT EXTRACTION
# =============================================================================

def extract_treatments(mimic_path: str, cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Extract vasopressor doses and IV fluid volumes in 4-hour windows.

    Vasopressors (converted to norepinephrine equivalents):
      - Norepinephrine: 1x
      - Epinephrine: 1x
      - Vasopressin: 2.5 units/hr = 0.04 mcg/kg/min NE equivalent
      - Dopamine: 1/100 of NE equivalent
      - Phenylephrine: 0.1x NE equivalent

    IV fluids: total volume of crystalloid + colloid in 4h window.
    """
    mp = Path(mimic_path)
    stay_ids = cohort["stay_id"].values
    stay_times = cohort.set_index("stay_id")[["intime"]].to_dict("index")

    inputevents_path = mp / "icu" / "inputevents.csv.gz"
    if not inputevents_path.exists():
        raise FileNotFoundError(f"inputevents not found at {inputevents_path}")

    # Vasopressor ItemIDs in MIMIC-IV
    vaso_items = {
        221906: 1.0,     # Norepinephrine (mcg/kg/min)
        221289: 1.0,     # Epinephrine
        222315: 0.04,    # Vasopressin (units/hr -> NE equivalent)
        221662: 0.01,    # Dopamine
        221749: 0.1,     # Phenylephrine
    }

    # IV fluid ItemIDs (crystalloids + colloids)
    fluid_items = [
        220949,  # Normal saline 0.9%
        220950,  # Dextrose 5%
        225158,  # Lactated Ringers
        225159,  # Normal saline (different entry)
        225168,  # Albumin 5%
        225174,  # Albumin 25%
    ]

    print("  Extracting treatments...")
    records = []
    for chunk in pd.read_csv(inputevents_path, chunksize=200000,
                             usecols=["stay_id", "starttime", "itemid",
                                      "amount", "rate", "rateuom"]):
        chunk = chunk[chunk["stay_id"].isin(stay_ids)]
        relevant = chunk[chunk["itemid"].isin(list(vaso_items.keys()) + fluid_items)]
        if len(relevant) > 0:
            records.append(relevant)

    if not records:
        raise FileNotFoundError("No treatment data found.")

    treatments = pd.concat(records, ignore_index=True)
    treatments["starttime"] = pd.to_datetime(treatments["starttime"])

    # Compute time windows and aggregate
    result_rows = []
    for stay_id in stay_ids:
        if stay_id not in stay_times:
            continue
        intime = stay_times[stay_id]["intime"]
        stay_treats = treatments[treatments["stay_id"] == stay_id].copy()
        if len(stay_treats) == 0:
            continue

        stay_treats["hours"] = (stay_treats["starttime"] - intime).dt.total_seconds() / 3600
        stay_treats["window_idx"] = (stay_treats["hours"] // TIME_WINDOW_HOURS).astype(int)
        stay_treats = stay_treats[stay_treats["window_idx"] >= 0]

        # Get max window
        max_window = stay_treats["window_idx"].max()
        for w in range(max_window + 1):
            window_treats = stay_treats[stay_treats["window_idx"] == w]

            # Vasopressor: max rate in window (NE equivalents)
            vaso_dose = 0.0
            for item_id, factor in vaso_items.items():
                item_rows = window_treats[window_treats["itemid"] == item_id]
                if len(item_rows) > 0:
                    vaso_dose += item_rows["rate"].max() * factor

            # IV fluid: total volume in window
            fluid_vol = 0.0
            fluid_rows = window_treats[window_treats["itemid"].isin(fluid_items)]
            if len(fluid_rows) > 0:
                fluid_vol = fluid_rows["amount"].sum()

            result_rows.append({
                "stay_id": stay_id,
                "window_idx": w,
                "vasopressor_dose": vaso_dose,
                "iv_fluid_4h": fluid_vol,
            })

    treatments_df = pd.DataFrame(result_rows)
    print(f"  Extracted treatments for {treatments_df['stay_id'].nunique()} stays")
    return treatments_df


# =============================================================================
# Step 4: STATE & ACTION DISCRETIZATION
# =============================================================================

def discretize_states(vitals_df: pd.DataFrame, n_clusters: int = 750):
    """
    Discretize continuous vital signs into discrete states via k-means.

    Following Komorowski et al. (2018): 750 clusters on standardized features.
    Uses MiniBatchKMeans for scalability.

    Returns:
        states: array of cluster labels
        kmeans: fitted model (for applying to new data)
        scaler: fitted StandardScaler
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler

    features = [f for f in STATE_FEATURES if f in vitals_df.columns]
    X = vitals_df[features].values

    # Handle any remaining NaN
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:, j])
        X[nan_mask, j] = col_medians[j]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                             batch_size=2000, n_init=3)
    states = kmeans.fit_predict(X_scaled)

    print(f"  Discretized into {n_clusters} states (k-means inertia: {kmeans.inertia_:.0f})")
    return states, kmeans, scaler


def discretize_actions(treatments_df: pd.DataFrame,
                       n_vaso_bins: int = 5,
                       n_fluid_bins: int = 5) -> np.ndarray:
    """
    Discretize treatments into 25 action bins (5 vaso x 5 fluid).

    Bin 0 = no treatment. Bins 1-4 are quartiles of nonzero doses.
    """
    vaso = treatments_df["vasopressor_dose"].values.copy()
    fluid = treatments_df["iv_fluid_4h"].values.copy()

    # Vasopressor bins
    vaso_bins = np.zeros(len(vaso), dtype=int)
    nonzero_vaso = vaso > 0
    if nonzero_vaso.sum() > 0:
        quantiles = np.percentile(vaso[nonzero_vaso],
                                  np.linspace(0, 100, n_vaso_bins))
        vaso_bins[nonzero_vaso] = np.clip(
            np.digitize(vaso[nonzero_vaso], quantiles[1:]), 0, n_vaso_bins - 2) + 1

    # Fluid bins
    fluid_bins = np.zeros(len(fluid), dtype=int)
    nonzero_fluid = fluid > 0
    if nonzero_fluid.sum() > 0:
        quantiles = np.percentile(fluid[nonzero_fluid],
                                  np.linspace(0, 100, n_fluid_bins))
        fluid_bins[nonzero_fluid] = np.clip(
            np.digitize(fluid[nonzero_fluid], quantiles[1:]), 0, n_fluid_bins - 2) + 1

    actions = vaso_bins * n_fluid_bins + fluid_bins
    print(f"  Discretized into {N_ACTIONS} actions ({n_vaso_bins} vaso x {n_fluid_bins} fluid)")
    print(f"  Action distribution: {np.bincount(actions, minlength=N_ACTIONS)}")
    return actions


# =============================================================================
# Step 5: BUILD TRANSITIONS AND REWARD
# =============================================================================

def build_transition_data(vitals_df: pd.DataFrame,
                          treatments_df: pd.DataFrame,
                          states: np.ndarray,
                          actions: np.ndarray,
                          cohort: pd.DataFrame,
                          n_states: int,
                          n_actions: int):
    """
    Build (s, a, s') transition tuples and reward matrix from clinical data.

    Each transition is a 4-hour time step within an ICU stay.
    Terminal transitions receive reward based on 90-day mortality.

    Returns:
        N: transition counts, shape (n_states, n_actions, n_states)
        T_mle: MLE estimates, shape (n_states, n_actions, n_states)
        R: reward matrix, shape (n_states, n_actions)
        trajectories: list of trajectories [(s, a, s'), ...]
    """
    # Merge states and actions by (stay_id, window_idx)
    vitals_df = vitals_df.copy()
    treatments_df = treatments_df.copy()
    vitals_df["state"] = states

    merged = vitals_df[["stay_id", "window_idx", "state"]].merge(
        treatments_df[["stay_id", "window_idx", "vasopressor_dose", "iv_fluid_4h"]],
        on=["stay_id", "window_idx"],
        how="inner"
    )

    # Add action column
    action_array = discretize_actions(
        merged[["vasopressor_dose", "iv_fluid_4h"]].rename(
            columns={"vasopressor_dose": "vasopressor_dose", "iv_fluid_4h": "iv_fluid_4h"}
        )
    )
    merged["action"] = action_array

    # Sort by stay and time
    merged = merged.sort_values(["stay_id", "window_idx"]).reset_index(drop=True)

    # Build transitions
    N = np.zeros((n_states, n_actions, n_states))
    trajectories = []
    mortality = cohort.set_index("stay_id")["mortality_90d"].to_dict()

    for stay_id, group in merged.groupby("stay_id"):
        group = group.sort_values("window_idx")
        s_arr = group["state"].values
        a_arr = group["action"].values

        traj = []
        for t in range(len(s_arr) - 1):
            s, a, s_next = int(s_arr[t]), int(a_arr[t]), int(s_arr[t + 1])
            N[s, a, s_next] += 1
            traj.append((s, a, s_next))

        if traj:
            trajectories.append(traj)

    # MLE
    T_mle = np.zeros_like(N)
    for s in range(n_states):
        for a in range(n_actions):
            total = N[s, a].sum()
            if total > 0:
                T_mle[s, a] = N[s, a] / total
            else:
                T_mle[s, a] = 1.0 / n_states

    # Reward: based on terminal state mortality rates
    # R(s, a) = P(survive | last state = s) * REWARD_SURVIVE + P(die | last state = s) * REWARD_DEATH
    R = np.zeros((n_states, n_actions))
    state_mortality_count = np.zeros(n_states)
    state_total_count = np.zeros(n_states)

    for stay_id, group in merged.groupby("stay_id"):
        last_state = int(group.iloc[-1]["state"])
        died = mortality.get(stay_id, 0)
        state_total_count[last_state] += 1
        state_mortality_count[last_state] += died

    for s in range(n_states):
        if state_total_count[s] > 0:
            mort_rate = state_mortality_count[s] / state_total_count[s]
            R[s, :] = (1 - mort_rate) * REWARD_SURVIVE + mort_rate * REWARD_DEATH
        else:
            R[s, :] = 0.0  # neutral for unobserved terminal states

    print(f"  Built {int(N.sum())} transitions across {len(trajectories)} stays")
    print(f"  Visited (s,a) pairs: {(N.sum(axis=2) > 0).sum()} / {n_states * n_actions}")
    print(f"  Mean transitions per (s,a): {N.sum() / max((N.sum(axis=2) > 0).sum(), 1):.1f}")

    return N, T_mle, R, trajectories


# =============================================================================
# Step 6: RUN ITL ON CLINICAL DATA
# =============================================================================

def run_itl_on_mimic(N, T_mle, R, n_states, n_actions, trajectories):
    """
    Run ITL solver on MIMIC-IV transition data.

    Tests multiple epsilon values and reports results.
    Also runs BITL for outlier detection if feasible.
    """
    print("\n  Running ITL solver...")

    # Determine appropriate epsilon range from Q-value gaps
    mdp_mle = TabularMDP(n_states, n_actions, T_mle, R, GAMMA)
    _, Q_mle, _ = mdp_mle.compute_optimal_policy()
    q_max = Q_mle.max(axis=1, keepdims=True)
    q_gaps = q_max - Q_mle
    nonzero_gaps = q_gaps[q_gaps > 1e-6]

    if len(nonzero_gaps) > 0:
        print(f"  Q-gap statistics: median={np.median(nonzero_gaps):.3f}, "
              f"mean={np.mean(nonzero_gaps):.3f}, max={np.max(nonzero_gaps):.3f}")
        eps_candidates = [
            np.percentile(nonzero_gaps, 25),
            np.percentile(nonzero_gaps, 50),
            np.percentile(nonzero_gaps, 75),
            np.percentile(nonzero_gaps, 90),
        ]
    else:
        eps_candidates = [1.0, 5.0, 10.0, 20.0]

    results = []
    for eps in eps_candidates:
        print(f"\n    epsilon = {eps:.3f}:")
        T_itl, info = solve_itl(N, T_mle, R, GAMMA, epsilon=eps, max_iter=10, verbose=False)
        mle_metrics = transition_mse_visited_vs_unvisited(T_mle, T_mle, N)
        itl_mle_diff = np.mean(np.sum((T_itl - T_mle) ** 2, axis=2))

        print(f"      Converged: {info['converged']}")
        print(f"      ||T_itl - T_mle||: {itl_mle_diff:.6f}")

        # Since we don't have ground truth, measure constraint satisfaction
        mdp_itl = TabularMDP(n_states, n_actions, T_itl, R, GAMMA)
        _, Q_itl, _ = mdp_itl.compute_optimal_policy()

        results.append({
            "epsilon": float(eps),
            "converged": info["converged"],
            "t_diff_from_mle": float(itl_mle_diff),
            "n_iterations": len(info["iterations"]),
        })

    return results


# =============================================================================
# Step 7: OUTLIER DETECTION (requires BITL)
# =============================================================================

def run_outlier_detection(N, T_mle, R, trajectories, n_states, n_actions):
    """
    Run BITL for outlier trajectory detection on clinical data.

    Identifies ICU stays whose transition patterns are unlikely under
    the posterior — potential data quality issues, unusual patients,
    or protocol deviations.
    """
    from src.bitl import bitl_sample, detect_outlier_trajectories

    # Use a smaller state space for BITL (cluster down if needed)
    if n_states > 100:
        print("  NOTE: BITL on 750 states is computationally expensive.")
        print("  Consider reducing n_clusters for BITL analysis.")
        return None

    print("\n  Running BITL for outlier detection...")

    # Get feasible start from ITL
    T_itl, _ = solve_itl(N, T_mle, R, GAMMA, epsilon=5.0, max_iter=10)

    samples, info = bitl_sample(
        N, T_mle, R, GAMMA, epsilon=5.0,
        n_samples=100, n_warmup=50,
        step_size=0.005, n_leapfrog=10,
        T_init=T_itl, seed=42, verbose=True,
    )

    outlier_result = detect_outlier_trajectories(
        trajectories, samples, threshold_percentile=5.0
    )

    print(f"  Detected {outlier_result['n_outliers']} outlier trajectories "
          f"out of {len(trajectories)}")
    print(f"  Outlier indices: {outlier_result['outlier_indices'][:10]}...")

    return outlier_result


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  MIMIC-IV SEPSIS EXPERIMENT")
    print("  Section 5.3 from Benac et al. (2024)")
    print("=" * 60)

    mimic_path = MIMIC_PATH

    if not os.path.exists(mimic_path):
        print(f"\n  MIMIC-IV data not found at '{mimic_path}'.")
        print("\n  To run this experiment, you need:")
        print("    1. PhysioNet credentialed access")
        print("       https://physionet.org/content/mimiciv/")
        print("    2. Complete CITI human subjects training")
        print("    3. Download MIMIC-IV v2.2+ tables:")
        print("       - hosp/patients.csv.gz")
        print("       - hosp/admissions.csv.gz")
        print("       - hosp/diagnoses_icd.csv.gz")
        print("       - hosp/labevents.csv.gz")
        print("       - icu/icustays.csv.gz")
        print("       - icu/chartevents.csv.gz")
        print("       - icu/inputevents.csv.gz")
        print("    4. Place files in data/mimic-iv/ maintaining directory structure")
        print("    5. Install: pip install scikit-learn")
        print()
        print("  Estimated cohort size: ~15,000 sepsis ICU stays")
        print("  Estimated transitions: ~200,000 (s, a, s') tuples")
        print()
        print("  Pipeline steps when data is available:")
        print("    1. extract_sepsis_cohort() — Sepsis-3 filtering")
        print("    2. extract_vitals_timeseries() — 4h vital sign windows")
        print("    3. extract_treatments() — vasopressor/fluid doses")
        print("    4. discretize_states() — k-means (750 clusters)")
        print("    5. discretize_actions() — 5x5 treatment grid (25 actions)")
        print("    6. build_transition_data() — (s,a,s') counts")
        print("    7. run_itl_on_mimic() — ITL with epsilon sweep")
        print("    8. run_outlier_detection() — BITL outlier flagging")
        return

    # ==========================================================================
    # FULL PIPELINE (runs when data is available)
    # ==========================================================================

    print("\n  Step 1: Extracting sepsis cohort...")
    cohort = extract_sepsis_cohort(mimic_path)

    print("\n  Step 2: Extracting vital signs (4h windows)...")
    vitals_df = extract_vitals_timeseries(mimic_path, cohort)

    print("\n  Step 3: Extracting treatments...")
    treatments_df = extract_treatments(mimic_path, cohort)

    print("\n  Step 4: Discretizing states...")
    states, kmeans, scaler = discretize_states(vitals_df, n_clusters=N_STATE_CLUSTERS)

    print("\n  Step 5: Building transition data...")
    # First discretize actions from the treatments
    actions = discretize_actions(treatments_df)
    N, T_mle, R, trajectories = build_transition_data(
        vitals_df, treatments_df, states, actions, cohort,
        n_states=N_STATE_CLUSTERS, n_actions=N_ACTIONS
    )

    print("\n  Step 6: Running ITL...")
    itl_results = run_itl_on_mimic(N, T_mle, R, N_STATE_CLUSTERS, N_ACTIONS, trajectories)

    # Step 7: Outlier detection (optional, computationally expensive)
    # outlier_results = run_outlier_detection(N, T_mle, R, trajectories,
    #                                         N_STATE_CLUSTERS, N_ACTIONS)

    # Save results
    os.makedirs(RESULTS_PATH, exist_ok=True)
    with open(os.path.join(RESULTS_PATH, "itl_results.json"), "w") as f:
        json.dump(itl_results, f, indent=2)

    print("\n" + "=" * 60)
    print("  MIMIC-IV EXPERIMENT COMPLETE")
    print(f"  Results saved to {RESULTS_PATH}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
