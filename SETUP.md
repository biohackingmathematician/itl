# Setup Guide

## Step 1: Install Git (if not already installed)

Open Terminal and check:
```bash
git --version
```
If not installed, macOS will prompt you to install Xcode Command Line Tools. Say yes.

## Step 2: Create a GitHub Account

1. Go to https://github.com/join
2. Sign up with your Columbia email (cagnachan@gmail.com or your .edu email)
3. Verify your email

## Step 3: Set Up Git Locally

In Terminal:
```bash
git config --global user.name "Agna Chan"
git config --global user.email "cagnachan@gmail.com"
```

## Step 4: Create the GitHub Repository

1. Go to https://github.com/new
2. Repository name: `itl-reproduction`
3. Description: "Reproducing Inverse Transition Learning (Benac et al., 2024)"
4. Set to **Private** (academic work, don't publish yet)
5. Do NOT initialize with README (we already have one)
6. Click "Create repository"

Then in Terminal, navigate to the project and push:
```bash
cd ~/Documents/Columbia/Demitrascu\ research/Demitrascu\ Research/itl-reproduction
git init
git add .
git commit -m "Initial ITL reproduction scaffold"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/itl-reproduction.git
git push -u origin main
```

GitHub will ask you to authenticate. Use either:
- A Personal Access Token (Settings > Developer settings > Personal access tokens > Tokens (classic) > Generate new token, check "repo" scope)
- Or set up SSH keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

## Step 5: Open the Project in Your Editor

Open the repo folder in whichever editor you prefer (VS Code, Sublime,
neovim, PyCharm, etc.):
```bash
cd ~/Documents/Columbia/Demitrascu\ research/Demitrascu\ Research/itl-reproduction
```

## Step 6: Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Step 7: Verify Everything Works

```bash
# Should match the hand calculations in the corridor verification
python -m experiments.run_corridor

# Main synthetic benchmark
python -m experiments.run_gridworld

# Run the test suite (under 30 s)
python -m pytest tests/
```

## Workflow Going Forward

1. Make changes locally
2. Test: `python -m experiments.run_*` and `python -m pytest tests/`
3. Commit: `git add . && git commit -m "description of changes"`
4. Push: `git push`
5. Discuss the GitHub repo and results in lab meetings
