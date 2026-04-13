# Setup Guide: GitHub + Cursor

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

## Step 5: Install Cursor

1. Go to https://cursor.com
2. Download for macOS
3. Install (drag to Applications)
4. Open Cursor
5. Sign in (free tier is fine to start)

## Step 6: Open the Project in Cursor

1. In Cursor: File > Open Folder
2. Navigate to: `~/Documents/Columbia/Demitrascu research/Demitrascu Research/itl-reproduction`
3. Open it

Cursor has built-in AI assistance. You can:
- Press Cmd+K to ask it to edit code
- Press Cmd+L to chat about the codebase
- It has full context of your project files

## Step 7: Install Python Dependencies

In Cursor's terminal (Ctrl+`):
```bash
pip install -r requirements.txt
```

## Step 8: Verify Everything Works

```bash
# This should match your hand calculations
python experiments/run_corridor.py

# This is the main benchmark
python experiments/run_gridworld.py
```

## Workflow Going Forward

1. Make changes in Cursor
2. Test locally: `python experiments/run_*.py`
3. Commit: `git add . && git commit -m "description of changes"`
4. Push: `git push`
5. Show Bianca the GitHub repo and results
