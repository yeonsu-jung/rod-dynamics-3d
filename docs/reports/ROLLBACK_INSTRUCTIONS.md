# Quick Rollback Instructions

## TL;DR - Get Back to Stable Rod-Only Code

```bash
git checkout main
```

That's it! You're back to the stable, tested rod simulation with all improvements.

---

## Current Situation

**Date**: November 23, 2025

You are on branch: `feature/sphere-shape-support`  
This branch contains design documents for adding sphere support.

The stable, working rod simulation is on: `main`

---

## What's on Main Branch (Stable) ✅

**Commit**: `8ce1cb6`

**Features**:
- ✅ Dynamic rod count formula: N = 6 × L³ × AR / l³
- ✅ Circular/ring COM calculation (no wrapping issues)
- ✅ Analysis tool: `parametric_study/quick_analyze_run.py`
- ✅ All tested and working

**Last commit message**:
```
feat: Add dynamic rod count formula, circular COM method, and analysis tools
```

---

## What's on Feature Branch (Current) 📋

**Branch**: `feature/sphere-shape-support`  
**Commits**: `6702939`, `d886c26`

**Contents**:
- 📋 Design document: `docs/sphere_implementation_design.md`
- 📋 Checkpoint: `docs/development_checkpoint.md`
- ⚠️ NO CODE CHANGES YET (only documentation)

---

## Rollback Scenarios

### Scenario 1: "I want to work on rods, forget about spheres for now"

```bash
# Switch back to stable rod code
git checkout main

# Optional: Delete the sphere branch if you're sure
git branch -D feature/sphere-shape-support
```

**Result**: You're back to stable rod simulations. Sphere work is deleted.

---

### Scenario 2: "I want to work on rods, but keep sphere branch for later"

```bash
# Switch to stable rod code
git checkout main

# Later, when ready to work on spheres again:
git checkout feature/sphere-shape-support
```

**Result**: You can switch between rod work and sphere design anytime.

---

### Scenario 3: "Something broke after sphere implementation, URGENT FIX"

```bash
# Immediately get back to working code
git checkout main

# Inspect what went wrong on feature branch
git checkout feature/sphere-shape-support
git log --oneline -5
git diff main

# If you want to undo recent commits on feature branch:
git reset --hard 8ce1cb6  # Goes back to main's state
```

**Result**: Back to safety, can investigate the problem later.

---

### Scenario 4: "I want to keep sphere design docs but not the branch"

```bash
# Switch to main
git checkout main

# Copy design docs from feature branch
git checkout feature/sphere-shape-support -- docs/sphere_implementation_design.md
git checkout feature/sphere-shape-support -- docs/development_checkpoint.md

# Commit them to main
git add docs/
git commit -m "docs: Add sphere design for future reference"

# Delete feature branch
git branch -D feature/sphere-shape-support
```

**Result**: Design docs preserved, branch deleted.

---

## Quick Commands Reference

```bash
# See all branches
git branch -a

# See current branch
git branch --show-current

# See commit history
git log --oneline --graph -10

# Switch to main (stable)
git checkout main

# Switch to feature branch
git checkout feature/sphere-shape-support

# See what changed between branches
git diff main feature/sphere-shape-support

# Delete feature branch (only works if you're NOT on it)
git branch -D feature/sphere-shape-support
```

---

## Run Stable Rod Simulation (from Main)

```bash
# Make sure you're on main
git checkout main

# Build
cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j8

# Run test
./rigidbody_viewer_3d --scene ../assets/scenes/confined_n2.json --headless --steps 1000

# Run analysis
cd ../parametric_study
python quick_analyze_run.py /path/to/run/directory
```

---

## Contact Information for Future You

**If you forget what's safe**:
- Main branch = ✅ Safe, tested, stable
- Feature branch = 📋 Design only, no code changes yet

**Stable commit to remember**: `8ce1cb6` on `main`

**What NOT to do**:
- ❌ Don't merge feature branch to main without testing
- ❌ Don't delete main branch (obviously!)
- ❌ Don't push feature branch to origin if you want to keep it private

**What's safe to do**:
- ✅ Switch between branches anytime
- ✅ Delete feature branch if you abandon sphere work
- ✅ Make new feature branches from main for other experiments
- ✅ Push main to origin (it's stable!)

---

## Verification After Rollback

After switching to main, verify everything works:

```bash
# Check you're on main
git branch --show-current
# Should show: main

# Check last commit
git log --oneline -1
# Should show: 8ce1cb6 feat: Add dynamic rod count formula...

# Quick smoke test
cd build-debug
./rigidbody_viewer_3d --scene ../assets/scenes/confined_n2.json --headless --steps 100
echo $?  # Should print: 0 (success)
```

If all those work, you're back to stable state! ✅

---

## Help! I'm Lost!

If you're confused about where you are:

```bash
# Where am I?
pwd
git branch --show-current

# What files changed?
git status

# What did I do recently?
git log --oneline -5

# Just get me back to safety!
git checkout main
git reset --hard origin/main  # If main got messed up too
```

---

**Remember**: Main branch is your friend. When in doubt, `git checkout main`! 🎯
