# SOP: Post-Compaction Re-Orientation — LLMs.jl Instance

This is the re-orientation checklist for the Claude Code instance working on `~/src/LLMs` (xukai92/LLMs.jl). Run after `/compact`, fresh session start, or long idle.

**Principle:** Commands, not knowledge. Knowledge is in `CLAUDE.md`, `memory/*.md`, and the codebase. SOP only reveals state that changes between sessions.

**Read first (if just compacted):** Any pre-compact snapshot that exists (Step 0). Then this SOP. Don't re-read CLAUDE.md or memory files — they're auto-loaded.

## Step 0: Read the pre-compact snapshot (if it exists)

```bash
ls -lt /tmp/pre-compact-llms-*.md 2>/dev/null | head -3
```

If there's a recent one, read it. It has in-flight non-reconstructable state captured right before compact. Trust it more than the compaction summary but less than live state.

## Step 1: Git and Branch State

```bash
# What branch are we on and what's uncommitted?
cd ~/src/LLMs && git branch --show-current && git status --short

# Recent commits on this branch (what did we just do?)
git log --oneline -8

# How far ahead of main?
git log --oneline main..HEAD 2>/dev/null

# Any open PRs from this branch?
gh pr list --state open --head "$(git branch --show-current)" 2>/dev/null
```

**What to look for:**
- **Uncommitted changes** → we were probably mid-edit. Read the diff before doing anything.
- **Commits ahead of main** → work in progress on a feature branch. Check if there's a PR.
- **On main** → between tasks. Check open issues for next work.

## Step 2: Open Issues and PR State

```bash
# Open issues (our roadmap)
gh issue list --state open --limit 15

# Open PRs (in-flight work)
gh pr list --state open

# Recently closed (what just shipped)
gh pr list --state merged --limit 5
```

**What to look for:**
- Which issue were we working on? Cross-ref with branch name and recent commits.
- Any PR comments or review requests pending?
- Recently merged PRs tell us what's on main now.

## Step 3: Benchmark Baseline Awareness

Don't re-run benchmarks (they take minutes). Instead, check what numbers we last recorded:

```bash
# README has the latest benchmark tables
head -100 ~/src/LLMs/README.md

# Check if benchmark results are in recent commit messages
git log --oneline -15 | grep -i bench
```

**What to look for:**
- Current FP16 and quantized matmul performance vs MLX
- End-to-end throughput numbers (tok/s)
- Which batch sizes have gaps (B>=256 for FP16, B>=16 for quantized)

## Step 4: Key Files Quick Scan

```bash
# Main module — what's currently included?
head -40 ~/src/LLMs/src/LLMs.jl

# Check for any research/exploration files in progress
ls ~/src/LLMs/src/metal_graph.jl ~/src/LLMs/src/mpsgraph_exploration.jl 2>/dev/null
```

## Step 5: Cross-Reference Compaction Summary

If this session was compacted:

1. Read the compaction summary at the top of context.
2. Read pre-compact snapshot from Step 0 if it exists.
3. For each mentioned item (issue number, kernel approach, benchmark finding, design decision), verify against live state from Steps 1-4.
4. **Trust order:** live state > pre-compact snapshot > compaction summary.
5. If snapshot mentions an in-flight approach or framing not in the summary, it was captured fresh — trust it.

## Step 6: Wait for Kai

After Steps 0-5, you have situational awareness. **Do NOT proactively start work.** Wait for Kai's next message. If there's an obvious half-finished edit (uncommitted changes mid-function), note it but don't complete it without asking.

## What NOT to do

- Don't re-read CLAUDE.md or memory files — they're auto-loaded into context.
- Don't re-run benchmarks "just to check" — they're slow and noisy. Use recorded numbers.
- Don't start a new Julia process — startup is expensive (~30s with compilation).
- Don't push to remote without asking. Don't create PRs without asking.
- Don't re-derive things that are in git history (use `git log`, `git show`, `gh issue view`).
- Don't modify GPUCompiler.jl or Metal.jl source directly — we use monkey-patches (`gpucompiler_patch.jl`, `metal_simd_patch.jl`).
- Don't poll or loop. Run checks once, then wait.
- Don't start optimizing a kernel without first checking which issue/branch it belongs to.

## When to re-run this SOP

- After `/compact`
- After a fresh session start
- If Kai says "where are we?" or "re-sync"
- After a long idle where state may have drifted

## Feedback

After post-compact recovery, note which steps helped, what was missing, and what was wrong. DURING normal work, whenever you think "I wish the SOP had told me X" or "step N was exactly right" or "that path is stale", append a dated bullet here immediately.

- *(no entries yet — append below as they arise)*
