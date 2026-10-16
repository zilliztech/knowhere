---
name: code-quality
description: Use when checking code style, running pre-commit hooks, or before committing changes to ensure compliance with Google style guide
---

# Code Quality

## Code Style

- **Base style**: Google C++ Style Guide with repository overrides
- **Line limit**: 120 characters
- **Indent**: 4 spaces
- **Config**: See `.clang-format` in repo root

## Pre-commit Hooks

### Setup

```bash
pip3 install pre-commit
pre-commit install --hook-type pre-commit --hook-type pre-push
```

### Usage

```bash
# Run on staged files
pre-commit run

# Run on specific files
pre-commit run --files src/index/hnsw/hnsw.cc include/knowhere/index/index.h

# Run on all files
pre-commit run --all-files
```

## Commit Workflow

When a commit is authorized, stage only the intended files and include the DCO sign-off. Replace
`<changed-files>` with explicit paths and follow the [git-workflow skill](../git-workflow/SKILL.md) for
commit and PR titles. If hooks change files or fail, inspect the changes, fix failures, and rerun the
checks before committing.

```bash
# Review and validate the intended changes
git diff --check
pre-commit run --files <changed-files>

# Review hook edits, then stage and inspect the commit contents
git add -- <changed-files>
git diff --cached
git commit -s -m "type: description"
```

## Common Issues

| Issue | Solution |
|-------|----------|
| Formatting errors | Run `clang-format -i <file>` or let pre-commit fix it |
| Line too long | Break line at logical points, max 120 chars |
| Include order | Follow Google style: related header, C system, C++ standard, other libs, project headers |

---

Installed hooks run automatically at their configured commit/push stages. Fix failures before pushing;
manual checks remain available when hooks are not installed.
