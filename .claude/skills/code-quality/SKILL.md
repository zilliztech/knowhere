---
name: code-quality
description: Use when checking code style, running pre-commit hooks, or before committing changes to ensure compliance with Google style guide
---

# Code Quality

## Code Style

- **Standard**: Google C++ Style Guide
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

```bash
# 1. Make changes
# 2. Run pre-commit on changed files
pre-commit run --files <changed-files>

# 3. Stage and commit
git add .
git commit -m "message"
```

## Common Issues

| Issue | Solution |
|-------|----------|
| Formatting errors | Run `clang-format -i <file>` or let pre-commit fix it |
| Line too long | Break line at logical points, max 120 chars |
| Include order | Follow Google style: related header, C system, C++ standard, other libs, project headers |

---

> Pre-commit hooks run automatically on commit. Fix any failures before pushing.
