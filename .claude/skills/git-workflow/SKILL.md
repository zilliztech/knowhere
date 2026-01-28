---
name: git-workflow
description: Use when committing and pushing code changes, ensures DCO compliance and pre-commit validation
---

# Git Workflow

## Before Committing

Always run pre-commit validation:

```bash
# Validate specific files
pre-commit run --files <changed-files>

# Or validate all staged files
pre-commit run
```

Fix any issues before proceeding.

## DCO (Developer Certificate of Origin)

All commits must include a Signed-off-by line for DCO compliance:

```bash
git commit -s -m "commit message"
```

Or add manually:
```
Signed-off-by: Your Name <your.email@example.com>
```

## Commit Workflow

1. **Stage changes**
   ```bash
   git add <files>
   ```

2. **Run pre-commit**
   ```bash
   pre-commit run
   ```

3. **Fix any issues** and re-stage if needed

4. **Commit with DCO sign-off**
   ```bash
   git commit -s -m "type: description"
   ```

## Commit Message Format

```
type: short description

Optional longer description.

Signed-off-by: Name <email>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

## Push

```bash
git push origin <branch>
```

## Quick Reference

| Task | Command |
|------|---------|
| Pre-commit check | `pre-commit run` |
| Commit with DCO | `git commit -s -m "message"` |
| Amend with DCO | `git commit --amend -s` |
| Push | `git push origin <branch>` |
