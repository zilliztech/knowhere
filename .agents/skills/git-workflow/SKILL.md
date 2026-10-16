---
name: git-workflow
description: Use when preparing Knowhere commits or pull requests, choosing titles and labels, or checking DCO and submission readiness.
---

# Git Workflow

## Titles

Choose a prefix that describes the change. Common Knowhere prefixes include:

| Prefix | Purpose |
|--------|---------|
| `fix:` | Correct a bug |
| `feat:` | Add a feature |
| `enhance:` | Improve existing behavior or update a dependency version/reference |
| `perf:` | Improve performance |

This is not an exhaustive list or a claim about CI title validation. Follow explicit
maintainer requirements for other change types. Use `enhance:` for dependency updates,
for example `enhance: update dependency to <version-or-sha>`; editing a CMake file does
not make the change a `build:` or `chore:` PR.

Use `<type>: <description>` for `main` and `<type>: [<branch>] <description>` for a
release branch. Keep the commit subject and PR title consistent unless the requested
workflow requires otherwise. A previous merged title is a reference, not a rule that
overrides current project conventions.

## Prepare and verify

1. Confirm the repository, target branch, and intended remote with `git status` and
   `git remote -v`. Keep unrelated changes out of the commit.
2. Inspect `git diff` and run `git diff --check`.
3. Run `pre-commit run --files <changed-files>` and the relevant build/tests described
   in the [building](../building/SKILL.md) and [testing](../testing/SKILL.md) skills.
   Documentation-only changes need formatting, link, and command checks rather than a C++ rebuild.
4. Prepare a PR description explaining the problem, resulting behavior, and actual
   validation results. Check `.github/mergify.yml` for current label and issue rules.

Select one `kind/*` label by the substance of the change, independently of the title
prefix. The current categories are `kind/bug`, `kind/improvement`, `kind/enhancement`,
and `kind/feature`. Small changes (fewer than 20 lines) without functionality changes
can use `kind/improvement`; other categories need a related issue in the description, such as
`issue: #1234`. Do not classify a functional change as an improvement solely by its line count.

## Commit and submit

Follow the user's submission instructions. If asked to prepare changes for confirmation,
leave the commit, push, issue creation, and PR creation pending. A request to submit a PR
authorizes its normal preparation and submission steps.

When submission is authorized, verify the author identity and add a DCO sign-off:

```bash
git config user.name
git config user.email
git add <changed-files>
git diff --cached
git commit -s -m "enhance: describe the change"
```

The `Signed-off-by` name and email must match the committing identity. Push the intended
branch to the confirmed remote, create the PR against the requested base, and verify
its title, diff, labels, DCO result, and CI status. Do not assume `origin` is a personal fork.
