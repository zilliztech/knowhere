# Knowhere

Knowhere is the C++ vector search library used by Milvus. Public interfaces live in
`include/knowhere/`, index implementations in `src/index/`, and unit tests in `tests/ut/`.

## Shared context

Maintain project instructions in this file and task-specific guidance in `.agents/skills/`.
`CLAUDE.md` imports this file. Read task-specific guidance through the skill links below,
and edit the shared files so agents receive the same guidance.

Read the relevant skill before working on its topic:

| Topic | Skill |
|-------|-------|
| Source layout, index implementations, dependencies | [architecture](.agents/skills/architecture/SKILL.md) |
| Build configuration and compiler errors | [building](.agents/skills/building/SKILL.md) |
| Running and adding tests | [testing](.agents/skills/testing/SKILL.md) |
| Index performance optimization and benchmarks | [index-optimization](.agents/skills/index-optimization/SKILL.md) |
| Formatting and static analysis | [code-quality](.agents/skills/code-quality/SKILL.md) |
| Commits, PR titles, DCO, and labels | [git-workflow](.agents/skills/git-workflow/SKILL.md) |

## Development

- Run commands from the repository root. Use `make help` for build options.
- Build CPU unit tests with `make WITH_UT=True`; the default executable is
  `build/Release/tests/ut/knowhere_tests`. Select relevant tests by name or Catch2 tag.
- Follow `.clang-format`: Google C++ style, 120-column limit, four-space indentation.
  Run `pre-commit run --files <changed-files>` before committing.
- Verify commands and supported features against the current `Makefile`, `conanfile.py`,
  CMake files, index registrations, and tests. Update skill guidance when those sources change.
- Keep changes scoped and preserve unrelated local work. Report the checks actually run
  and any validation still pending. Follow the user's requested boundary for committing,
  pushing, and creating PRs.

## Maintaining the shared layout

Keep `CLAUDE.md` as an `@AGENTS.md` import. Add new skills under `.agents/skills/`
and update the table above. Keep relative references inside a skill with that skill.
