# AGENTS.md — Guidelines for AI agents working on pyobjcryst

## Repository overview

`pyobjcryst` is a Python binding (via Boost.Python) of the ObjCryst++ C++
library. It is built and maintained with
[scikit-package](https://scikit-package.github.io/scikit-package/), which
provides the release scripts, CI workflows, and changelog tooling used here.

The C++ bindings live under `src/extensions/`; pure-Python wrappers and
utilities are in `src/pyobjcryst/`. Tests live in `tests/`.

## Dependency chain

```
vincefn/objcryst   ← real upstream (C++ library, GUI, Fox)
      ↓
diffpy/libobjcryst ← repackages ObjCryst++ as a shared library for conda/pip
      ↓
diffpy/pyobjcryst  ← Python bindings (this repo)
```

**Any C++ feature or fix must flow through the full chain before it can land
here:**

1. Implement or fix it in [vincefn/objcryst](https://github.com/vincefn/objcryst).
2. Pick it up in [diffpy/libobjcryst](https://github.com/diffpy/libobjcryst)
   (libobjcryst is a convenience repackaging — updating the vendored sources
   and cutting a release is usually all that is needed).
3. Only then expose or rely on it in pyobjcryst.

If a PR here is blocked on an upstream change, note the upstream issue/PR in
the description and news item.

## Branching and commits

- Always branch from `main`.
- Use [Conventional Commits](https://www.conventionalcommits.org/) prefixes:
  `fix:`, `feat:`, `docs:`, `refactor:`, `test:`, `chore:`, etc.
- Keep commits focused and atomic.
- Make sure `git config user.email` is a verified address on the author's
  GitHub account so commits are attributed correctly.

## Coding standards

1. Where possible, make the code conform to Billinge group coding
   standards. These can be found in the documentation of
   [scikit-package](https://scikit-package.github.io/scikit-package/)

## Before opening a PR

1. **News item** — every PR that changes user-visible behaviour must add a
   file `news/<PR-number>.rst` following `news/TEMPLATE.rst` (sections:
   `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`; leave
   unused sections as `* <news item>`).
   The `check-news-item` CI job (provided by scikit-package release scripts)
   enforces this.
2. **Do not edit `CHANGELOG.rst` directly** — it is auto-generated from the
   `news/` items at release time by the scikit-package tooling.
3. Run the test suite locally before pushing if possible: `pytest tests/`

## CI checks on PRs

| Workflow          | Trigger  | What it does                                        |
| ----------------- | -------- | --------------------------------------------------- |
| `tests-on-pr`     | every PR | Builds the C extension and runs the full test suite |
| `check-news-item` | every PR | Verifies a `news/<PR-number>.rst` file exists       |

Both workflows are provided by `scikit-package/release-scripts`.

## Merge policy — human review required

**PRs must never be auto-merged.** Every PR requires at least one human
review and approval before merging, regardless of CI status.
Do not use `gh pr merge --auto` or any equivalent automated merge on this
repository.

## What agents should NOT do

- Do not force-push to `main`.
- Do not bypass CI or skip the news-item requirement.
- Do not merge PRs autonomously — always leave them open for human review.
- Do not commit secrets, credentials, or large binary files.
- Do not expose a new C++ feature in pyobjcryst before it is available in a
  released version of diffpy/libobjcryst.
