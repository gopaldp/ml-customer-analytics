---
description: |
  Generates documentation that is missing and updates documentation that is
  out of date, so README.md and docs/ match what the code actually does.
  Opens one draft PR. Manual trigger only, to keep API usage under control.
  Repo-specific context comes from .github/copilot-instructions.md.

on:
  workflow_dispatch:

permissions:
  contents: read
  issues: read
  pull-requests: read

# Gemini CLI, authenticated with the GEMINI_API_KEY repo secret.
# Copilot is not used: the Copilot Student token only accepts automatic
# model choice, which fails behind the gh-aw API proxy.
#
# Gemini CLI is pinned to 0.43.0: newer versions exit with
# "Invalid auth method selected" (code 41) behind the gh-aw API proxy.
# See https://github.com/github/gh-aw/issues/58445. Unpin once fixed.
#
# The model is pinned to a Flash-Lite model on the Gemini API free tier.
# With the default "auto" routing, the CLI picks gemini-3.1-pro (free-tier
# limit 0) and routes with gemini-2.5-flash-lite (retired for new users).
# gemini-3.5-flash returned 503 "high demand" on every retry; switch back
# to it for more detailed docs once it is reliably available.
engine:
  id: gemini
  version: "0.43.0"
  model: gemini-3.5-flash-lite

network: defaults

tools:
  github:
    toolsets: [default]
  edit:
  # "safeoutputs" is listed explicitly: gh-aw adds "safeoutputs:*", but
  # Gemini CLI 0.43 does not match the ":*" form, so the PR call was denied.
  bash: ["ls", "cat", "find", "grep", "head", "tail", "wc", "git ls-files", "git log", "git diff", "safeoutputs"]

safe-outputs:
  create-pull-request:
    title-prefix: "[docs] "
    labels: [documentation]
    draft: true
  # AI threat detection only runs on copilot, claude or codex. Copilot fails
  # with this account's token and the others need paid API keys, so skip the
  # AI scan. Output is a docs-only draft PR that a human reviews before merge.
  threat-detection:
    engine: false

timeout-minutes: 20
---

# Generate or Update Docs

You are the documentation maintainer for `${{ github.repository }}`. Your job is
to make the repository's documentation complete and accurate: create what is
missing, update what is outdated, and keep what is already correct.

## 1. Understand the repository

- Read `.github/copilot-instructions.md` first, if it exists. It describes the
  project and what to ignore.
- List tracked files with `git ls-files`. Ignore virtual environments (`venv/`,
  `.venv/`), `node_modules/`, data files, model binaries, notebook checkpoints
  and generated output.
- Identify the language and stack, the entry points (scripts, apps, servers,
  CLIs), how dependencies are installed, how to run the project, how tests run,
  and any container or deployment setup (Dockerfile, compose, CI workflows).

## 2. Inventory the existing documentation

Find `README.md`, everything under `docs/`, and any other Markdown guides in
the repository root (for example `CONTRIBUTING.md` or `*_GUIDE.md`). For each
one, note what it covers and which statements are wrong, outdated or missing
compared with the code.

## 3. Create or update — per file

Apply this rule to every documentation file:

- **The file exists:** edit it in place. Keep content that is still accurate,
  including its structure, tone, links, badges and demo URLs. Correct anything
  the code contradicts, and add missing sections. Do not rewrite a file from
  scratch when targeted edits are enough.
- **The file does not exist:** create it.
- **Never delete** an existing documentation file. If one is obsolete, say so in
  the pull request instead.

### README.md (always handled)

Create it if missing, otherwise update it. It must contain:

1. Project name and a short description of what it does
2. Key features, based on the code
3. Tech stack
4. Prerequisites (language and runtime versions, tools)
5. Installation
6. Usage: how to run every entry point, with exact commands
7. Configuration: environment variables and config files, using
   `.env.example` or similar if present
8. Project structure: a short tree of the important folders and files
9. Testing, if tests exist
10. Docker or deployment, if present
11. Links to the files in `docs/`

### docs/ (create or update as needed)

- `docs/architecture.md`: components and modules, how data and control flow
  between them, and what each main module is responsible for.
- `docs/setup.md`: detailed local setup, dev container or Docker setup, and
  troubleshooting for common problems.

If the repository already has docs covering these topics under other names,
update those files instead of creating duplicates, and link to them from the
README.

## Rules

- Change documentation only: `README.md`, files under `docs/`, and existing
  Markdown guides. Never modify code, notebooks, dependencies, configuration
  or CI files.
- Base every statement on the code. Do not invent features, metrics, commands,
  environment variables or file names. Where something is unclear, write
  `TODO: confirm …` instead of guessing.
- Fix broken character encoding in existing docs (for example `ðŸ“Š` instead
  of an emoji).
- Use plain, concise English and GitHub-flavoured Markdown. Use relative links.
- If all documentation is already accurate and complete, change nothing and
  do not open a pull request.

## Git and branches

Do not run git commands to create branches, switch branches, stage or commit.
They are blocked and will fail. Edit the files in place with your file tools,
then create the pull request once: it creates the branch, the commit and the
pull request for you. Read-only git commands (`git ls-files`, `git log`,
`git diff`) are allowed.

## Pull request

If `create_pull_request` is not available as a direct tool, use the
`safeoutputs` command in the shell instead, for example
`safeoutputs create_pull_request '<json>'` (run `safeoutputs --help` if
unsure). Call it exactly once, then stop.

Open one draft pull request with a short descriptive title. In the body:

- list each file as **created** or **updated**, with a one-line summary;
- list every `TODO` you left;
- list any docs you think are obsolete but did not delete.
