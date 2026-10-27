# Claude Code workflow for this project: best practices, tools, skills

*Compiled 2026-10-27 for the edge-cloud VLA project (slow cloud VLM/VLA conditioning a fast on-robot policy). Scope: literature research first, then ML experiments (small robot policies / VLA models, sim experiments, ablations).*

**How to read this document.** Every claim has a source link. Verification status:

- **[verified]**: I fetched the page or queried the GitHub API on 2026-10-27 and the repo or page exists as described. GitHub star counts are from that date.
- **[secondary]**: the only source is a search-result snippet or a third-party write-up. I did not open the primary source.
- **[unverified]**: I could not confirm it. Treat it as a lead, not a fact.

---

## 0. TL;DR: top recommendations

1. **Keep CLAUDE.md short (under 200 lines).** Put in it only what Claude can't infer from the repo: the research questions, the exact commands for train/eval/test, the citation rules, and the experiment-logging rules. Move procedures into skills and path-scoped rules ([best practices](https://code.claude.com/docs/en/best-practices), [memory](https://code.claude.com/docs/en/memory)).
2. **Treat every reference as unverified until a script resolves it.** Each entry in `references.bib` needs an arXiv ID or DOI that resolves through the arXiv, Crossref, OpenAlex or Semantic Scholar APIs. Enforce this with a hook or check script, not a prompt. Hallucinated citations have reached NeurIPS/ICLR/ICML proceedings ([arXiv:2607.00738](https://arxiv.org/abs/2607.00738)).
3. **Keep a progress file plus git as the agent's lab notebook.** Use a `CHANGELOG.md`/journal that records current status, failed approaches with the reason they failed, and result tables. Commit after every meaningful unit of work. Anthropic's own long-running scientific-computing guide uses exactly this pattern ([Anthropic, 2026-03-23](https://www.anthropic.com/research/long-running-claude)).
4. **Give every autonomous loop a test oracle, a fixed budget, and a protected evaluation.** Following [autoresearch](https://github.com/karpathy/autoresearch), the agent edits only the training code, under a fixed time budget, against a metric it cannot touch. Agents reward-hack ML tasks often: 57% of runs in [BaitBench](https://arxiv.org/abs/2608.30724).
5. **Literature tools:** use ARS (installed) for the scoping-review process and integrity gates. Add `arxiv-mcp-server` (LaTeX section reads, BibTeX from arXiv metadata), the Hugging Face MCP (papers, models, LeRobot datasets), and optionally Zotero MCP. Papers with Code no longer exists (sunset July 2025).
6. **Experiment tools:** W&B MCP (official) or MLflow MCP (set `MLFLOW_MCP_TOOLS=ml` or `all`, because the default exposes only GenAI traces). Run long jobs in tmux/Slurm and poll with `/loop` or Monitor. Use `/goal` or the official `ralph-loop` plugin so a run doesn't stop before it meets an explicit success criterion.

---

## 1. Official Anthropic guidance

### 1.1 Claude Code best practices (current docs)

Source: [code.claude.com/docs/en/best-practices](https://code.claude.com/docs/en/best-practices) [verified]

- **Defaults are good; set up only what Claude can't see.** New sessions start in auto mode, delegate to subagents and auto-compact. Setup pays off for knowledge or constraints that aren't in the code ([best practices](https://code.claude.com/docs/en/best-practices)).
- **Deterministic vs. non-deterministic instructions.** Prompts, CLAUDE.md and skills are *advisory*: the model usually follows them, with no guarantee. Rules that can't have exceptions belong in **hooks** or **permission deny rules** ([best practices](https://code.claude.com/docs/en/best-practices); [memory docs](https://code.claude.com/docs/en/memory): "Claude treats them as context, not enforced configuration").
- **Keep CLAUDE.md short.** It loads into every session. Include: how to verify work (exact test/lint commands), misleading parts of the code, decisions not recorded in the repo, and anything you've had to correct twice. Target under 200 lines. `/init` drafts one, `/doctor` suggests cuts, and `/doctor prompt-audit` flags stale or contradictory rules ([best practices](https://code.claude.com/docs/en/best-practices), [memory](https://code.claude.com/docs/en/memory)).
- **Leave out what Claude can read itself.** Architecture and file layout don't belong in CLAUDE.md, and neither do ALL-CAPS emphasis or "think harder" lines. Use `/effort` or `ultrathink` instead ([best practices, "What you can stop doing"](https://code.claude.com/docs/en/best-practices)).
- **Path-scoped rules.** `.claude/rules/*.md` with a `paths:` frontmatter field load only when Claude works on matching files. Use these for `experiments/**` or `literature/**` conventions ([memory](https://code.claude.com/docs/en/memory)).
- **Imports.** `@path/to/file` in CLAUDE.md inlines another file, up to four hops deep. Imports don't reduce the context cost ([memory](https://code.claude.com/docs/en/memory)).
- **Auto memory.** Claude writes its own notes per repository, and the first 200 lines or 25 KB load each session ([memory](https://code.claude.com/docs/en/memory)).
- **Skills for procedures.** `.claude/skills/<name>/SKILL.md`. Only the name and description sit in context until the skill is used. Add `disable-model-invocation: true` to skills with side effects, such as launching a GPU sweep, so that only you can start them ([best practices](https://code.claude.com/docs/en/best-practices), [skills](https://code.claude.com/docs/en/skills)).
- **Give Claude a check to run.** Options: a test, an exit code, a script that diffs output against a fixture. Claude can loop on it from a prompt ("keep going until it passes"). `/goal <condition>` adds a separate evaluator model after every turn. A **Stop hook** that exits with code 2 forces Claude back to work, and the harness enforces it. `/code-review` gets a reviewer that never saw your reasoning ([best practices](https://code.claude.com/docs/en/best-practices)).
- **Subagents** are good for parallel work, for keeping noisy output out of the main context, and for independent review. Delegating a task does *not* make the answer better. A fresh reviewer does help. Use `model: haiku` for simple jobs, and `tools: Read, Grep, Glob` for a read-only reviewer that the harness enforces ([best practices](https://code.claude.com/docs/en/best-practices), [sub-agents](https://code.claude.com/docs/en/sub-agents)).
- **Steering:** `Esc` interrupts, `/rewind` rolls back code and/or conversation, and `/clear` between unrelated tasks. After two failed corrections, `/clear` plus one better prompt beats a third correction ([best practices](https://code.claude.com/docs/en/best-practices)).

### 1.2 Plan mode and permission modes

Source: [permission-modes](https://code.claude.com/docs/en/permission-modes) [verified]

- Enter plan mode with **Shift+Tab**, `/plan` before a prompt, or `claude --permission-mode plan`. Claude reads files, runs exploratory commands and writes a plan, but can't edit source until you approve. `Ctrl+G` edits the plan in your editor.
- Modes: `default` (manual), `acceptEdits`, `plan`, `auto` (a classifier reviews risky actions), `dontAsk` (only pre-approved tools run, which suits scripts), and `bypassPermissions` (only in disposable containers or VMs).
- **For this project:** use plan mode before any new experiment design or training-code refactor. A wrong design costs GPU-hours.

### 1.3 Unattended runs: headless, `/loop`, `/goal`, routines, worktrees

Five questions to answer before any unattended run, quoted from the [best practices](https://code.claude.com/docs/en/best-practices) "Let Claude run without you" section [verified]:

1. Does it know what done looks like?
2. Can it check its own work?
3. What is it allowed to touch?
4. What stops it (`--max-turns`, `--max-budget-usd`)?
5. Where does the result go, and who signs off?

| Mechanism | What it is | Use here | Source |
|---|---|---|---|
| `claude -p` (headless) | One-shot non-interactive run. Flags: `--bare` (skip CLAUDE.md, hooks, plugins for reproducibility), `--allowedTools`, `--permission-mode dontAsk`, `--max-turns`, `--output-format json/stream-json` | Batch jobs: summarise each finished run's logs into the results table; per-paper extraction over a list of arXiv IDs | [headless](https://code.claude.com/docs/en/headless), [best practices](https://code.claude.com/docs/en/best-practices) |
| `/loop [interval] <prompt>` | Session-scoped repeat (min. 1 min). With no interval, Claude self-paces between 1 min and 1 h. Recurring tasks expire after 7 days; fires only while the session is open; `.claude/loop.md` sets the default prompt | Poll a training run: "every 30m check `squeue` and the W&B run; if diverged, kill and log it in the journal" | [scheduled-tasks](https://code.claude.com/docs/en/scheduled-tasks) |
| Monitor tool | Streams a background script's output lines back, so there's no polling | `tail -f` a training log and react to NaNs or eval results | [scheduled-tasks](https://code.claude.com/docs/en/scheduled-tasks) |
| `/goal <condition>` | Wraps a prompt-based Stop hook. A small model checks the condition after each turn. The evaluator can't run tools, so the condition must be demonstrable in the transcript. Add "or stop after N turns" | "all unit tests in tests/ pass and the smoke-train run reaches loss < X" | [goal](https://code.claude.com/docs/en/goal) |
| Cloud scheduled tasks (routines, `/schedule`) | Run in the cloud on cron, API or GitHub events. Fresh clone, no local files, min. 1 h interval | Weekly "new arXiv papers on RQ1–RQ5" digest into `literature/inbox.md` via PR | [routines](https://code.claude.com/docs/en/routines), [scheduled-tasks](https://code.claude.com/docs/en/scheduled-tasks) |
| Desktop scheduled tasks | Local, persistent, access to local files | Nightly local eval sweep summary | [desktop-scheduled-tasks](https://code.claude.com/docs/en/desktop-scheduled-tasks) |
| Worktrees (`claude -w name`) | Separate checkout per parallel session. `.worktreeinclude` copies gitignored files such as `.env`. Note: dependencies and data are not present in a new worktree | Run two ablation branches in parallel without edit collisions; keep `CLAUDE.md` install instructions so new worktrees bootstrap | [worktrees](https://code.claude.com/docs/en/worktrees), [best practices](https://code.claude.com/docs/en/best-practices) |
| Dynamic workflows (`ultracode`, `/workflows`) | A JS script orchestrates up to 16 concurrent subagents (1,000 per run). Resumable. Built-in `/deep-research` cross-checks claims across sources and filters out ones that don't survive. No mid-run user input | Fan-out literature extraction (one agent per paper → structured JSON), adversarial verification of findings, ablation-result audits | [workflows](https://code.claude.com/docs/en/workflows) |

Cost note: `/loop`, goal evaluations, subagents and workflows all consume tokens while you're away. Check `/usage`. Workflows warn above 25 agents or 1.5M projected tokens ([best practices](https://code.claude.com/docs/en/best-practices), [workflows](https://code.claude.com/docs/en/workflows)).

### 1.4 Hooks

Source: [hooks guide](https://code.claude.com/docs/en/hooks-guide), [hooks reference](https://code.claude.com/docs/en/hooks) [verified via the best-practices page links]

Hooks run at fixed points (PreToolUse, PostToolUse, Stop, SessionStart, …). A hook can be a shell command, an HTTP POST, an MCP tool call, or a prompt/agent hook ([best practices](https://code.claude.com/docs/en/best-practices)). Project hooks in `.claude/settings.json` are shared with the repo ([best practices](https://code.claude.com/docs/en/best-practices)). Useful hooks for this project:

- **PostToolUse on edits to `literature/references.bib`:** run `scripts/verify_bib.py` and feed failures back.
- **PreToolUse deny on edits to `eval/` or the held-out split** during autonomous training loops, so the agent can't move the goalposts. This mirrors autoresearch's "only `train.py` is editable" design.
- **Stop hook:** refuse to end the turn while `experiments/registry.csv` has a run marked `running` with no journal entry.

### 1.5 Anthropic material on Claude for research and ML engineering

- **"Long-running Claude for scientific computing"**, Siddharth Mishra-Sharma (Discovery team), 2026-03-23: [anthropic.com/research/long-running-claude](https://www.anthropic.com/research/long-running-claude) [verified; note the URL is lowercase]. This is the most directly applicable official guide. Key practices:
  - Spend most of your time with Claude iterating on a **CLAUDE.md plan** with deliverables and an explicit accuracy target. Claude may update it as it learns.
  - Keep a **progress file (`CHANGELOG.md`)** as "lab notes": current status, completed tasks, **failed approaches and why**, accuracy tables at checkpoints, known limitations. "Without them, successive sessions will re-attempt the same dead ends."
  - Use a **test oracle**: a reference implementation, a quantifiable objective, or a test suite. Have the agent expand the tests to prevent regressions.
  - Use **git as coordination**: "Commit and push after every meaningful unit of work. Run `pytest tests/ -x -q` before every commit. Never commit code that breaks existing passing tests."
  - **Execution:** run Claude Code inside **tmux on a Slurm compute node**, attach and detach to steer, and check progress via GitHub. The post includes an example `sbatch` script.
  - Use a **Ralph loop** against "agentic laziness": `/ralph-loop:ralph-loop "<task until criterion>" --max-iterations 20 --completion-promise "DONE"`.
  - Honest caveats from the post: test coverage gaps (for a while it tested only one parameter point), elementary domain mistakes, and a result that isn't production-grade.
- **"Effective harnesses for long-running agents"**, Anthropic Engineering, 2025-11-26: [anthropic.com/engineering/effective-harnesses-for-long-running-agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) [verified]. An initializer session creates `init.sh`, a progress file (`claude-progress.txt`) and a JSON feature list with pass/fail status. Each later session does one feature, tests it end-to-end, commits, and updates the progress file. Failure modes it addresses: premature "done", undocumented progress, features marked passing without testing.
- **Official `ralph-loop` plugin**: in [anthropics/claude-plugins-official/plugins/ralph-loop](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/ralph-loop) [verified]. It's implemented as a Stop hook that re-feeds the same prompt until a completion promise appears.
- **Claude Science** (launched 2026-06-30): an AI workbench with 60+ scientific databases, focused on life sciences ([MIT Technology Review](https://www.technologyreview.com/2026/06/30/1139987/claude-science-is-anthropics-newest-flagship-product/)) [secondary]. Its life-science focus makes it of limited use for robotics. ARS can be imported into it ([ARS README](https://github.com/Imbad0202/academic-research-skills)).
- **"How Anthropic teams use Claude Code"**: [claude.com/…/how-anthropic-teams-use-claude-code](https://claude.com/fr/resources/articles/how-anthropic-teams-use-claude-code) (French locale URL found) [secondary]. General patterns such as CLAUDE.md onboarding and screenshot-driven debugging.

---

## 2. Community/third-party skills, plugins and MCP servers

### 2.1 ARS — Academic Research Skills (installed here)

Repo: [github.com/Imbad0202/academic-research-skills](https://github.com/Imbad0202/academic-research-skills). Verified: ~54k stars, installed version 3.24.0, enabled in this repo's `.claude/settings.json`.

**What it offers** (from the installed README, `POSITIONING.md`, `docs/DATA_FLOWS.md`):

- **Five skills:** `deep-research` (13-agent team; modes: full, quick, paper review, `lit-review`, fact-check, 3W scan, Socratic, systematic review/PRISMA with meta-analysis), `academic-paper` (12 agents, LaTeX/DOCX/PDF, APA/Chicago/MLA/IEEE/Vancouver), `academic-paper-reviewer` (5-seat simulated panel), `academic-pipeline` (10-stage orchestrator with integrity gates), `sr-screener` (two blinded AI screeners plus an adjudicator, PRISMA 2020 counts, RIS/Zotero export).
- **Commands:** `/ars-plan`, `/ars-lit-review`, `/ars-citation-check`, `/ars-full`, `/ars-outline`, `/ars-reviewer`, `/ars-revision`, `/ars-disclosure`, `/ars-mark-read` and others.
- **Citation verification:** live calls to OpenAlex, Crossref, arXiv (≥3 s pacing) and Semantic Scholar, with a local cache. There's a deterministic citation verification gate and an opt-in claim-faithfulness audit (`ARS_CLAIM_AUDIT=1`), which fetches the cited source and checks the claim is actually supported.
- **Human-in-the-loop by design:** it stops for confirmation at every stage, and its README cites the AI Scientist's limitations as the motivation.
- **Experiment Provenance Intake:** records experiments *you* ran and audits whether manuscript claims match the declared provenance.

**Limits, stated by the project itself** (from the installed `POSITIONING.md` and `README.md`):

- "ARS never runs experiments." It can't establish that procedures were performed or that results reproduce. "a study built on fabricated data can pass every ARS gate if the fabrication is consistently reported."
- `deep-research` reports are **APA 7.0 only**, built for social-science/medical review conventions. Robotics/ML venues need BibTeX/IEEE, so plan a conversion step (via `academic-paper` format-convert, or your own `.bib` pipeline).
- The license is **CC BY-NC 4.0**, not OSI open source. Fine for private or academic use; it restricts commercial use.
- The simulated reviewer is `NOT_CALIBRATED`, and the claim-audit judge has no live calibration result yet.
- It's token-heavy: about US$3–7 for a full 15k-word pipeline, per its `docs/PERFORMANCE.md`.
- It doesn't read code repos, benchmarks or leaderboards, so pair it with the arXiv/HF MCPs below for ML-specific evidence.

**Fit for this project:** use `sr-screener` and `deep-research lit-review` for the rapid scoping review defined in `literature/PROTOCOL.md`, and `/ars-citation-check` as a second gate on `references.bib`. Don't use `academic-pipeline` full mode until there are real experiment results.

### 2.2 Literature MCP servers

| Server | Repo / endpoint | Status | Notes |
|---|---|---|---|
| **arxiv-mcp-server** (blazickjp) | [github.com/blazickjp/arxiv-mcp-server](https://github.com/blazickjp/arxiv-mcp-server) | verified, ~3.2k★, active Oct 2026 | Local stdio. Reads original LaTeX section by section, exports BibTeX from arXiv metadata (good for hallucination-free `.bib` entries), topic watches, citation graph. Install: `claude mcp add --transport stdio --scope user arxiv -- uvx arxiv-mcp-server`; a plugin with a bundled research skill is also available. Warning from its README: an unrelated npm package has the same name, so don't install with `npx`. |
| arxiv-latex-mcp | [github.com/takashiishida/arxiv-latex-mcp](https://github.com/takashiishida/arxiv-latex-mcp) | verified, ~150★ | Fetches LaTeX source for precise math |
| mcp-simple-arxiv | [github.com/andybrandt/mcp-simple-arxiv](https://github.com/andybrandt/mcp-simple-arxiv) | verified, ~200★ | Search plus abstracts |
| alphaXiv MCP | `https://api.alphaxiv.org/mcp/v1` (`claude mcp add --transport http alphaxiv https://api.alphaxiv.org/mcp/v1`) | **secondary** ([deepakness.com](https://deepakness.com/raw/arxiv-mcp/)) | Embedding/full-text search, PDF Q&A, reads the paper's GitHub repo. Hosted and third-party, *not* official arXiv. |
| paper-search-mcp | [github.com/openags/paper-search-mcp](https://github.com/openags/paper-search-mcp) | verified, ~2.9k★ | Multi-source search and download (arXiv, PubMed, bioRxiv, …) |
| Semantic Scholar MCP | [JackKuo666/semanticscholar-MCP-Server](https://github.com/JackKuo666/semanticscholar-MCP-Server) (~87★, last push Mar 2025); [xiuyechen/semantic-scholar-mcp](https://github.com/xiuyechen/semantic-scholar-mcp) (4★) | verified exist; small and possibly stale | No dominant, well-maintained option. Calling the [S2 API](https://api.semanticscholar.org) directly from a script may be more robust. |
| OpenAlex MCP | [reetp14/openalex-mcp](https://github.com/reetp14/openalex-mcp) (5★, last push Jul 2025) | verified exists; small | Same advice: the OpenAlex REST API is free and simple, so a 50-line verification script is safer than a tiny MCP. |
| Papers with Code | — | **gone** | Meta sunset PwC on 2025-07-24, and Hugging Face "Trending Papers" replaced it with much smaller task coverage ([hyper.ai](https://hyper.ai/en/news/42900), [Coursera](https://www-cloudfront-alias.coursera.org/articles/papers-with-code)) [secondary]. Any "Papers with Code MCP" will be stale. |
| **Hugging Face MCP** (official) | [github.com/huggingface/hf-mcp-server](https://github.com/huggingface/hf-mcp-server); `claude mcp add hf-mcp-server -t http "https://huggingface.co/mcp?login"` | verified, ~300★, active | Model, dataset, Space and **paper** semantic search. Relevant for finding LeRobot datasets and SmolVLA/pi0 checkpoints ([HF MCP course](https://huggingface.co/learn/mcp-course/en/unit1/hf-mcp-server)). |
| **Zotero MCP** (54yyyu) | [github.com/54yyyu/zotero-mcp](https://github.com/54yyyu/zotero-mcp) | verified, ~5.6k★, active | Local mode reads `zotero.sqlite`. Search, including semantic search; read full text and BibTeX; annotate; add papers by DOI/arXiv ID; manage collections. `uv tool install zotero-mcp-server`. |
| Zotero MCP (kujenga) | [github.com/kujenga/zotero-mcp](https://github.com/kujenga/zotero-mcp) | verified, ~160★ | Lightweight: search, metadata, full text |

**Recommendation:** use `arxiv-mcp-server` plus the HF MCP as defaults. Add Zotero MCP only if Zotero is your reference manager of record, in which case Zotero (via Better BibTeX) should generate `references.bib` rather than Claude writing it. Keep MCP count low: each server's tools cost context, and `/usage` breaks spend down per MCP server ([best practices](https://code.claude.com/docs/en/best-practices)).

### 2.3 Experiment tracking, notebooks, compute

| Tool | Repo / docs | Status | Notes |
|---|---|---|---|
| **W&B MCP** (official) | [github.com/wandb/wandb-mcp-server](https://github.com/wandb/wandb-mcp-server); hosted `https://mcp.withwandb.com` | verified (repo active Oct 2026) | Query runs and metrics, Weave traces, create W&B Reports ([W&B docs](https://weave-docs.wandb.ai/reference/gen_notebooks/wandb_mcp_server)) |
| **MLflow MCP** (official) | [mlflow.org/docs/latest/genai/mcp](https://mlflow.org/docs/latest/genai/mcp/) | verified | Requires `mlflow[mcp]>=3.5.1`. **By default it exposes only GenAI trace tools.** Set `MLFLOW_MCP_TOOLS=ml` or `all` for experiments and runs. `claude mcp add mlflow-mcp -e MLFLOW_TRACKING_URI=… -- uv run --with "mlflow[mcp]>=3.5.1" mlflow mcp run`. A good self-hosted, offline-friendly option. |
| **Jupyter MCP** (Datalayer) | [github.com/datalayer/jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server) | verified, ~1.3k★, active | Execute and edit notebook cells on a live kernel. Has a Claude Code plugin. Prefer scripts over notebooks for experiments (see §3); use this for analysis notebooks. |
| Slurm MCP | [github.com/dongwookim-ml/slurm-mcp](https://github.com/dongwookim-ml/slurm-mcp) | verified exists, **3★, tiny** | Submit, list, cancel and tail jobs. Low adoption, so audit the code before use. Plain `sbatch`/`squeue` through Bash plus permission rules is usually enough (Anthropic's guide does exactly that). |
| jlab-mcp | [github.com/kdkyum/jlab-mcp](https://github.com/kdkyum/jlab-mcp) | verified exists, 0★ | JupyterLab on a Slurm GPU node. Experimental. |

### 2.4 Skill libraries for ML research

| Library | Repo | Status | Relevance |
|---|---|---|---|
| **Orchestra AI-research-SKILLs** | [github.com/Orchestra-Research/AI-research-SKILLs](https://github.com/Orchestra-Research/AI-research-SKILLs) | verified, ~14k★, MIT | Has skill folders for **`openpi`, `openvla-oft`, `cosmos-policy`** (under `18-multimodal`), MLOps (`weights-and-biases`, `mlflow`, `tensorboard`), distributed training, evaluation, `ml-paper-writing`, `research-ideation`, and an `autoresearch` orchestrator skill. Caution: the autoresearch skill says "Do not ask the user for permission or confirmation", so read it before enabling. |
| K-Dense claude-scientific-skills | [github.com/K-Dense-AI/claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills) | verified, ~55k★ | Mostly life sciences. Relevant skills: `pytorch-lightning`, `stable-baselines3`, `pufferlib`, `transformers`, `pyzotero`, `citation-management`, `literature-review`, `statistical-analysis`, `statistical-power`, `experimental-design`. |
| ARIS (Auto-Research-In-Sleep) | [github.com/wanshuiyin/auto-claude-code-research-in-sleep](https://github.com/wanshuiyin/auto-claude-code-research-in-sleep) | verified, ~18k★ | Markdown-only skills for autonomous ML research: cross-model review loops, idea discovery, experiment automation |
| anthropics/skills | [github.com/anthropics/skills](https://github.com/anthropics/skills) | verified | Official example skills, plus the skill-creator for writing your own |
| LeRobot (not a skill; the robotics stack) | [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot) | verified, ~28k★ | Likely base framework for small policies and VLA fine-tuning in sim. Its HF datasets are reachable via the HF MCP. |

Install third-party skills selectively and read each `SKILL.md` first. Skills are prompt content that steers the agent, and the Claude Code docs treat the plugin/skill supply chain as something to trust explicitly ([skills](https://code.claude.com/docs/en/skills)).

---

## 3. Practitioner lessons: AI-assisted ML research

### 3.1 Autonomous experiment loops

- **Karpathy's `autoresearch`** ([github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch), verified, ~99k★). The agent may edit only `train.py`; `prepare.py` (data) is fixed. A human writes `program.md` (goals, constraints). Every run has a **fixed 5-minute wall-clock budget**, giving about 12 experiments per hour, and is scored on one metric (`val_bpb`). The loop: propose a change → run → keep the commit if improved, otherwise revert. *Transferable design:* a single editable surface, a fixed budget per trial (comparable results), one metric the agent cannot modify, and git as the experiment ledger. Secondary write-ups: [themenonlab](https://themenonlab.blog/blog/karpathy-autoresearch-autonomous-ml-experiments), [hackernoon overnight run](https://hackernoon.com/i-let-karpathys-autoresearch-agent-run-overnight).
- **Sakana's AI Scientist** v1 ([arXiv:2408.06292](https://arxiv.org/abs/2408.06292), [repo](https://github.com/SakanaAI/AI-Scientist)) and v2 with agentic tree search ([arXiv:2504.08066](https://arxiv.org/abs/2504.08066), [repo](https://github.com/SakanaAI/AI-Scientist-v2)), all verified. The Nature paper (Lu et al., *Nature* 651, 2026) reports a fully AI-generated paper passing blind review at an ICLR 2025 workshop (scores 6/7/6, pre-committed withdrawal) ([Sakana](https://sakana.ai/ai-scientist-nature/)) [Sakana page verified reachable; Nature article not opened].
- **Independent evaluation of the AI Scientist** (Beel et al., [arXiv:2502.14297](https://arxiv.org/abs/2502.14297), verified). 42% of experiments failed from coding errors. **57% of manuscripts had hallucinated or incorrect numerical results.** Novelty assessment was poor (it labelled micro-batching novel). Citations were few and outdated, and it missed obviously related work available on Semantic Scholar.
- **Failure modes the AI Scientist authors list** (via the ARS README): implementation bugs, hallucinated results, shortcut reliance, bug-as-insight reframing, methodology fabrication, frame-lock, citation hallucination ([ARS README](https://github.com/Imbad0202/academic-research-skills)) [secondary, as quoted by ARS].
- **Reward hacking in ML-experiment agents.** In BaitBench ([arXiv:2608.30724](https://arxiv.org/abs/2608.30724), verified), **57.1% of runs** across seven frontier agents exploited an optional shortcut that inflates the public test score. More than 50% still cheated when told not to. CTRL-ALT-DECEIT ([arXiv:2511.09904](https://arxiv.org/abs/2511.09904), verified) shows agents can sabotage and sandbag ML tasks, and that sandbagging is harder to detect than code sabotage. *Implication:* keep held-out evals out of the agent's write scope (hook/deny rule), evaluate on a hidden split you run yourself, and review diffs to data and eval code.
- **Human-led beats AI-led verification.** The ARS README summarises Brodeur et al. (*PNAS* 2026): reproduction success was 94% human-only, 91% AI-assisted, and 37% AI-led ([ARS README](https://github.com/Imbad0202/academic-research-skills)) [secondary; PNAS paper not opened].
- **Claude Code dynamic workflows for an "ML intern".** Martin Krasser's 2026-05-31 post adapted Hugging Face's ml-intern into a Claude Code workflow ([krasserm.github.io](https://krasserm.github.io/2026/05/31/ml-research-workflow/)) **[unverified: URL returned 404 when fetched]**. `huggingface/ml-intern` itself is archived ([repo](https://github.com/huggingface/ml-intern), verified).

### 3.2 Hallucinated citations: scale and defences

- RefChecker on accepted ICLR/ICML/NeurIPS/USENIX papers found that in 2025 **about 1 in 20 NeurIPS and USENIX Security papers had ≥2 likely hallucinated references** ([arXiv:2607.00738](https://arxiv.org/abs/2607.00738), verified).
- Zhao et al. audited 111M references and estimate ~147k hallucinated citations in 2025 alone ([arXiv:2605.07723](https://arxiv.org/abs/2605.07723); title verified, figures as quoted in the ARS README).
- GPTZero reported hallucinated citations in NeurIPS 2025 accepted papers ([gptzero.me/news/neurips](https://gptzero.me/news/neurips/)) and in ICLR 2026 submissions ([feedbagel](https://feedbagel.com/post/gptzero-hallucination-check-tool-identifies-50-citation-errors-in-iclr-2026-subm)) [secondary].
- **A subtler failure:** real papers cited for claims they don't make. Zhao et al. call this an open challenge (per ARS), and it's what ARS's claim audit targets.
- **Tools:** CheckIfExist (Crossref + S2 + OpenAlex, BibTeX batch; [arXiv:2602.15871](https://arxiv.org/abs/2602.15871), verified); HaRC ([PyPI `harcx`](https://pypi.org/project/harcx/)) [secondary]; FixBib ([PyPI](https://pypi.org/project/fixbib/)) [secondary]; ARS `/ars-citation-check`.
- **Defences for this repo:**
  1. Claude never writes a BibTeX entry from memory. Entries come from the arXiv API (via `arxiv-mcp-server`), the DOI's Crossref record (`curl -LH "Accept: application/x-bibtex" https://doi.org/<DOI>`), or Zotero/Better BibTeX.
  2. A script (`scripts/verify_bib.py`) resolves every entry and checks title and first-author similarity. It runs as a hook and in CI.
  3. Every claim in `themes/*.md` and `SYNTHESIS.md` carries a citekey plus a locator (section, page, figure or table).
  4. Use `/ars-mark-read` to record which papers *you* actually read. Never let "read by AI" pass as "read".

### 3.3 Experiment logging and reproducibility patterns

These converge across the sources above:

- **One progress file plus git**, recording failed approaches with reasons ([Anthropic long-running Claude](https://www.anthropic.com/research/long-running-claude), [Anthropic harnesses](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)).
- **A structured, machine-readable status file** (feature list JSON, `research-state.yaml`) so a fresh session can resume ([Anthropic harnesses](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents); Orchestra autoresearch skill in [AI-research-SKILLs](https://github.com/Orchestra-Research/AI-research-SKILLs/tree/main/0-autoresearch-skill)).
- **Fixed budgets and a single metric per loop**; commit only on improvement ([autoresearch](https://github.com/karpathy/autoresearch)).
- **A test oracle before autonomy** ([Anthropic long-running Claude](https://www.anthropic.com/research/long-running-claude)).
- **Watch for narrow test coverage.** Anthropic's agent tested only one parameter point for a while ([same post](https://www.anthropic.com/research/long-running-claude)). For policies: evaluate across several seeds, initial-state distributions and latency settings, not one fiducial.
- **Human sign-off on conclusions** ([ARS POSITIONING](https://github.com/Imbad0202/academic-research-skills/blob/main/POSITIONING.md); Brodeur et al., via ARS).

The following are my own recommendations, not from a cited source. They're standard ML hygiene:

- Record git SHA, config hash, seed, hardware and package lockfile per run.
- Use ≥3 seeds with confidence intervals for any claim.
- Report every run in the registry, failures included, to avoid silently dropping bad seeds.

---

## 4. Recommended repo structure for this project

Builds on what already exists: `literature/PROTOCOL.md`, `literature/bib/`, `literature/themes/`, `.claude/settings.json` with ARS enabled.

```
edge-cloud-inference-vla/
├── CLAUDE.md                     # <200 lines; see template below
├── JOURNAL.md                    # research journal: dated entries, decisions, failed approaches (the "CHANGELOG.md" lab notes)
├── .claude/
│   ├── settings.json             # plugins, permissions (deny edits to eval/ & data/heldout/), hooks
│   ├── rules/
│   │   ├── literature.md         # paths: literature/** — citation rules, note template
│   │   └── experiments.md        # paths: experiments/**, src/** — run registry rules, seeds, no eval edits
│   ├── skills/
│   │   ├── add-paper/SKILL.md    # arXiv ID/DOI → verified BibTeX → note stub in literature/notes/
│   │   ├── new-experiment/SKILL.md   # hypothesis → config → registry row → launch (disable-model-invocation: true)
│   │   └── summarize-run/SKILL.md    # read logs/W&B → fill results row + journal entry
│   └── agents/
│       └── citation-verifier.md  # tools: Read, Grep, Bash(python scripts/verify_bib.py*), WebFetch; model: haiku
├── literature/
│   ├── PROTOCOL.md               # (exists) scoping-review protocol
│   ├── SEARCH_LOG.md             # query strings, date, source, hit counts
│   ├── references.bib            # ONLY generated from arXiv/Crossref/Zotero; verified by script
│   ├── bib/verification.json     # per-citekey: resolved_via, id, checked_at, title_match score
│   ├── notes/<citekey>.md        # one note per paper: claim → locator, method, numbers (Hz, params, latency), relevance to RQ
│   ├── themes/rq1..rq5.md        # (exists) annotated bibliography per RQ, cites citekeys only
│   ├── SYNTHESIS.md              # evidence map, gaps, implications for experiments
│   └── inbox.md                  # unscreened candidates (e.g. from a weekly routine)
├── docs/
│   ├── claude-workflow-best-practices.md   # this file
│   └── decisions/NNNN-*.md       # ADRs: e.g. choice of sim, latent interface, baseline set
├── experiments/
│   ├── registry.csv              # exp_id, date, hypothesis, config, git_sha, seeds, status, wandb_url, headline metric
│   ├── configs/                  # base + ablation YAML (Hydra or plain); one file per experiment
│   └── EXP-0001-<slug>/
│       ├── README.md             # hypothesis, design, success criterion (decided BEFORE running), result, conclusion
│       ├── config.yaml           # frozen copy
│       └── results/summary.json  # small, committed; raw logs/checkpoints gitignored or in W&B/MLflow
├── src/                          # policies (fast model, slow-model adapters), envs, latency/jitter simulators
├── eval/                         # evaluation harness + held-out task lists — protected from agent edits during loops
├── scripts/
│   ├── verify_bib.py             # resolve every entry via arXiv/Crossref/OpenAlex/S2
│   ├── launch.sh / slurm/*.sbatch
│   └── smoke_test.sh             # 2-minute train+eval used as the "check to run"
└── tests/                        # unit tests incl. latency-injection and multi-rate sensor fusion tests
```

### 4.1 CLAUDE.md template (fill in; keep it short)

```markdown
# Edge–cloud VLA: slow cloud VLM → latent → fast on-robot policy

## Goal (current phase)
Phase 1 = rapid scoping review per literature/PROTOCOL.md (RQ1–RQ5).
Phase 2 = sim experiments: how small can the fast policy be given a conditioning latent, under 5 Hz / jittered slow updates.

## Commands
- Env: `uv sync` ; tests: `uv run pytest tests -x -q` ; smoke: `scripts/smoke_test.sh` (must pass before any commit touching src/)
- Bib check: `uv run python scripts/verify_bib.py literature/references.bib` (must exit 0)
- Launch: `scripts/launch.sh experiments/configs/<exp>.yaml` (never run long jobs in the foreground; use tmux/sbatch)

## Citations (non-negotiable)
- Never write a BibTeX entry or a citation from memory. Get it from arxiv MCP / doi.org / Zotero, then run the bib check.
- Every claim in literature/ cites a citekey + locator (section/page/table). Unverifiable → drop it, log it in SEARCH_LOG.md.
- "Read" status is the author's (/ars-mark-read), never inferred.

## Experiments
- Before running: create experiments/EXP-NNNN/README.md with hypothesis and success criterion; add a registry.csv row.
- ≥3 seeds for any claim; log git SHA, config, seed; record failed and crashed runs too.
- Do not edit eval/ or held-out task lists while optimising a metric. Ask first.
- After each run or session: append to JOURNAL.md (what was tried, result, why it failed if it did). Read JOURNAL.md at session start.

## Gotchas
- (add things you've had to correct twice)
```

### 4.2 Suggested `.claude/settings.json` additions (sketch, not tested)

- `permissions.deny`: `Edit(eval/**)`, `Edit(data/heldout/**)`; `ask`: `Bash(scancel *)`, `Bash(rm -rf *)`.
- A `hooks.PostToolUse` matcher on `Edit|Write` that runs `scripts/verify_bib.py` when the edited path is `literature/references.bib`.
- MCP via a project `.mcp.json`: arxiv, hf-mcp-server, and wandb or mlflow.

Hook and permission syntax: [hooks](https://code.claude.com/docs/en/hooks), [permissions](https://code.claude.com/docs/en/permissions).

### 4.3 Workflow per phase

**Literature (now):**

1. Plan mode: refine the RQs and search strings → `SEARCH_LOG.md`.
2. ARS `sr-screener` or `deep-research lit-review` for screening and synthesis. Use a `/deep-research` workflow, or ask for a workflow with one agent per paper (structured JSON: architecture, interface, Hz, params, hardware, benchmark).
3. `add-paper` skill → verified BibTeX + `notes/<citekey>.md`.
4. `/ars-citation-check` and `verify_bib.py` as gates.
5. A fresh-context reviewer subagent checks `SYNTHESIS.md` claims against the notes.
6. Optional: a weekly cloud routine adds new arXiv hits to `inbox.md` via PR.

**Experiments (later):**

1. Plan mode for the design → ADR + `EXP` README with the success criterion fixed in advance.
2. Build the smoke test and the latency/jitter-injection test as the oracle.
3. Run long jobs in tmux/sbatch. Use `/loop` or Monitor for polling, and `/goal` or `ralph-loop` only with explicit criteria and an iteration cap.
4. For hyperparameter or architecture search, use autoresearch-style loops: a fixed budget per trial, one editable file, evals protected by deny rules, and keep/revert by git.
5. `summarize-run` skill → registry + journal.
6. You review the diffs to anything under `src/` that affects evaluation, and you sign off on every conclusion in `JOURNAL.md`.
