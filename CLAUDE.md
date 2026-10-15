## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Project status

- Data collection is finished: 2010–2025, 6,911 driver results over 329 races (counts in `results/collection_2010_2025.json`). Do not re-run it. The raw parquet is gitignored and exists only locally at `data/raw/season_*.parquet`.
- The old prototype was replaced by the `f1pred` package, specified under `specs/001-core-predictor/`. The prototype survives only at tag `v0-vibe-coded`.

## Working rules

### Get approval before building
- If you draft a spec, plan or design that will guide later work, mark it unapproved and stop for review before implementing it.
- Answering your scoping questions is not approval of the finished document. The author never approves their own spec (Constitution Article IX).
- Flag gaps in the rules proactively instead of following the letter of a rule that allows what it should prevent.

### Spec → implement → validate
- When a task or phase is done, verify it and fix what is wrong, not just report it: the "done when" clause must be true, the whole test suite must run, anything touching data is checked against the real corpus (not only fixtures), and the task board's wording must match what the code does.
- When replanning, write the requirement first, get sign-off, then implement against it, then validate against the requirement as written (Constitution Article VII).

### Explain like to a beginner
- Everyday words, no jargon. If a technical term is unavoidable, give its plain meaning next to it once.
- No bare IDs like `T-106` or `FR-206`. Say what the thing does; put the ID in parentheses after it if it's needed.
- Short bullets and small tables over long paragraphs. Lead with what happened, then the result, then what it means.

### Retired drivers
- In anything a person reads (CLI output, README, summaries), show a retired driver as "Retired (collision / engine / …)", never as a numbered finishing position.
- Keep the internal ordering for the model (it needs a full order per race); strip it only when displaying.

### Run independent work in parallel
- Batch independent tool calls into one message; run long commands in the background.
- When a parallel route exists, offer it and say in a line or two which method you're using and why.
- Say so when something can't be parallelised (e.g. the rate-limited jolpica API, two workers writing one file, a step needing another's output).

### Keep token cost low, never at the code's expense
- Use the cheapest model that can do the job (`haiku`/`sonnet` for mechanical work, `opus` only for hard design).
- Prefer doing work inline over spawning subagents; always verify subagent results.
- Orient with graphify, then read only the exact lines needed (`sed -n 'A,Bp'`, tight `grep -n`), never whole files already in context.
- Give subagents exact symbols, line ranges and signatures rather than lists of files to read.
- Run targeted tests while iterating and the full suite once before committing.
- Saving tokens never justifies thinner tests, skipped verification or weaker code.
