# Project Plan

## Phase 0 - Setup

- Create local repo scaffold.
- Install or locate Python, Git, Kaggle CLI, and training stack.
- Accept Kaggle rules and download official files.
- Verify base model access and license constraints.

## Phase 1 - Understand The Benchmark

- Inspect official data schema and sample prompts.
- Build a parser for IDs, category labels if present, prompts, answers, and answer format.
- Reproduce the provided baseline if Kaggle supplies one.
- Pull in public trajectory data for error analysis, subject to competition rules.

Deliverable: `docs/data_report.md` with category counts, prompt patterns, and answer-format rules.

## Phase 2 - Evaluation Harness

- Implement exact-answer extraction and scoring locally.
- Add per-category metrics.
- Track experiment metadata: model, adapter rank, data mix, prompt template, train steps, validation score.
- Create a small smoke test so packaging mistakes are caught before Kaggle submissions.

Deliverable: repeatable `evaluate` command.

## Phase 3 - Synthetic Data Factory

- Build deterministic generators for each task family:
  - bit operations
  - ciphers
  - numerals/base conversions
  - unit conversions
  - gravity/physics formulas
  - symbolic equations
  - numeric equations
- Generate paired examples with concise reasoning and exact final-answer formatting.
- Add adversarial variations: longer numbers, distractor wording, unusual bases, mixed units, and boundary cases.

Deliverable: versioned JSONL training corpora.

## Phase 4 - First LoRA Fine-Tune

- Use a conservative LoRA config: rank <= 32, small learning-rate sweep, short runs.
- Start with SFT on high-confidence synthetic plus any official examples allowed for training.
- Evaluate by category and inspect failures manually.
- Package adapter exactly as Kaggle expects.

Deliverable: first valid Kaggle submission.

## Phase 5 - Improve

- Use failure-driven data generation: for each weak category, create targeted synthetic tasks.
- Tune answer extraction/final-response style.
- Try curriculum mixes: simple-to-hard, category-balanced, and hard-negative-heavy.
- Consider lightweight RL or rejection-sampled traces after SFT baseline stabilizes.

Deliverable: leaderboard-improving adapter candidates with experiment notes.

## Immediate Next Steps

1. Install or expose Git and Kaggle CLI on PATH.
2. Download the official competition files into `data/raw/`.
3. Verify the submission format and adapter constraints.
4. Build the local data inspection report.
