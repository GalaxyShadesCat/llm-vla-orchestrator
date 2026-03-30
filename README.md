# LLM-VLA Orchestrator

Minimal scaffold for a multimodal-LLM + VLA closed loop with sequential subtasks.

## Pipeline modes

The orchestrator supports two modes selected by `pipeline.type` in config:
- `motion`: arm line-crossing simulation (existing behaviour)
- `chess_turn`: directory-based chessboard observation with legal-move memory validation

## ReAct orchestration model

The orchestrator now uses an agent that can call exactly two tools:
- `move_left`
- `move_right`

Execution order is strict and sequential across the task’s subtask list.
Each subtask must be completed before the orchestrator proceeds to the next one.

For each attempt inside a subtask:
- Agent chooses one tool action
- Motion chunk executes in environment (simulated VLA step)
- Verifier checks completion from BEFORE/AFTER frames
- If not complete, params may be adjusted and next attempt starts

## Current simulation task

Task: `arm_line_crossing`
- Subtask 1: move right and cross white center line
- Subtask 2: move left and cross white center line

This is an example task. The orchestrator supports any number of subtasks and executes them in order.

## What the verifier LLM sees

Per verification call:
- `BEFORE (before.png)`
- `AFTER (after.png)`
- instruction + success criteria + params
- explicit scene mapping:
  - black background
  - white vertical center line (goal boundary)
  - green rectangle (arm marker)

## Install

Quick setup (recommended):

```bash
./setup.sh
```

Manual setup:

```bash
conda env create -f envs/environment.yml
conda activate llm-vla-orchestrator
```

To sync an existing environment to the spec:

```bash
conda env update -n llm-vla-orchestrator -f envs/environment.yml --prune
```

Install a new package ad hoc (without editing `requirements.txt` yet):

```bash
python -m pip install <package-name>
```

If you need to pull the latest `openpi` submodule commit, run:

```bash
git submodule update --remote openpi
```

The recommended interpreter for this repo is:
- `/home/lem/miniconda3/envs/llm-vla-orchestrator/bin/python`

## Run with local rule-based ReAct agent + stub verifier

```bash
python -m orchestrator.run --config configs/line_crossing.yaml
```

## Run with Azure ReAct agent + Azure vision verifier

Create a `.env` file in repo root:

```bash
AZURE_AGENT_DEPLOYMENT=your_azure_agent_deployment
AZURE_VISION_DEPLOYMENT=your_azure_vision_deployment
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_VERSION=
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=llm-vla-orchestrator
```

Set deployment names in `configs/line_crossing_azure.yaml`, then run:

```bash
python -m orchestrator.run --config configs/line_crossing_azure.yaml
```

## Run chess turn pipeline (directory camera + chess memory)

`chess_turn` mode expects one current image in:
- `data/chess_camera/inbox/current.jpg`

Run:

```bash
python -m orchestrator.run --config configs/chess_turn.yaml
```

Default behaviour is to persist and append chess memory across runs.
To reset state before a run, use either:

```bash
python -m orchestrator.run --config configs/chess_turn.yaml --reset-game-state
```

or set `chess.memory.reset_on_start: true` in config.

To add a note for a specific turn run:

```bash
python -m orchestrator.run --config configs/chess_turn.yaml --turn-note "White testing kingside pressure."
```

The chess pipeline will:
- run `chesscog` inference on the current image
- extract piece placement
- compare with previous canonical state
- accept only legal one-move transitions
- persist game memory and move logs
- store extensible memory structures (`notes`, `journal`, `events`, `stats`, `metadata`) for future reasoning/validation agents

Render board images from a run:

```bash
python -m orchestrator.render_chess_boards --run-dir runs/<timestamp>
```

This writes SVG board renders to `runs/<timestamp>/boards/`.

### Important caveat about FEN

A single image can provide piece placement, but it cannot reliably provide full game-state fields
(`side_to_move`, castling rights, en-passant target, halfmove clock, fullmove number).
The pipeline therefore stores canonical game state from move history and can optionally emit a
synthetic full FEN with configurable defaults for downstream tools.

`api_version` is optional in this code.
- If set, the code uses the Azure API-versioned client.
- If blank, the code uses Azure OpenAI v1-style base URL mode.
If your Azure setup requires API-versioned requests, set either:
- `agent.api_version` / `verifier.api_version` in config, or
- `AZURE_OPENAI_API_VERSION` in `.env`.

## LangSmith tracing

Tracing can be configured via YAML + `.env`.

- Config toggle: `langsmith.enabled` (true/false)
- Project name: `LANGSMITH_PROJECT` in `.env`
- Credentials: `LANGSMITH_API_KEY` in `.env`

Optional `.env` fields:

```bash
LANGSMITH_API_KEY=...
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

When enabled, traces include:
- top-level orchestrator task/subtask runs
- agent decision runs
- motion chunk execution
- verifier calls
- Azure OpenAI API calls (via wrapped client)

## Output artifacts

Each run writes to `runs/YYYYMMDD_HHMMSS_microseconds/`:
- `steps.jsonl` (one JSON record per attempt, including timestamps, params, execution summary, verifier result, and image paths)
- `images/<subtask>/attempt_<n>_a.png`
- `images/<subtask>/attempt_<n>_b.png`

Naming convention:
- `_a.png`: frame before the action chunk
- `_b.png`: frame after the action chunk

In `chess_turn` mode, each run writes:
- `turns.jsonl` (turn-level machine-friendly memory records)
- `game.pgn` (human-readable chess game history)
