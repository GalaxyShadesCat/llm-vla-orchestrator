# LLM-VLA Orchestrator

Minimal scaffold for a multimodal-LLM + VLA closed loop with sequential subtasks.

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

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install a new package ad hoc (without editing `requirements.txt` yet):

```bash
python -m pip install <package-name>
```

If you need to pull the latest `openpi` submodule commit, run:

```bash
git submodule update --remote openpi
```

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

Each run writes to `runs/YYYYMMDD_HHMMSS/`:
- `steps.jsonl` (one JSON record per attempt, including timestamps, params, execution summary, verifier result, and image paths)
- `images/<subtask>/attempt_<n>_a.png`
- `images/<subtask>/attempt_<n>_b.png`

Naming convention:
- `_a.png`: frame before the action chunk
- `_b.png`: frame after the action chunk
