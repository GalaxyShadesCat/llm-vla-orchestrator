"""Entrypoint for running either motion or chess orchestration pipelines."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from envs.mock_env import MockEnv
from orchestrator.agent import AzureReActTaskAgent, RuleBasedTaskAgent
from orchestrator.chess_pipeline import (
    ChessMemoryStore,
    ChessTurnLogger,
    ChessTurnPipeline,
    ChesscogCliRecognizer,
    DirectoryCamera,
)
from orchestrator.core import Orchestrator
from orchestrator.logger import RunLogger
from orchestrator.specs import SubtaskSpec, TaskSpec
from verifier.azure_chatgpt import AzureChatGPTVisionVerifier
from verifier.stub import StubVerifier


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def apply_langsmith_settings(cfg: dict[str, Any]) -> None:
    ls_cfg = cfg.get("langsmith", {})
    if not isinstance(ls_cfg, dict):
        return

    enabled = ls_cfg.get("enabled")
    if enabled is not None:
        os.environ["LANGSMITH_TRACING"] = "true" if bool(enabled) else "false"

    project = ls_cfg.get("project")
    if project:
        os.environ.setdefault("LANGSMITH_PROJECT", str(project))

    api_key = ls_cfg.get("api_key")
    if api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", str(api_key))

    endpoint = ls_cfg.get("endpoint")
    if endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", str(endpoint))


def build_task_spec(cfg: dict[str, Any]) -> TaskSpec:
    task_cfg = cfg["task"]
    subtasks: list[SubtaskSpec] = []

    for item in task_cfg.get("subtasks", []):
        subtasks.append(
            SubtaskSpec(
                name=item["name"],
                instruction=item["instruction"],
                success_criteria=item["success_criteria"],
                params=dict(item.get("params", {})),
                max_attempts=int(item.get("max_attempts", 10)),
                max_attempt_seconds=float(item.get("max_attempt_seconds", 10.0)),
            )
        )

    return TaskSpec(name=task_cfg["name"], subtasks=subtasks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run task orchestration pipeline")
    parser.add_argument(
        "--config",
        default="configs/line_crossing.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--reset-game-state",
        action="store_true",
        help="Reset chess game state before running chess_turn pipeline.",
    )
    parser.add_argument(
        "--turn-note",
        default="",
        help="Optional note attached to this chess turn and stored in memory.",
    )
    return parser.parse_args()


def build_verifier(cfg: dict[str, Any]) -> Any:
    verifier_cfg = cfg.get("verifier", {})
    verifier_type = str(verifier_cfg.get("type", "stub")).lower()

    if verifier_type == "stub":
        return StubVerifier(
            crossing_margin_px=int(verifier_cfg.get("crossing_margin_px", 4)),
        )

    if verifier_type == "azure_chatgpt":
        deployment_cfg = str(verifier_cfg.get("deployment") or "").strip()
        if deployment_cfg.startswith("your_"):
            deployment_cfg = ""
        deployment = str(deployment_cfg or os.getenv("AZURE_VISION_DEPLOYMENT") or "").strip()
        if not deployment:
            raise ValueError(
                "verifier.deployment is required when verifier.type=azure_chatgpt "
                "(or set AZURE_VISION_DEPLOYMENT in .env)"
            )
        return AzureChatGPTVisionVerifier(
            deployment=deployment,
            api_key=verifier_cfg.get("api_key"),
            api_version=verifier_cfg.get("api_version"),
            endpoint=verifier_cfg.get("endpoint"),
            timeout_s=float(verifier_cfg.get("timeout_s", 30.0)),
        )

    raise ValueError(f"Unsupported verifier type: {verifier_type}")


def build_task_agent(cfg: dict[str, Any]) -> Any:
    agent_cfg = cfg.get("agent", {})
    agent_type = str(agent_cfg.get("type", "rule_based")).lower()

    if agent_type == "rule_based":
        return RuleBasedTaskAgent()

    if agent_type == "azure_react":
        deployment_cfg = str(agent_cfg.get("deployment") or "").strip()
        if deployment_cfg.startswith("your_"):
            deployment_cfg = ""
        deployment = str(deployment_cfg or os.getenv("AZURE_AGENT_DEPLOYMENT") or "").strip()
        if not deployment:
            raise ValueError(
                "agent.deployment is required when agent.type=azure_react "
                "(or set AZURE_AGENT_DEPLOYMENT in .env)"
            )
        return AzureReActTaskAgent(
            deployment=deployment,
            api_key=agent_cfg.get("api_key"),
            api_version=agent_cfg.get("api_version"),
            endpoint=agent_cfg.get("endpoint"),
            timeout_s=float(agent_cfg.get("timeout_s", 30.0)),
        )

    raise ValueError(f"Unsupported agent type: {agent_type}")


def main() -> None:
    args = parse_args()
    load_env_file(".env")
    cfg = load_config(args.config)
    apply_langsmith_settings(cfg)

    pipeline_cfg = cfg.get("pipeline", {})
    pipeline_type = str(pipeline_cfg.get("type", "motion")).lower()
    if pipeline_type == "chess_turn":
        _run_chess_turn_pipeline(
            cfg,
            reset_game_state=bool(args.reset_game_state),
            turn_note=str(args.turn_note or "").strip() or None,
        )
        return

    control_hz = int(cfg.get("control_hz", 50))
    env_cfg = cfg.get("env", {})
    run_dir = str(cfg.get("run_dir", "runs"))
    verbose = bool(cfg.get("verbose", False))

    env = MockEnv(
        control_hz=control_hz,
        arm_limit=float(env_cfg.get("arm_limit", 1.0)),
    )
    verifier = build_verifier(cfg)
    task_agent = build_task_agent(cfg)
    logger = RunLogger(base_dir=run_dir)

    orchestrator = Orchestrator(
        env=env,
        verifier=verifier,
        logger=logger,
        control_hz=control_hz,
        task_agent=task_agent,
        verbose=verbose,
    )

    task = build_task_spec(cfg)
    episode = orchestrator.run_task(task)
    env.close()

    print(f"Task: {episode['task_name']}")
    print(f"Status: {episode['status']}")
    print(f"Run directory: {episode['run_dir']}")
    print(f"Steps log: {episode['steps_log']}")


def _run_chess_turn_pipeline(
    cfg: dict[str, Any],
    *,
    reset_game_state: bool = False,
    turn_note: str | None = None,
) -> None:
    run_dir = str(cfg.get("run_dir", "runs"))
    chess_cfg = cfg.get("chess", {})
    retries_cfg = cfg.get("retries", {})
    camera_cfg = chess_cfg.get("camera", {})
    memory_cfg = chess_cfg.get("memory", {})
    recogniser_cfg = chess_cfg.get("recogniser", {})

    camera = DirectoryCamera(
        inbox_dir=str(camera_cfg.get("inbox_dir", "data/chess_camera/inbox")),
        current_filename=str(camera_cfg.get("current_filename", "current.jpg")),
    )
    recogniser = ChesscogCliRecognizer(
        python_executable=str(recogniser_cfg.get("python_executable", "python")),
        module=str(recogniser_cfg.get("module", "chesscog.recognition.recognition")),
        assume_white_bottom=bool(recogniser_cfg.get("assume_white_bottom", True)),
    )
    memory_store = ChessMemoryStore(
        state_path=str(memory_cfg.get("state_path", "data/chess_camera/state/game_state.json")),
        initial_fen=str(
            memory_cfg.get(
                "initial_fen",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            )
        ),
    )
    if bool(memory_cfg.get("reset_on_start", False)) or reset_game_state:
        reset_reason = "cli_flag" if reset_game_state else "config_reset_on_start"
        memory_store.reset(reason=reset_reason)
    logger = ChessTurnLogger(base_dir=run_dir)

    pipeline = ChessTurnPipeline(
        camera=camera,
        recogniser=recogniser,
        memory_store=memory_store,
        logger=logger,
        max_vision_retries_per_turn=int(retries_cfg.get("max_vision_retries_per_turn", 3)),
        legal_match_min_confidence=float(chess_cfg.get("legal_match_min_confidence", 0.75)),
        emit_full_fen=bool(chess_cfg.get("emit_full_fen", True)),
        full_fen_defaults=dict(chess_cfg.get("full_fen_defaults", {})),
        max_execution_retries_per_turn=int(retries_cfg.get("max_execution_retries_per_turn", 3)),
        turn_note=turn_note or str(memory_cfg.get("turn_note", "")).strip() or None,
    )
    result = pipeline.run_turn()

    print("Pipeline: chess_turn")
    print(f"Status: {result['status']}")
    print(f"Turn index: {result['turn_index']}")
    print(f"Run directory: {result['run_dir']}")
    print(f"Turns log: {result['turns_log']}")
    print(f"PGN path: {result['pgn_path']}")
    if result["selected_move_uci"]:
        print(f"Detected move (UCI): {result['selected_move_uci']}")
    else:
        print("Detected move (UCI): none")


if __name__ == "__main__":
    main()
