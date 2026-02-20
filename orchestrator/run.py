from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from envs.mock_env import MockEnv
from orchestrator.agent import AzureReActTaskAgent, RuleBasedTaskAgent
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
    parser.add_argument("--config", default="configs/line_crossing.yaml", help="Path to YAML config")
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


if __name__ == "__main__":
    main()
