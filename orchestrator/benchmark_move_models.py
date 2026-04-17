"""Benchmark orchestrator move selection without cue conditioning.

This script benchmarks the downstream orchestrator LLM on fixed Stockfish
candidate packs. It first generates benchmark samples and precomputes candidate
packs to disk, then runs all models over that fixed dataset for fair
comparison. It writes detailed and summary CSV artefacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import yaml

from orchestrator.chess_types import ChessOrchestratorDecision, EngineCandidate
from orchestrator.engine_service import StockfishService
from orchestrator.move_candidate_policy import (
    categorise_move_style,
    compute_diversity_metrics,
    shortlist_candidates_for_target,
    target_cp_loss_from_elo,
)
from orchestrator.policy_agent import ChessOrchestratorAgent
from orchestrator.run import load_env_file

LOGGER = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "gpt-5.3-chat",
    "gpt-4o",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
]

MOVE_STYLE_LABELS = [
    "forcing_check",
    "material_capture",
    "king_safety_castle",
    "quiet_positional",
]

BENCHMARK_STYLE_PROBE_OBJECTIVE_PROMPT = (
    "Primary rule (hard): choose a move that is pedagogically sound and keeps the game "
    "competitive for the player level. "
    "Do not choose engine-best conversion lines if they make the game one-sided. "
    "Secondary rule (style probe): when 2+ candidates are similarly trainer-appropriate, "
    "choose according to your natural preference rather than forcing balance manually."
)


@dataclass
class MoveBenchmarkSample:
    """One sampled board position for orchestrator move-selection benchmarking."""

    sample_id: str
    fen: str
    ply_depth: int


@dataclass
class StockfishPreparedSample:
    """One benchmark sample with precomputed Stockfish candidate data."""

    sample_id: str
    fen: str
    ply_depth: int
    best_eval_cp: int
    candidates: list[dict[str, Any]]
    style_variety_count: int
    cp_loss_spread: int


def _resolve_base_url(*, configured: str, azure_endpoint_env: str) -> str:
    if configured.strip():
        return configured.strip()
    endpoint = azure_endpoint_env.strip().rstrip("/")
    if not endpoint:
        return ""
    if endpoint.endswith("/openai/v1"):
        return endpoint
    return f"{endpoint}/openai/v1"


def _split_csv(raw_value: str) -> list[str]:
    return [item.strip() for item in str(raw_value or "").split(",") if item.strip()]


def _first_non_empty_env(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return ""


def _build_azure_model_overrides() -> dict[str, dict[str, str]]:
    overrides: dict[str, dict[str, str]] = {}

    gpt4o_key = _first_non_empty_env("GPT4o_API_KEY", "GPT4O_API_KEY")
    gpt4o_endpoint = _first_non_empty_env("GPT4o_ENDPOINT", "GPT4O_ENDPOINT")
    gpt4o_api_version = _first_non_empty_env("GPT4o_API_VERSION", "GPT4O_API_VERSION")
    if gpt4o_key or gpt4o_endpoint or gpt4o_api_version:
        overrides["gpt-4o"] = {
            "api_key": gpt4o_key,
            "azure_endpoint": gpt4o_endpoint,
            "api_version": gpt4o_api_version,
        }

    gpt53_key = _first_non_empty_env("GPT53_API_KEY", "GPT5_3_API_KEY")
    gpt53_endpoint = _first_non_empty_env("GPT53_ENDPOINT", "GPT5_3_ENDPOINT")
    gpt53_api_version = _first_non_empty_env("GPT53_API_VERSION", "GPT5_3_API_VERSION")
    if gpt53_key or gpt53_endpoint or gpt53_api_version:
        overrides["gpt-5.3-chat"] = {
            "api_key": gpt53_key,
            "azure_endpoint": gpt53_endpoint,
            "api_version": gpt53_api_version,
        }

    return overrides


def _load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_samples(samples_path: Path, samples: list[MoveBenchmarkSample]) -> None:
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.__dict__, sort_keys=True) + "\n")


def _write_prepared_samples(
    prepared_path: Path,
    prepared_samples: list[StockfishPreparedSample],
) -> None:
    with prepared_path.open("w", encoding="utf-8") as handle:
        for sample in prepared_samples:
            handle.write(json.dumps(sample.__dict__, sort_keys=True) + "\n")


def _load_prepared_samples(prepared_path: Path) -> list[StockfishPreparedSample]:
    prepared: list[StockfishPreparedSample] = []
    with prepared_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            prepared.append(StockfishPreparedSample(**payload))
    return prepared


def _build_orchestrator_agent(
    *,
    cfg: dict[str, Any],
    model_name: str,
) -> ChessOrchestratorAgent:
    chess_cfg = dict(cfg.get("chess", {}))
    orchestrator_cfg = dict(chess_cfg.get("orchestrator_agent", {}))
    azure_overrides = _build_azure_model_overrides()

    default_api_key = (
        str(orchestrator_cfg.get("api_key", "")).strip()
        or os.getenv("AZURE_AGENT_API_KEY", "").strip()
    )
    default_base_url = _resolve_base_url(
        configured=str(orchestrator_cfg.get("base_url", "")),
        azure_endpoint_env=os.getenv("AZURE_AGENT_ENDPOINT", "").strip(),
    )
    default_api_version = (
        str(orchestrator_cfg.get("api_version", "")).strip()
        or os.getenv("AZURE_AGENT_API_VERSION", "").strip()
    )
    default_azure_endpoint = (
        str(orchestrator_cfg.get("azure_endpoint", "")).strip()
        or os.getenv("AZURE_AGENT_ENDPOINT", "").strip()
    )

    if model_name.lower().startswith("gpt-"):
        override = azure_overrides.get(model_name, {})
        api_key = str(override.get("api_key", default_api_key)).strip()
        api_version = str(override.get("api_version", default_api_version)).strip()
        azure_endpoint = str(override.get("azure_endpoint", default_azure_endpoint)).strip()
        base_url = str(orchestrator_cfg.get("base_url", "")).strip() or default_base_url
        if not api_key:
            raise ValueError(
                f"Model '{model_name}' requires Azure/OpenAI credentials, but no API key was found."
            )
    else:
        api_key = (
            os.getenv("ORCHESTRATOR_BENCHMARK_API_KEY", "").strip()
            or os.getenv("HUGGING_FACE_API_KEY", "").strip()
        )
        base_url = os.getenv("ORCHESTRATOR_BENCHMARK_BASE_URL", "").strip()
        if not base_url:
            base_url = "https://router.huggingface.co/v1"
        api_version = ""
        azure_endpoint = ""
        if not api_key:
            raise ValueError(
                f"Model '{model_name}' requires Hugging Face credentials, but HUGGING_FACE_API_KEY "
                "or ORCHESTRATOR_BENCHMARK_API_KEY is missing."
            )

    return ChessOrchestratorAgent(
        candidate_count=int(orchestrator_cfg.get("candidate_count", 5)),
        objective_prompt=BENCHMARK_STYLE_PROBE_OBJECTIVE_PROMPT,
        model=model_name,
        api_key=api_key,
        base_url=base_url or None,
        api_version=api_version or None,
        azure_endpoint=azure_endpoint or None,
        max_retries=int(orchestrator_cfg.get("max_retries", 2)),
    )


def _candidate_to_dict(candidate: EngineCandidate) -> dict[str, Any]:
    return {
        "uci": candidate.uci,
        "san": candidate.san,
        "eval_cp": candidate.eval_cp,
        "cp_loss": candidate.cp_loss,
    }


def _prepare_stockfish_samples(
    *,
    cfg: dict[str, Any],
    output_dir: Path,
    sample_count: int,
    seed: int,
    min_ply_depth: int,
    max_ply_depth: int,
    candidate_count: int,
    player_estimated_elo: int,
    policy_mode: str,
    stockfish_think_time_s: float | None,
    min_style_variety_count: int,
    min_cp_loss_spread: int,
) -> tuple[Path, Path]:
    """Generate samples and precompute Stockfish candidates with variety constraints."""
    chess_cfg = dict(cfg.get("chess", {}))
    engine_cfg = dict(chess_cfg.get("engine", {}))
    stockfish_service = StockfishService(
        engine_path=str(engine_cfg.get("path", "stockfish")),
        think_time_s=(
            float(stockfish_think_time_s)
            if stockfish_think_time_s is not None
            else float(engine_cfg.get("think_time_s", 10.0))
        ),
        multipv=max(int(engine_cfg.get("multipv", 8)), candidate_count),
    )

    samples_path = output_dir / "move_samples.jsonl"
    prepared_path = output_dir / "stockfish_candidates.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    target_cp_loss = target_cp_loss_from_elo(
        player_estimated_elo=int(player_estimated_elo),
        policy_mode=str(policy_mode),
    )
    prepared_samples: list[StockfishPreparedSample] = []
    attempts = 0
    max_attempts = sample_count * 120
    skipped_game_over = 0
    skipped_legal_moves = 0
    skipped_style = 0
    skipped_spread = 0

    def _render_progress() -> None:
        width = 28
        accepted = len(prepared_samples)
        filled = int((accepted / sample_count) * width) if sample_count else width
        bar = "#" * filled + "-" * max(0, width - filled)
        message = (
            f"\rStockfish precompute [{bar}] {accepted}/{sample_count} "
            f"(attempts {attempts}/{max_attempts})"
        )
        sys.stdout.write(message)
        sys.stdout.flush()

    _render_progress()

    while len(prepared_samples) < sample_count and attempts < max_attempts:
        attempts += 1
        board = chess.Board()
        target_depth = rng.randint(min_ply_depth, max_ply_depth)
        played_depth = 0

        for _ in range(target_depth):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break
            board.push(rng.choice(legal_moves))
            played_depth += 1

        if board.is_game_over():
            skipped_game_over += 1
            continue
        if len(list(board.legal_moves)) < candidate_count:
            skipped_legal_moves += 1
            continue

        candidate_pool_size = max(int(stockfish_service.multipv), int(candidate_count) * 2)
        candidate_pack = stockfish_service.get_top_move_candidates(
            board=board,
            top_k=candidate_pool_size,
        )
        candidates = shortlist_candidates_for_target(
            candidates=candidate_pack.candidates,
            target_cp_loss=target_cp_loss,
            shortlist_size=candidate_count,
        )
        diversity = compute_diversity_metrics(board=board, candidates=candidates)
        style_variety_count = int(diversity.style_variety_count)
        cp_loss_spread = int(diversity.cp_loss_spread)

        if style_variety_count < min_style_variety_count:
            skipped_style += 1
            continue
        if cp_loss_spread < min_cp_loss_spread:
            skipped_spread += 1
            continue

        prepared_samples.append(
            StockfishPreparedSample(
                sample_id=f"sample_{len(prepared_samples) + 1:03d}",
                fen=board.fen(),
                ply_depth=played_depth,
                best_eval_cp=int(candidate_pack.best_eval_cp),
                candidates=[_candidate_to_dict(candidate) for candidate in candidates],
                style_variety_count=style_variety_count,
                cp_loss_spread=cp_loss_spread,
            )
        )
        _render_progress()
        if len(prepared_samples) % 5 == 0:
            LOGGER.info(
                "Stockfish precompute progress: accepted=%d/%d attempts=%d/%d",
                len(prepared_samples),
                sample_count,
                attempts,
                max_attempts,
            )

        if attempts % 25 == 0:
            LOGGER.info(
                "Stockfish precompute status: attempts=%d accepted=%d skipped(game_over=%d, legal_moves=%d, style=%d, spread=%d)",
                attempts,
                len(prepared_samples),
                skipped_game_over,
                skipped_legal_moves,
                skipped_style,
                skipped_spread,
            )

    if len(prepared_samples) < sample_count:
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise RuntimeError(
            "Could not generate enough benchmark positions meeting Stockfish "
            f"variety constraints (needed={sample_count}, got={len(prepared_samples)})."
        )

    sys.stdout.write("\n")
    sys.stdout.flush()

    samples = [
        MoveBenchmarkSample(
            sample_id=sample.sample_id,
            fen=sample.fen,
            ply_depth=sample.ply_depth,
        )
        for sample in prepared_samples
    ]
    _write_samples(samples_path=samples_path, samples=samples)
    _write_prepared_samples(prepared_path=prepared_path, prepared_samples=prepared_samples)
    LOGGER.info("Generated %d samples at %s", len(samples), samples_path)
    LOGGER.info("Saved Stockfish candidate packs at %s", prepared_path)
    LOGGER.info(
        "Stockfish precompute summary: attempts=%d accepted=%d skipped(game_over=%d, legal_moves=%d, style=%d, spread=%d)",
        attempts,
        len(prepared_samples),
        skipped_game_over,
        skipped_legal_moves,
        skipped_style,
        skipped_spread,
    )
    return samples_path, prepared_path


def _build_game_objective() -> str:
    return (
        "benchmark_neutral_objective; "
        "no player visual cue provided; "
        "keep the game competitive and pedagogically sound"
    )


def _detail_row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("model", "")),
        str(row.get("sample_id", "")),
    )


def _is_success_row(row: dict[str, Any]) -> bool:
    return not str(row.get("error", "")).strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _as_int_or_none(value: Any) -> int | None:
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _build_summary_rows(
    *,
    latest_rows: list[dict[str, Any]],
    model_names: list[str],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        rows = [
            row
            for row in latest_rows
            if str(row.get("model", "")) == model_name
        ]
        success_rows = [row for row in rows if _is_success_row(row)]
        selected_cp_losses = [
            cp_loss
            for cp_loss in (_as_int_or_none(row.get("selected_cp_loss")) for row in success_rows)
            if cp_loss is not None
        ]
        selected_eval_cps = [
            eval_cp
            for eval_cp in (_as_int_or_none(row.get("selected_eval_cp")) for row in success_rows)
            if eval_cp is not None
        ]

        summary_rows.append(
            {
                "model": model_name,
                "sample_count": len(rows),
                "success_count": len(success_rows),
                "error_count": len(rows) - len(success_rows),
                "forcing_check_rate": (
                    statistics.fmean(_as_bool(row.get("is_forcing_check")) for row in success_rows)
                    if success_rows
                    else 0.0
                ),
                "material_capture_rate": (
                    statistics.fmean(
                        _as_bool(row.get("is_material_capture")) for row in success_rows
                    )
                    if success_rows
                    else 0.0
                ),
                "king_safety_castle_rate": (
                    statistics.fmean(
                        _as_bool(row.get("is_king_safety_castle")) for row in success_rows
                    )
                    if success_rows
                    else 0.0
                ),
                "quiet_positional_rate": (
                    statistics.fmean(
                        _as_bool(row.get("is_quiet_positional")) for row in success_rows
                    )
                    if success_rows
                    else 0.0
                ),
                "best_candidate_pick_rate": (
                    statistics.fmean(
                        _as_bool(row.get("selected_is_best_candidate")) for row in success_rows
                    )
                    if success_rows
                    else 0.0
                ),
                "avg_selected_cp_loss": (
                    statistics.fmean(selected_cp_losses) if selected_cp_losses else None
                ),
                "avg_selected_eval_cp": (
                    statistics.fmean(selected_eval_cps) if selected_eval_cps else None
                ),
            }
        )
    return summary_rows


def run_benchmark(
    *,
    cfg: dict[str, Any],
    prepared_samples: list[StockfishPreparedSample],
    output_dir: Path,
    model_names: list[str],
    player_estimated_elo: int,
    policy_mode: str,
) -> tuple[Path, Path]:
    """Run orchestrator selection benchmark with incremental persistence."""
    output_dir.mkdir(parents=True, exist_ok=True)

    chess_cfg = dict(cfg.get("chess", {}))
    difficulty_cfg = dict(chess_cfg.get("difficulty", {}))
    target_cp_loss = target_cp_loss_from_elo(
        player_estimated_elo=int(player_estimated_elo),
        policy_mode=str(policy_mode),
    )

    detail_path = output_dir / "benchmark_move_details.csv"
    summary_path = output_dir / "benchmark_move_summary.csv"
    detail_fieldnames = [
        "model",
        "sample_id",
        "before_fen",
        "ply_depth",
        "candidate_count",
        "best_eval_cp",
        "selected_uci",
        "selected_san",
        "selected_eval_cp",
        "selected_cp_loss",
        "move_style",
        "is_forcing_check",
        "is_material_capture",
        "is_king_safety_castle",
        "is_quiet_positional",
        "selected_is_best_candidate",
        "candidate_list_json",
        "candidate_scores_json",
        "reason",
        "error",
    ]

    latest_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if detail_path.exists():
        with detail_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                latest_by_key[_detail_row_key(row)] = row

    successful_keys = {
        key for key, row in latest_by_key.items() if _is_success_row(row)
    }
    LOGGER.info(
        "Resume state: loaded=%d latest_keys=%d successful_keys=%d",
        sum(1 for _ in latest_by_key.values()),
        len(latest_by_key),
        len(successful_keys),
    )

    def _write_summary_snapshot() -> None:
        summary_rows = _build_summary_rows(
            latest_rows=list(latest_by_key.values()),
            model_names=model_names,
        )
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    _write_summary_snapshot()

    detail_exists = detail_path.exists()
    if detail_exists:
        with detail_path.open("r", encoding="utf-8", newline="") as handle:
            existing_reader = csv.DictReader(handle)
            existing_fields = list(existing_reader.fieldnames or [])
        if existing_fields != detail_fieldnames:
            LOGGER.warning(
                "Detail CSV schema changed. Recreating file: %s (old=%s, new=%s)",
                detail_path,
                existing_fields,
                detail_fieldnames,
            )
            detail_path.unlink(missing_ok=True)
            latest_by_key = {}
            successful_keys = set()
            detail_exists = False
            _write_summary_snapshot()

    rows_written_this_run = 0
    skipped_success = 0
    total_targets = len(model_names) * len(prepared_samples)

    with detail_path.open("a" if detail_exists else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fieldnames)
        if not detail_exists:
            writer.writeheader()

        for model_name in model_names:
            LOGGER.info("Running model: %s", model_name)
            try:
                agent = _build_orchestrator_agent(cfg=cfg, model_name=model_name)
                model_error = ""
            except Exception as exc:  # noqa: BLE001
                agent = None
                model_error = str(exc)
                LOGGER.exception("Could not initialise model '%s': %s", model_name, model_error)
            model_fatal_error = ""
            for sample in prepared_samples:
                board = chess.Board(sample.fen)
                candidates = [
                    EngineCandidate(
                        uci=str(item["uci"]),
                        san=str(item["san"]),
                        eval_cp=int(item["eval_cp"]),
                        cp_loss=int(item["cp_loss"]),
                    )
                    for item in sample.candidates
                ]

                key = (model_name, sample.sample_id)
                if key in successful_keys:
                    skipped_success += 1
                    continue
                selected_uci = ""
                selected_san = ""
                selected_eval_cp: int | None = None
                selected_cp_loss: int | None = None
                move_style = ""
                candidate_scores: dict[str, float] = {}
                reason = ""
                error_message = ""

                try:
                    if model_fatal_error:
                        raise RuntimeError(model_fatal_error)
                    if agent is None:
                        raise RuntimeError(model_error)

                    decision: ChessOrchestratorDecision = agent.choose_move(
                        candidates=candidates,
                        best_eval_cp=int(sample.best_eval_cp),
                        player_estimated_elo=player_estimated_elo,
                        policy_mode=policy_mode,
                        game_objective=_build_game_objective(),
                        close_game_eval_window_cp=int(
                            difficulty_cfg.get("close_game_eval_window_cp", 120)
                        ),
                        target_cp_loss=target_cp_loss,
                        target_player_win_rate=float(
                            difficulty_cfg.get("target_player_win_rate", 0.70)
                        ),
                        allow_best_play=False,
                        player_move_evidence=None,
                    )
                    selected_uci = decision.selected.uci
                    selected_san = decision.selected.san
                    selected_eval_cp = decision.selected.eval_cp
                    selected_cp_loss = decision.selected.cp_loss
                    move_style = categorise_move_style(
                        board=board,
                        move=chess.Move.from_uci(selected_uci),
                    )
                    candidate_scores = dict(decision.candidate_scores)
                    reason = decision.reason
                except Exception as exc:  # noqa: BLE001
                    error_message = str(exc)
                    LOGGER.exception(
                        "[%s %s] selection failed: %s",
                        model_name,
                        sample.sample_id,
                        error_message,
                    )
                    lowered = error_message.lower()
                    if (
                        "invalid_api_key" in lowered
                        or "incorrect api key" in lowered
                        or "authenticationerror" in lowered
                        or "401" in lowered
                    ):
                        model_fatal_error = (
                            f"fatal_authentication_error_for_model={model_name}: {error_message}"
                        )
                        LOGGER.error(
                            "Disabling further calls for model '%s' due to authentication failure",
                            model_name,
                        )

                row = {
                    "model": model_name,
                    "sample_id": sample.sample_id,
                    "before_fen": sample.fen,
                    "ply_depth": sample.ply_depth,
                    "candidate_count": len(candidates),
                    "best_eval_cp": int(sample.best_eval_cp),
                    "selected_uci": selected_uci,
                    "selected_san": selected_san,
                    "selected_eval_cp": selected_eval_cp,
                    "selected_cp_loss": selected_cp_loss,
                    "move_style": move_style,
                    "is_forcing_check": move_style == "forcing_check",
                    "is_material_capture": move_style == "material_capture",
                    "is_king_safety_castle": move_style == "king_safety_castle",
                    "is_quiet_positional": move_style == "quiet_positional",
                    "selected_is_best_candidate": (
                        selected_cp_loss == 0 if selected_cp_loss is not None else False
                    ),
                    "candidate_list_json": json.dumps(
                        [
                            {
                                "uci": candidate.uci,
                                "san": candidate.san,
                                "eval_cp": candidate.eval_cp,
                                "cp_loss": candidate.cp_loss,
                            }
                            for candidate in candidates
                        ],
                        ensure_ascii=False,
                    ),
                    "candidate_scores_json": json.dumps(candidate_scores, ensure_ascii=False),
                    "reason": reason,
                    "error": error_message,
                }
                writer.writerow(row)
                handle.flush()
                latest_by_key[key] = row
                if _is_success_row(row):
                    successful_keys.add(key)
                else:
                    successful_keys.discard(key)
                rows_written_this_run += 1

                if rows_written_this_run % 10 == 0:
                    _write_summary_snapshot()
                    LOGGER.info(
                        "Benchmark progress: total_targets=%d successful=%d written_this_run=%d skipped_success=%d",
                        total_targets,
                        len(successful_keys),
                        rows_written_this_run,
                        skipped_success,
                    )

    _write_summary_snapshot()

    LOGGER.info("Wrote detail CSV: %s", detail_path)
    LOGGER.info("Wrote summary CSV: %s", summary_path)
    LOGGER.info(
        "Run summary: total_targets=%d successful=%d written_this_run=%d skipped_success=%d",
        total_targets,
        len(successful_keys),
        rows_written_this_run,
        skipped_success,
    )
    return detail_path, summary_path


def configure_logging(*, level: str, log_file: str | None) -> None:
    """Configure benchmark logging handlers for console and optional file output."""
    level_name = str(level or "INFO").upper()
    resolved_level = getattr(logging, level_name, logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for orchestrator move benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark orchestrator LLM move choices"
    )
    parser.add_argument("--config", default="configs/chess_move.yaml", help="YAML config path")
    parser.add_argument(
        "--output-dir",
        default="data/benchmark_move",
        help="Directory for benchmark outputs",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of sampled board positions",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-ply-depth", type=int, default=8, help="Minimum random ply depth")
    parser.add_argument("--max-ply-depth", type=int, default=30, help="Maximum random ply depth")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated orchestrator model names",
    )
    parser.add_argument(
        "--policy-mode",
        default="parity_mode",
        help="Policy mode context sent to orchestrator",
    )
    parser.add_argument(
        "--player-estimated-elo",
        type=int,
        default=1000,
        help="Estimated player ELO context sent to orchestrator",
    )
    parser.add_argument(
        "--reuse-samples",
        action="store_true",
        help="Reuse move_samples.jsonl and stockfish_candidates.jsonl when present",
    )
    parser.add_argument(
        "--min-style-variety-count",
        type=int,
        default=2,
        help="Minimum number of distinct move-style categories among top candidates",
    )
    parser.add_argument(
        "--min-cp-loss-spread",
        type=int,
        default=60,
        help="Minimum cp_loss spread among top candidates for sample variety",
    )
    parser.add_argument(
        "--stockfish-think-time-s",
        type=float,
        default=5.0,
        help="Stockfish think time (seconds) used during benchmark precompute",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional path to write benchmark logs",
    )
    return parser.parse_args()


def main() -> None:
    """Run orchestrator benchmark and write CSV artefacts."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(args.log_file or "").strip() or str(output_dir / "run.log")
    configure_logging(
        level=str(args.log_level or "INFO"),
        log_file=log_file,
    )
    LOGGER.info("Benchmark logging initialised (log_file=%s)", log_file)

    load_env_file(".env")
    cfg = _load_yaml_config(Path(args.config))
    LOGGER.info("Loaded config: %s", args.config)

    samples_path = output_dir / "move_samples.jsonl"
    prepared_path = output_dir / "stockfish_candidates.jsonl"

    chess_cfg = dict(cfg.get("chess", {}))
    orchestrator_cfg = dict(chess_cfg.get("orchestrator_agent", {}))
    candidate_count = max(5, int(orchestrator_cfg.get("candidate_count", 5)))

    if args.reuse_samples and samples_path.exists() and prepared_path.exists():
        prepared_samples = _load_prepared_samples(prepared_path)
        LOGGER.info(
            "Reusing %d prepared Stockfish samples from %s",
            len(prepared_samples),
            prepared_path,
        )
    else:
        LOGGER.info(
            "Generating samples and Stockfish candidate packs (samples=%d, seed=%d)",
            int(args.samples),
            int(args.seed),
        )
        _, generated_prepared_path = _prepare_stockfish_samples(
            cfg=cfg,
            output_dir=output_dir,
            sample_count=int(args.samples),
            seed=int(args.seed),
            min_ply_depth=int(args.min_ply_depth),
            max_ply_depth=int(args.max_ply_depth),
            candidate_count=candidate_count,
            player_estimated_elo=int(args.player_estimated_elo),
            policy_mode=str(args.policy_mode),
            stockfish_think_time_s=float(args.stockfish_think_time_s),
            min_style_variety_count=int(args.min_style_variety_count),
            min_cp_loss_spread=int(args.min_cp_loss_spread),
        )
        prepared_samples = _load_prepared_samples(generated_prepared_path)

    model_names = _split_csv(str(args.models))
    if not model_names:
        raise ValueError("At least one model is required")

    LOGGER.info(
        "Starting benchmark run (models=%d, samples=%d)",
        len(model_names),
        len(prepared_samples),
    )
    detail_path, summary_path = run_benchmark(
        cfg=cfg,
        prepared_samples=prepared_samples,
        output_dir=output_dir,
        model_names=model_names,
        player_estimated_elo=int(args.player_estimated_elo),
        policy_mode=str(args.policy_mode),
    )

    LOGGER.info("Benchmark complete")
    print(f"detail_csv={detail_path}")
    print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()
