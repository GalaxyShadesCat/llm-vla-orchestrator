"""Generate 3D-rendered chess benchmark data and evaluate vision models on move identification."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import chess
import yaml

from orchestrator.game_service import ChessGameService
from orchestrator.run import load_env_file
from orchestrator.vision_agent import VisionModelResolver, VisionProviderSettings

LOGGER = logging.getLogger(__name__)


DEFAULT_MODELS = [
    "gpt-5.3-chat",
    "gpt-4o",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "zai-org/GLM-4.5V",
]


@dataclass
class BenchmarkSample:
    """Single benchmark sample describing one legal move transition and rendered image."""

    sample_id: str
    before_fen: str
    after_fen: str
    after_piece_placement: str
    move_uci: str
    move_san: str
    image_path: str
    view_mode: str
    camera_pitch_deg: float
    camera_distance: float


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


def _build_azure_model_overrides() -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}

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


def _load_samples(samples_path: Path) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            samples.append(BenchmarkSample(**payload))
    return samples


def _write_samples(samples_path: Path, samples: list[BenchmarkSample]) -> None:
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.__dict__, sort_keys=True) + "\n")


def generate_samples(
    *,
    output_dir: Path,
    sample_count: int,
    seed: int,
    min_ply_depth: int,
    max_ply_depth: int,
    camera_pitch_base: float,
    camera_distance_base: float,
    camera_pitch_jitter: float,
    camera_distance_jitter: float,
) -> Path:
    """Generate random legal transitions and sample metadata for frontend 3D rendering."""
    LOGGER.info(
        "Generating %d samples (seed=%d, ply_depth=[%d,%d], pitch=%s±%s, distance=%s±%s)",
        sample_count,
        seed,
        min_ply_depth,
        max_ply_depth,
        camera_pitch_base,
        camera_pitch_jitter,
        camera_distance_base,
        camera_distance_jitter,
    )
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"

    samples: list[BenchmarkSample] = []
    images_dir_abs = images_dir.resolve()

    for idx in range(sample_count):
        board = chess.Board()
        target_depth = rng.randint(min_ply_depth, max_ply_depth)
        for _ in range(target_depth):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(rng.choice(legal_moves))
            if board.is_game_over():
                break

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            board = chess.Board()
            legal_moves = list(board.legal_moves)

        chosen_move = rng.choice(legal_moves)
        before_fen = board.fen()
        move_san = board.san(chosen_move)
        board.push(chosen_move)
        after_fen = board.fen()

        samples.append(
            BenchmarkSample(
                sample_id=f"sample_{idx + 1:03d}",
                before_fen=before_fen,
                after_fen=after_fen,
                after_piece_placement=board.board_fen(),
                move_uci=chosen_move.uci(),
                move_san=move_san,
                image_path=str(images_dir_abs / f"sample_{idx + 1:03d}.png"),
                view_mode="3d",
                camera_pitch_deg=max(
                    8.0,
                    min(
                        88.0,
                        camera_pitch_base + rng.uniform(-camera_pitch_jitter, camera_pitch_jitter),
                    ),
                ),
                camera_distance=max(
                    10.0,
                    min(
                        24.0,
                        camera_distance_base
                        + rng.uniform(-camera_distance_jitter, camera_distance_jitter),
                    ),
                ),
            )
        )

    _write_samples(samples_path, samples)
    LOGGER.info("Wrote samples metadata to %s", samples_path)
    return samples_path


def _all_images_exist(samples: list[BenchmarkSample]) -> bool:
    return all(Path(sample.image_path).exists() for sample in samples)


def _normalise_sample_image_paths(*, samples: list[BenchmarkSample], output_dir: Path) -> list[BenchmarkSample]:
    """Normalise sample image paths to absolute paths under the benchmark output directory."""
    images_dir_abs = (output_dir / "images").resolve()
    normalised: list[BenchmarkSample] = []
    for sample in samples:
        current_path = Path(sample.image_path)
        if current_path.is_absolute():
            next_path = current_path
        else:
            # Historical runs stored paths relative to project root. Keep filename stable.
            next_path = images_dir_abs / current_path.name
        normalised.append(
            BenchmarkSample(
                sample_id=sample.sample_id,
                before_fen=sample.before_fen,
                after_fen=sample.after_fen,
                after_piece_placement=sample.after_piece_placement,
                move_uci=sample.move_uci,
                move_san=sample.move_san,
                image_path=str(next_path),
                view_mode=sample.view_mode,
                camera_pitch_deg=sample.camera_pitch_deg,
                camera_distance=sample.camera_distance,
            )
        )
    return normalised


def _wait_for_http_ready(base_url: str, timeout_s: float) -> None:
    LOGGER.info("Waiting for frontend server at %s", base_url)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(base_url, timeout=1.5) as response:
                if response.status < 500:
                    return
        except (URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for frontend server at {base_url}")


def render_images_with_frontend(
    *,
    samples_path: Path,
    project_root: Path,
    frontend_port: int,
    render_timeout_ms: int,
) -> None:
    """Render benchmark images via the frontend ChessBoard3D snapshot mode."""
    LOGGER.info(
        "Rendering images via frontend snapshot mode (port=%d, timeout_ms=%d)",
        frontend_port,
        render_timeout_ms,
    )
    frontend_dir = project_root / "frontend"
    script_path = frontend_dir / "scripts" / "render_chess_positions.mjs"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing renderer script: {script_path}")

    base_url = f"http://127.0.0.1:{frontend_port}"
    absolute_samples_path = samples_path.resolve()
    dev_server_cmd = [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        "127.0.0.1",
        "--port",
        str(frontend_port),
        "--strictPort",
    ]

    dev_server = subprocess.Popen(
        dev_server_cmd,
        cwd=str(frontend_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_for_http_ready(base_url=base_url, timeout_s=90.0)
        render_cmd = [
            "node",
            str(script_path),
            "--samples-jsonl",
            str(absolute_samples_path),
            "--base-url",
            base_url,
            "--timeout-ms",
            str(render_timeout_ms),
            "--width",
            "560",
        ]
        LOGGER.info("Running renderer script for samples: %s", absolute_samples_path)
        subprocess.run(render_cmd, cwd=str(frontend_dir), check=True)
        LOGGER.info("Frontend rendering complete")
    finally:
        LOGGER.info("Shutting down temporary frontend server")
        dev_server.terminate()
        try:
            dev_server.wait(timeout=8)
        except subprocess.TimeoutExpired:
            dev_server.kill()
            dev_server.wait(timeout=4)


def _build_model_resolver(cfg: dict[str, Any], requested_models: list[str]) -> VisionModelResolver:
    chess_cfg = dict(cfg.get("chess", {}))
    vision_cfg = dict(chess_cfg.get("vision", {}))

    default_model = str(vision_cfg.get("model", "")).strip() or "gpt-5.3-chat"
    vision_api_key = str(vision_cfg.get("api_key", "")).strip() or os.getenv(
        "AZURE_VISION_API_KEY", ""
    ).strip()
    vision_base_url = _resolve_base_url(
        configured=str(vision_cfg.get("base_url", "")),
        azure_endpoint_env=os.getenv("AZURE_VISION_ENDPOINT", "").strip(),
    )
    vision_api_version = str(vision_cfg.get("api_version", "")).strip() or os.getenv(
        "AZURE_VISION_API_VERSION", ""
    ).strip()
    vision_azure_endpoint = str(vision_cfg.get("azure_endpoint", "")).strip() or os.getenv(
        "AZURE_VISION_ENDPOINT", ""
    ).strip()
    azure_allowlist = {
        model.strip()
        for model in _split_csv(str(vision_cfg.get("azure_model_allowlist", "")))
        if model.strip()
    }
    for model_name in requested_models:
        if model_name.lower().startswith("gpt-"):
            azure_allowlist.add(model_name)

    settings = VisionProviderSettings(
        azure_api_key=vision_api_key,
        azure_base_url=vision_base_url,
        azure_api_version=vision_api_version,
        azure_endpoint=vision_azure_endpoint,
        azure_max_retries=int(vision_cfg.get("max_retries", 2)),
        azure_timeout_s=float(vision_cfg.get("azure_timeout_s", 120.0)),
        hugging_face_api_key=os.getenv("HUGGING_FACE_API_KEY", "").strip(),
        hugging_face_max_retries=int(vision_cfg.get("hf_max_retries", 4)),
        hugging_face_timeout_s=float(vision_cfg.get("hf_timeout_s", 90.0)),
        azure_model_allowlist=azure_allowlist,
        azure_model_overrides=_build_azure_model_overrides(),
    )

    LOGGER.info(
        "Initialised vision resolver (default_model=%s, requested_models=%s)",
        default_model,
        ",".join(requested_models),
    )
    return VisionModelResolver(default_model=default_model, settings=settings)


def run_benchmark(
    *,
    cfg: dict[str, Any],
    samples_path: Path,
    output_dir: Path,
    model_names: list[str],
) -> tuple[Path, Path]:
    """Run model benchmarking and write detailed and summary CSV reports."""
    samples = _load_samples(samples_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Running benchmark over %d samples across %d models",
        len(samples),
        len(model_names),
    )

    resolver = _build_model_resolver(cfg, model_names)
    chess_cfg = dict(cfg.get("chess", {}))
    vision_cfg = dict(chess_cfg.get("vision", {}))
    legal_retry_attempts = max(1, int(vision_cfg.get("illegal_retry_attempts", 3)))
    detail_rows: list[dict[str, Any]] = []

    for model_name in model_names:
        LOGGER.info("Starting model: %s", model_name)
        try:
            resolved_name, recogniser = resolver.resolve(model_name)
            resolve_error = ""
            LOGGER.info("Resolved model '%s' -> '%s'", model_name, resolved_name)
        except Exception as exc:  # noqa: BLE001
            resolved_name = model_name
            recogniser = None
            resolve_error = str(exc)
            LOGGER.exception("Failed to resolve model '%s': %s", model_name, resolve_error)

        for index, sample in enumerate(samples, start=1):
            before_board = chess.Board(sample.before_fen)
            started_at = time.perf_counter()
            runtime_s: float | None = None
            attempts_used = 0
            error_message = ""
            predicted_san = ""
            predicted_after_piece_placement = ""
            predicted_uci = ""
            san_is_legal = False
            after_piece_placement_is_legal = False
            inferred_move_from_placement = ""
            accepted_move_uci = ""
            exact_move_match = False
            exact_piece_placement_match = False
            confidence: float | None = None
            prediction_attempt_count = 0
            webapp_selected_path = ""
            webapp_feedback = ""
            predicted_history: list[dict[str, Any]] = []
            observed_piece_placement = ""

            try:
                if recogniser is None:
                    raise RuntimeError(resolve_error)
                LOGGER.info(
                    "[%s %d/%d] infer sample=%s image=%s",
                    model_name,
                    index,
                    len(samples),
                    sample.sample_id,
                    sample.image_path,
                )
                feedback: str | None = None
                latest_prediction = None
                for _ in range(legal_retry_attempts):
                    prediction, attempt_count = recogniser.recognise_move(
                        image_path=sample.image_path,
                        before_fen=sample.before_fen,
                        feedback=feedback,
                    )
                    attempts_used += attempt_count
                    latest_prediction = prediction

                    placement_validation = ChessGameService._validate_transition(
                        board_before=before_board,
                        observed_piece_placement=prediction.after_piece_placement,
                    )

                    recognised_move = None
                    try:
                        recognised_move = before_board.parse_san(prediction.move_san)
                    except ValueError:
                        recognised_move = None

                    inferred_san_from_placement = placement_validation.matched_move_san
                    predicted_history.append(
                        {
                            "attempt": len(predicted_history) + 1,
                            "predicted_move_san": prediction.move_san,
                            "predicted_after_piece_placement": prediction.after_piece_placement,
                            "san_is_legal": recognised_move is not None,
                            "after_piece_placement_is_legal": placement_validation.is_legal,
                            "inferred_move_san_from_placement": inferred_san_from_placement,
                        }
                    )

                    if recognised_move is not None and placement_validation.is_legal:
                        san_uci = recognised_move.uci()
                        placement_uci = str(placement_validation.matched_move_uci)
                        if san_uci == placement_uci:
                            observed_piece_placement = prediction.after_piece_placement
                            webapp_selected_path = "san_and_placement_agree"
                            break

                    if recognised_move is not None:
                        trial_board = before_board.copy(stack=False)
                        trial_board.push(recognised_move)
                        observed_piece_placement = trial_board.board_fen()
                        webapp_selected_path = "san_only"
                        break

                    if placement_validation.is_legal:
                        observed_piece_placement = prediction.after_piece_placement
                        webapp_selected_path = "placement_only"
                        break

                    previous_predictions = ", ".join(
                        str(item.get("predicted_move_san") or "?") for item in predicted_history
                    )
                    feedback = (
                        "This is not what the player moved. "
                        f"Previous predicted SAN moves: [{previous_predictions}]. "
                        f"Your latest SAN '{prediction.move_san}' is not legal from before_fen "
                        f"'{sample.before_fen}', and after_piece_placement "
                        f"'{prediction.after_piece_placement}' is not a legal single-move transition. "
                        "Try again and output one legal move with matching SAN and placement."
                    )
                    webapp_feedback = feedback

                if latest_prediction is None:
                    raise RuntimeError("Vision recogniser returned no result.")
                if not observed_piece_placement:
                    raise RuntimeError(
                        "Vision could not identify a legal SAN move from the provided before_fen "
                        "and after-move image."
                    )

                prediction_attempt_count = len(predicted_history)
                runtime_s = time.perf_counter() - started_at
                predicted_san = str(latest_prediction.move_san)
                predicted_after_piece_placement = observed_piece_placement
                confidence = latest_prediction.overall_confidence

                try:
                    parsed_move = before_board.parse_san(predicted_san)
                    san_is_legal = True
                    predicted_uci = parsed_move.uci()
                except ValueError:
                    san_is_legal = False

                validation = ChessGameService._validate_transition(
                    board_before=before_board,
                    observed_piece_placement=predicted_after_piece_placement,
                )
                after_piece_placement_is_legal = validation.is_legal
                inferred_move_from_placement = str(validation.matched_move_uci or "")
                if webapp_selected_path in {"san_and_placement_agree", "placement_only"}:
                    accepted_move_uci = inferred_move_from_placement
                elif webapp_selected_path == "san_only":
                    accepted_move_uci = predicted_uci
                else:
                    accepted_move_uci = inferred_move_from_placement or predicted_uci

                exact_move_match = accepted_move_uci == sample.move_uci
                exact_piece_placement_match = (
                    predicted_after_piece_placement == sample.after_piece_placement
                )
                LOGGER.info(
                    "[%s %s] ok runtime=%.3fs attempts=%d retries=%d path=%s exact_move=%s exact_placement=%s",
                    model_name,
                    sample.sample_id,
                    runtime_s,
                    attempts_used,
                    prediction_attempt_count,
                    webapp_selected_path,
                    exact_move_match,
                    exact_piece_placement_match,
                )
            except Exception as exc:  # noqa: BLE001
                error_message = str(exc)
                LOGGER.exception(
                    "[%s %s] failed after %.3fs: %s",
                    model_name,
                    sample.sample_id,
                    time.perf_counter() - started_at,
                    error_message,
                )

            detail_rows.append(
                {
                    "model": model_name,
                    "resolved_model": resolved_name,
                    "sample_id": sample.sample_id,
                    "before_fen": sample.before_fen,
                    "ground_truth_move_uci": sample.move_uci,
                    "ground_truth_move_san": sample.move_san,
                    "ground_truth_after_piece_placement": sample.after_piece_placement,
                    "image_path": sample.image_path,
                    "attempts_used": attempts_used,
                    "webapp_retry_rounds_used": prediction_attempt_count,
                    "webapp_selected_path": webapp_selected_path,
                    "runtime_s": runtime_s,
                    "predicted_move_san": predicted_san,
                    "predicted_move_uci": predicted_uci,
                    "accepted_move_uci": accepted_move_uci,
                    "predicted_after_piece_placement": predicted_after_piece_placement,
                    "san_is_legal": san_is_legal,
                    "after_piece_placement_is_legal": after_piece_placement_is_legal,
                    "inferred_move_uci_from_piece_placement": inferred_move_from_placement,
                    "ground_truth_in_feedback_used": bool(webapp_feedback),
                    "predictions_history_json": json.dumps(predicted_history, ensure_ascii=False),
                    "exact_move_match": exact_move_match,
                    "exact_piece_placement_match": exact_piece_placement_match,
                    "overall_confidence": confidence,
                    "error": error_message,
                }
            )

    detail_path = output_dir / "vision_predicitions.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)
    LOGGER.info("Wrote detailed predictions CSV: %s", detail_path)

    summary_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        rows = [row for row in detail_rows if row["model"] == model_name]
        success_rows = [row for row in rows if not row["error"]]
        runtimes = [float(row["runtime_s"]) for row in success_rows if row["runtime_s"] is not None]
        attempts = [int(row["attempts_used"]) for row in success_rows]
        confidences = [
            float(row["overall_confidence"])
            for row in success_rows
            if row["overall_confidence"] is not None
        ]
        exact_move_matches = [bool(row["exact_move_match"]) for row in rows]
        exact_placement_matches = [bool(row["exact_piece_placement_match"]) for row in rows]
        legal_san = [bool(row["san_is_legal"]) for row in rows]
        legal_placement = [bool(row["after_piece_placement_is_legal"]) for row in rows]

        p95_runtime_s = None
        if runtimes:
            sorted_runtimes = sorted(runtimes)
            percentile_index = min(
                len(sorted_runtimes) - 1,
                max(0, math.ceil(0.95 * len(sorted_runtimes)) - 1),
            )
            p95_runtime_s = sorted_runtimes[percentile_index]

        summary_rows.append(
            {
                "model": model_name,
                "sample_count": len(rows),
                "successful_calls": len(success_rows),
                "failed_calls": len(rows) - len(success_rows),
                "successful_call_rate": (len(success_rows) / len(rows)) if rows else 0.0,
                "exact_move_accuracy": statistics.fmean(exact_move_matches) if exact_move_matches else 0.0,
                "exact_piece_placement_accuracy": (
                    statistics.fmean(exact_placement_matches) if exact_placement_matches else 0.0
                ),
                "legal_san_rate": statistics.fmean(legal_san) if legal_san else 0.0,
                "legal_after_piece_placement_rate": (
                    statistics.fmean(legal_placement) if legal_placement else 0.0
                ),
                "avg_runtime_s_success_only": statistics.fmean(runtimes) if runtimes else None,
                "p95_runtime_s_success_only": p95_runtime_s,
                "avg_attempts_success_only": statistics.fmean(attempts) if attempts else None,
                "avg_confidence_success_only": statistics.fmean(confidences) if confidences else None,
            }
        )

    summary_path = output_dir / "vision_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    LOGGER.info("Wrote summary CSV: %s", summary_path)

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
    """Parse command-line arguments for data generation, rendering, and benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark chess vision models on random positions")
    parser.add_argument("--config", default="configs/chess_move.yaml", help="Path to YAML config")
    parser.add_argument(
        "--output-dir",
        default="data/benchmark_vision",
        help="Directory for generated samples and benchmark results",
    )
    parser.add_argument("--samples", type=int, default=20, help="Number of benchmark samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-ply-depth", type=int, default=6, help="Minimum random ply depth")
    parser.add_argument("--max-ply-depth", type=int, default=26, help="Maximum random ply depth")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names (Azure deployments for GPT models, HF models otherwise)",
    )
    parser.add_argument(
        "--reuse-samples",
        action="store_true",
        help="Reuse existing samples.jsonl if present instead of regenerating",
    )
    parser.add_argument(
        "--reuse-rendered-images",
        action="store_true",
        help="Skip frontend rendering when all sample images already exist",
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=5173,
        help="Port used by temporary frontend snapshot server",
    )
    parser.add_argument(
        "--render-timeout-ms",
        type=int,
        default=90000,
        help="Per-image render timeout for frontend snapshot capture",
    )
    parser.add_argument(
        "--camera-pitch-base",
        type=float,
        default=65.0,
        help="Base camera pitch in degrees used for sample rendering",
    )
    parser.add_argument(
        "--camera-distance-base",
        type=float,
        default=18.0,
        help="Base camera distance used for sample rendering",
    )
    parser.add_argument(
        "--camera-pitch-jitter",
        type=float,
        default=2.0,
        help="Per-sample random pitch jitter (+/- degrees)",
    )
    parser.add_argument(
        "--camera-distance-jitter",
        type=float,
        default=0.8,
        help="Per-sample random distance jitter (+/- units)",
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
    """Generate benchmark data, render ChessBoard3D snapshots, and evaluate models."""
    args = parse_args()
    configure_logging(
        level=str(args.log_level or "INFO"),
        log_file=str(args.log_file or "").strip() or None,
    )
    LOGGER.info("Starting benchmark run")
    load_env_file(".env")

    cfg = _load_yaml_config(Path(args.config))
    project_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir)
    samples_path = output_dir / "samples.jsonl"

    if args.reuse_samples and samples_path.exists():
        LOGGER.info("Reusing existing samples file: %s", samples_path)
    else:
        min_ply_depth = max(0, int(args.min_ply_depth))
        max_ply_depth = max(min_ply_depth + 1, int(args.max_ply_depth))
        samples_path = generate_samples(
            output_dir=output_dir,
            sample_count=max(1, int(args.samples)),
            seed=int(args.seed),
            min_ply_depth=min_ply_depth,
            max_ply_depth=max_ply_depth,
            camera_pitch_base=float(args.camera_pitch_base),
            camera_distance_base=float(args.camera_distance_base),
            camera_pitch_jitter=max(0.0, float(args.camera_pitch_jitter)),
            camera_distance_jitter=max(0.0, float(args.camera_distance_jitter)),
        )

    samples = _load_samples(samples_path)
    samples = _normalise_sample_image_paths(samples=samples, output_dir=output_dir)
    _write_samples(samples_path, samples)
    LOGGER.info("Normalised sample image paths for %d samples", len(samples))
    if not (args.reuse_rendered_images and _all_images_exist(samples)):
        render_images_with_frontend(
            samples_path=samples_path,
            project_root=project_root,
            frontend_port=int(args.frontend_port),
            render_timeout_ms=max(1000, int(args.render_timeout_ms)),
        )
    else:
        LOGGER.info("Reusing rendered images from previous run")

    model_names = _split_csv(args.models)
    if not model_names:
        model_names = list(DEFAULT_MODELS)

    detail_path, summary_path = run_benchmark(
        cfg=cfg,
        samples_path=samples_path,
        output_dir=output_dir,
        model_names=model_names,
    )

    print(f"samples: {samples_path}")
    print(f"detail_csv: {detail_path}")
    print(f"summary_csv: {summary_path}")
    LOGGER.info("Benchmark run completed")


if __name__ == "__main__":
    main()
