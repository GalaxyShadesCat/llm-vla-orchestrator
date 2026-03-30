"""Render chess pipeline turn records into board images.

This utility reads a run's ``turns.jsonl`` file and writes SVG board renders
for each turn into ``<run_dir>/boards``. PNG export is optional and requires
``cairosvg`` to be installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chess
import chess.svg


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render chess board images from a runs/*/turns.jsonl file.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing turns.jsonl (for example runs/20260330_145455).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        help="Board render size in pixels for SVG output (default: 640).",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also write PNG files (requires cairosvg).",
    )
    return parser.parse_args()


def load_turn_records(turns_path: Path) -> list[dict[str, Any]]:
    """Load turn records from a JSONL file."""
    records: list[dict[str, Any]] = []
    for line in turns_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    return records


def board_from_piece_placement(piece_placement: str, reference_fen: str) -> chess.Board:
    """Build a board from piece placement using a full-FEN reference."""
    board = chess.Board(reference_fen)
    board.set_board_fen(piece_placement)
    return board


def render_svg(board: chess.Board, out_path: Path, size: int) -> None:
    """Render a board as SVG."""
    svg = chess.svg.board(board=board, size=size)
    out_path.write_text(svg, encoding="utf-8")


def render_png_from_svg(svg_path: Path, png_path: Path) -> None:
    """Render a PNG from an existing SVG file."""
    try:
        import cairosvg  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PNG rendering requested but cairosvg is not installed. "
            "Install it with: python -m pip install cairosvg"
        ) from exc

    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))


def render_turn_record(
    record: dict[str, Any],
    out_dir: Path,
    size: int,
    also_png: bool,
) -> None:
    """Render all useful board views for one turn record."""
    turn_index = int(record["turn_index"])
    stem = f"turn_{turn_index:04d}"

    pre_fen = str(record["pre_fen"])
    pre_board = chess.Board(pre_fen)
    pre_svg = out_dir / f"{stem}_pre.svg"
    render_svg(pre_board, pre_svg, size)

    post_observed = record.get("post_fen_observed")
    if post_observed:
        post_board = chess.Board(str(post_observed))
        post_svg = out_dir / f"{stem}_post_observed.svg"
        render_svg(post_board, post_svg, size)
        if also_png:
            render_png_from_svg(post_svg, out_dir / f"{stem}_post_observed.png")

    post_expected = record.get("post_fen_expected")
    if post_expected:
        expected_board = chess.Board(str(post_expected))
        expected_svg = out_dir / f"{stem}_post_expected.svg"
        render_svg(expected_board, expected_svg, size)
        if also_png:
            render_png_from_svg(expected_svg, out_dir / f"{stem}_post_expected.png")

    observed_piece_placement = str(record["observed_piece_placement"])
    observed_board = board_from_piece_placement(observed_piece_placement, pre_fen)
    observed_svg = out_dir / f"{stem}_observed_piece_placement.svg"
    render_svg(observed_board, observed_svg, size)

    if also_png:
        render_png_from_svg(pre_svg, out_dir / f"{stem}_pre.png")
        render_png_from_svg(observed_svg, out_dir / f"{stem}_observed_piece_placement.png")


def main() -> None:
    """Entrypoint for rendering board images from chess run records."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    turns_path = run_dir / "turns.jsonl"
    if not turns_path.exists():
        raise FileNotFoundError(f"turns.jsonl not found at: {turns_path}")

    records = load_turn_records(turns_path)
    if not records:
        raise ValueError(f"No turn records found in: {turns_path}")

    out_dir = run_dir / "boards"
    out_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        render_turn_record(record, out_dir, size=args.size, also_png=bool(args.png))

    print(f"Rendered {len(records)} turn(s) to: {out_dir}")


if __name__ == "__main__":
    main()
