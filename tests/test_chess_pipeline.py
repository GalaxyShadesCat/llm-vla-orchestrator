"""Tests for the chess turn pipeline and move-memory validation."""

from __future__ import annotations

import json
from pathlib import Path

import chess

from orchestrator.chess_pipeline import (
    ChessMemoryStore,
    ChessTurnLogger,
    ChessTurnPipeline,
    ChesscogCliRecognizer,
    DirectoryCamera,
    RecognitionResult,
)


class FakeRecogniser:
    """Deterministic recogniser used for pipeline unit tests."""

    def __init__(self, piece_placement: str, confidence: float | None = None) -> None:
        self._piece_placement = piece_placement
        self._confidence = confidence

    def recognise(self, image_path: str) -> RecognitionResult:
        return RecognitionResult(
            piece_placement=self._piece_placement,
            confidence=self._confidence,
            raw_output=f"fake:{image_path}",
        )


def _build_pipeline(
    *,
    tmp_path: Path,
    piece_placement: str,
    initial_fen: str = chess.STARTING_FEN,
) -> ChessTurnPipeline:
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    (inbox_dir / "current.jpg").write_bytes(b"test")

    camera = DirectoryCamera(inbox_dir=str(inbox_dir), current_filename="current.jpg")
    recogniser = FakeRecogniser(piece_placement=piece_placement)
    memory_store = ChessMemoryStore(
        state_path=str(tmp_path / "state" / "game_state.json"),
        initial_fen=initial_fen,
    )
    logger = ChessTurnLogger(base_dir=str(tmp_path / "runs"))

    return ChessTurnPipeline(
        camera=camera,
        recogniser=recogniser,
        memory_store=memory_store,
        logger=logger,
        max_vision_retries_per_turn=3,
        legal_match_min_confidence=0.75,
        emit_full_fen=True,
        full_fen_defaults={
            "side_to_move": "w",
            "castling": "-",
            "en_passant": "-",
            "halfmove": 0,
            "fullmove": 1,
        },
        max_execution_retries_per_turn=3,
    )


def test_chesscog_output_parser_extracts_piece_placement() -> None:
    stdout = """
. K R . . R . .
P . P P Q . . P
. P B B . . . .
. . . . . P . .
. . b . . p . q
. p . . . . . .
p b p p . . . p
. k r . . . r .

You can view this position at https://lichess.org/editor/1KR2R2
"""
    piece_placement = ChesscogCliRecognizer._parse_piece_placement(stdout)
    assert piece_placement == "1KR2R2/P1PPQ2P/1PBB4/5P2/2b2p1q/1p6/pbpp3p/1kr3r1"


def test_pipeline_accepts_legal_transition_and_updates_memory(tmp_path: Path) -> None:
    board = chess.Board()
    board.push_san("e4")
    pipeline = _build_pipeline(tmp_path=tmp_path, piece_placement=board.board_fen())

    result = pipeline.run_turn()

    assert result["status"] == "ok"
    assert result["selected_move_uci"] == "e2e4"

    state_path = tmp_path / "state" / "game_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["moves_uci"] == ["e2e4"]
    assert state["turn_index"] == 1

    turns_log = Path(result["turns_log"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(turns_log) == 1

    pgn_text = Path(result["pgn_path"]).read_text(encoding="utf-8")
    assert "1. e4" in pgn_text

    boards_dir = Path(result["boards_dir"])
    assert (boards_dir / "turn_0001_pre.svg").exists()
    assert (boards_dir / "turn_0001_observed_piece_placement.svg").exists()
    assert (boards_dir / "turn_0001_post_observed.svg").exists()
    assert (boards_dir / "turn_0001_post_expected.svg").exists()


def test_pipeline_rejects_illegal_transition(tmp_path: Path) -> None:
    illegal_piece_placement = "8/8/8/8/8/8/8/8"
    pipeline = _build_pipeline(tmp_path=tmp_path, piece_placement=illegal_piece_placement)

    result = pipeline.run_turn()

    assert result["status"] == "invalid_transition"
    assert result["selected_move_uci"] is None

    state_path = tmp_path / "state" / "game_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["moves_uci"] == []
    assert state["turn_index"] == 0

    boards_dir = Path(result["boards_dir"])
    assert (boards_dir / "turn_0001_pre.svg").exists()
    assert (boards_dir / "turn_0001_observed_piece_placement.svg").exists()
    assert (boards_dir / "turn_0001_post_expected.svg").exists()
    assert not (boards_dir / "turn_0001_post_observed.svg").exists()
