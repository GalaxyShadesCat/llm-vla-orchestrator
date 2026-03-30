"""Chess turn pipeline with directory-based camera input and extensible game memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any
from zoneinfo import ZoneInfo

import chess
import chess.pgn
import chess.svg

HONG_KONG_TZ = ZoneInfo("Asia/Hong_Kong")


@dataclass
class RecognitionResult:
    """Normalised output from a board recogniser."""

    piece_placement: str
    confidence: float | None
    raw_output: str


@dataclass
class ChessTurnRecord:
    """Single turn record persisted as JSONL."""

    timestamp: str
    turn_index: int
    image_path: str
    pre_fen: str
    observed_piece_placement: str
    selected_move_uci: str | None
    selected_move_san: str | None
    post_fen_expected: str | None
    post_fen_observed: str | None
    is_legal_transition: bool
    status: str
    notes: str
    recogniser_confidence: float | None
    retries_used: int
    memory_payload: dict[str, Any]


class DirectoryCamera:
    """Reads the current chessboard image from a configured directory."""

    VALID_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    def __init__(self, inbox_dir: str, current_filename: str = "current.jpg") -> None:
        self.inbox_dir = Path(inbox_dir)
        self.current_filename = current_filename

    def get_current_image(self) -> Path:
        explicit = self.inbox_dir / self.current_filename
        if explicit.exists():
            return explicit

        # If a specific filename is configured (for example current.jpg),
        # accept the same stem with any supported image extension.
        stem = Path(self.current_filename).stem
        for suffix in self.VALID_SUFFIXES:
            candidate = self.inbox_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate

        candidates = [
            path
            for path in self.inbox_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.VALID_SUFFIXES
        ]
        if not candidates:
            raise ValueError(f"No image found in inbox directory: {self.inbox_dir}")
        return max(candidates, key=lambda path: path.stat().st_mtime)


class ChesscogCliRecognizer:
    """Runs chesscog inference from CLI and parses piece placement."""

    BOARD_LINE_PATTERN = re.compile(r"^(?:[\.prnbqkPRNBQK](?:\s+[\.prnbqkPRNBQK]){7})$")

    def __init__(
        self,
        python_executable: str = "python",
        module: str = "chesscog.recognition.recognition",
        assume_white_bottom: bool = True,
    ) -> None:
        self.python_executable = python_executable
        self.module = module
        self.assume_white_bottom = assume_white_bottom

    def recognise(self, image_path: str) -> RecognitionResult:
        command = [
            self.python_executable,
            "-m",
            self.module,
            image_path,
        ]
        if self.assume_white_bottom:
            command.append("--white")

        env = os.environ.copy()
        # chesscog wheels may omit config assets; use bundled repo configs.
        bundled_config_dir = Path(__file__).resolve().parents[1] / "chesscog_configs"
        if "CONFIG_DIR" not in env and bundled_config_dir.exists():
            env["CONFIG_DIR"] = str(bundled_config_dir)

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            if "No module named 'chesscog'" in stderr:
                raise RuntimeError(self._missing_chesscog_message()) from None
            raise RuntimeError(f"chesscog inference failed: {stderr}")

        piece_placement = self._parse_piece_placement(completed.stdout)
        return RecognitionResult(
            piece_placement=piece_placement,
            confidence=None,
            raw_output=completed.stdout,
        )

    @staticmethod
    def _missing_chesscog_message() -> str:
        return (
            "chesscog is not installed in the interpreter configured by "
            "chess.recogniser.python_executable.\n"
            "Install it in that environment, for example:\n"
            "  python -m pip install git+https://github.com/georg-wolflein/chesscog.git\n"
            "Or clone and install editable:\n"
            "  git clone https://github.com/georg-wolflein/chesscog.git\n"
            "  cd chesscog && python -m pip install -e ."
        )

    @classmethod
    def _parse_piece_placement(cls, stdout: str) -> str:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        board_lines = [line for line in lines if cls.BOARD_LINE_PATTERN.match(line)]
        if len(board_lines) < 8:
            raise ValueError("Could not parse 8 board lines from chesscog output")

        ranks = board_lines[:8]
        return cls._board_lines_to_fen_piece_placement(ranks)

    @staticmethod
    def _board_lines_to_fen_piece_placement(board_lines: list[str]) -> str:
        fen_ranks: list[str] = []
        for line in board_lines:
            tokens = line.split()
            empties = 0
            rank_out = ""
            for token in tokens:
                if token == ".":
                    empties += 1
                else:
                    if empties:
                        rank_out += str(empties)
                        empties = 0
                    rank_out += token
            if empties:
                rank_out += str(empties)
            fen_ranks.append(rank_out)
        return "/".join(fen_ranks)


class ChessMemoryStore:
    """Persists the canonical board state and move history across runs."""

    def __init__(self, state_path: str, initial_fen: str) -> None:
        self.state_path = Path(state_path)
        self.initial_fen = initial_fen

    def _new_state(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "current_fen": self.initial_fen,
            "initial_fen": self.initial_fen,
            "moves_uci": [],
            "turn_index": 0,
            "memory": {
                "notes": [],
                "journal": [],
                "events": [],
                "metadata": {},
                "stats": {
                    "total_runs": 0,
                    "total_turns": 0,
                    "ok_turns": 0,
                    "invalid_turns": 0,
                    "resets": 0,
                },
                "tags": [],
                "artifacts": {},
                "agents": {},
                "custom": {},
            },
        }

    def _normalise_state(self, state: dict[str, Any]) -> dict[str, Any]:
        state.setdefault("schema_version", 2)
        state.setdefault("current_fen", self.initial_fen)
        state.setdefault("initial_fen", self.initial_fen)
        state.setdefault("moves_uci", [])
        state.setdefault("turn_index", 0)

        memory = state.get("memory")
        if not isinstance(memory, dict):
            memory = {}
            state["memory"] = memory
        memory.setdefault("notes", [])
        memory.setdefault("journal", [])
        memory.setdefault("events", [])
        memory.setdefault("metadata", {})
        memory.setdefault(
            "stats",
            {
                "total_runs": 0,
                "total_turns": 0,
                "ok_turns": 0,
                "invalid_turns": 0,
                "resets": 0,
            },
        )
        memory.setdefault("tags", [])
        memory.setdefault("artifacts", {})
        memory.setdefault("agents", {})
        memory.setdefault("custom", {})
        return state

    def load(self) -> dict[str, Any]:
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
            state = self._normalise_state(state)
            self.save(state)
            return state

        state = self._new_state()
        self.save(state)
        return state

    def reset(self, reason: str = "manual") -> dict[str, Any]:
        """Reset persisted memory to a clean initial state."""
        state = self._new_state()
        state["memory"]["metadata"]["last_reset_reason"] = reason
        state["memory"]["metadata"]["last_reset_at"] = datetime.now(HONG_KONG_TZ).isoformat()
        state["memory"]["stats"]["resets"] = 1
        self.save(state)
        return state

    def save(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)


class ChessTurnLogger:
    """Writes chess-specific run artefacts under a timestamped run directory."""

    def __init__(self, base_dir: str = "runs") -> None:
        ts = datetime.now(HONG_KONG_TZ).strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = Path(base_dir) / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.turns_path = self.run_dir / "turns.jsonl"
        self.pgn_path = self.run_dir / "game.pgn"
        self.boards_dir = self.run_dir / "boards"
        self.boards_dir.mkdir(parents=True, exist_ok=True)

    def append_turn(self, record: ChessTurnRecord) -> None:
        with self.turns_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def write_pgn(self, initial_fen: str, moves_uci: list[str]) -> None:
        board = chess.Board(initial_fen)
        for move_uci in moves_uci:
            board.push(chess.Move.from_uci(move_uci))

        game = chess.pgn.Game.from_board(board)
        if initial_fen != chess.STARTING_FEN:
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = initial_fen

        with self.pgn_path.open("w", encoding="utf-8") as handle:
            print(game, file=handle, end="\n")

    def render_turn_boards(self, record: ChessTurnRecord, size: int = 640) -> None:
        """Render turn-specific SVG boards to the run's boards directory."""
        stem = f"turn_{record.turn_index:04d}"

        pre_board = chess.Board(record.pre_fen)
        self._write_svg(pre_board, self.boards_dir / f"{stem}_pre.svg", size=size)

        if record.post_fen_observed:
            post_observed_board = chess.Board(record.post_fen_observed)
            self._write_svg(
                post_observed_board,
                self.boards_dir / f"{stem}_post_observed.svg",
                size=size,
            )

        if record.post_fen_expected:
            post_expected_board = chess.Board(record.post_fen_expected)
            self._write_svg(
                post_expected_board,
                self.boards_dir / f"{stem}_post_expected.svg",
                size=size,
            )

        observed_board = chess.Board(record.pre_fen)
        observed_board.set_board_fen(record.observed_piece_placement)
        self._write_svg(
            observed_board,
            self.boards_dir / f"{stem}_observed_piece_placement.svg",
            size=size,
        )

    @staticmethod
    def _write_svg(board: chess.Board, out_path: Path, size: int) -> None:
        """Write a board render to an SVG file."""
        svg = chess.svg.board(board=board, size=size)
        out_path.write_text(svg, encoding="utf-8")


class ChessTurnPipeline:
    """End-to-end single-turn chess pipeline with legal move validation."""

    def __init__(
        self,
        *,
        camera: DirectoryCamera,
        recogniser: ChesscogCliRecognizer,
        memory_store: ChessMemoryStore,
        logger: ChessTurnLogger,
        max_vision_retries_per_turn: int,
        legal_match_min_confidence: float,
        emit_full_fen: bool,
        full_fen_defaults: dict[str, Any],
        max_execution_retries_per_turn: int,
        turn_note: str | None = None,
    ) -> None:
        self.camera = camera
        self.recogniser = recogniser
        self.memory_store = memory_store
        self.logger = logger
        self.max_vision_retries_per_turn = max_vision_retries_per_turn
        self.legal_match_min_confidence = legal_match_min_confidence
        self.emit_full_fen = emit_full_fen
        self.full_fen_defaults = full_fen_defaults
        self.max_execution_retries_per_turn = max_execution_retries_per_turn
        self.turn_note = turn_note.strip() if turn_note else None

    def run_turn(self) -> dict[str, Any]:
        state = self.memory_store.load()
        memory = state["memory"]
        stats = memory["stats"]
        stats["total_runs"] = int(stats.get("total_runs", 0)) + 1
        pre_fen = str(state["current_fen"])
        initial_fen = str(state["initial_fen"])
        turn_index = int(state["turn_index"]) + 1

        image_path = self.camera.get_current_image()
        recognition_result, retries_used = self._recognise_with_retries(str(image_path))

        board_before = chess.Board(pre_fen)
        matching_moves = self._find_matching_legal_moves(
            board_before,
            recognition_result.piece_placement,
        )

        selected_move: chess.Move | None = matching_moves[0] if matching_moves else None
        selected_move_uci = selected_move.uci() if selected_move else None
        selected_move_san = board_before.san(selected_move) if selected_move else None

        if selected_move is not None:
            board_after = chess.Board(pre_fen)
            board_after.push(selected_move)
            post_fen_observed = board_after.fen()
            status = "ok"
            notes = "Observed board matches a legal move from previous state."

            state["current_fen"] = post_fen_observed
            state["turn_index"] = turn_index
            state["moves_uci"].append(selected_move_uci)
            stats["ok_turns"] = int(stats.get("ok_turns", 0)) + 1
        else:
            post_fen_observed = None
            status = "invalid_transition"
            notes = "Observed board does not match any legal move from previous state."
            stats["invalid_turns"] = int(stats.get("invalid_turns", 0)) + 1
        stats["total_turns"] = int(stats.get("total_turns", 0)) + 1

        post_fen_expected = None
        if self.emit_full_fen:
            post_fen_expected = self._synthetic_full_fen(recognition_result.piece_placement)

        memory_payload = {
            "turn_index": turn_index,
            "current_fen": pre_fen,
            "recent_moves_uci": state["moves_uci"][-10:],
            "recent_moves_pgn": self._recent_moves_san(
                initial_fen=initial_fen,
                moves_uci=state["moves_uci"],
            ),
            "status": status,
            "max_execution_retries_per_turn": self.max_execution_retries_per_turn,
            "validator_context": {
                "pre_fen": pre_fen,
                "selected_move_uci": selected_move_uci,
                "selected_move_san": selected_move_san,
                "observed_piece_placement": recognition_result.piece_placement,
                "post_fen_observed": post_fen_observed,
            },
        }

        entry = {
            "timestamp": datetime.now(HONG_KONG_TZ).isoformat(),
            "turn_index": turn_index,
            "status": status,
            "selected_move_uci": selected_move_uci,
            "selected_move_san": selected_move_san,
            "observed_piece_placement": recognition_result.piece_placement,
            "image_path": str(image_path),
            "notes": notes,
            "run_dir": str(self.logger.run_dir),
        }
        if self.turn_note:
            entry["turn_note"] = self.turn_note
            memory["notes"].append(
                {
                    "timestamp": entry["timestamp"],
                    "turn_index": turn_index,
                    "text": self.turn_note,
                }
            )
        memory["journal"].append(entry)
        memory["events"].append(
            {
                "timestamp": entry["timestamp"],
                "event_type": "turn_processed",
                "turn_index": turn_index,
                "status": status,
                "run_dir": str(self.logger.run_dir),
            }
        )
        memory["metadata"]["last_run_dir"] = str(self.logger.run_dir)
        memory["metadata"]["last_image_path"] = str(image_path)
        memory["metadata"]["last_status"] = status
        self.memory_store.save(state)
        self.logger.write_pgn(initial_fen=initial_fen, moves_uci=state["moves_uci"])

        record = ChessTurnRecord(
            timestamp=datetime.now(HONG_KONG_TZ).isoformat(),
            turn_index=turn_index,
            image_path=str(image_path),
            pre_fen=pre_fen,
            observed_piece_placement=recognition_result.piece_placement,
            selected_move_uci=selected_move_uci,
            selected_move_san=selected_move_san,
            post_fen_expected=post_fen_expected,
            post_fen_observed=post_fen_observed,
            is_legal_transition=selected_move is not None,
            status=status,
            notes=notes,
            recogniser_confidence=recognition_result.confidence,
            retries_used=retries_used,
            memory_payload=memory_payload,
        )
        self.logger.append_turn(record)
        self.logger.render_turn_boards(record)

        return {
            "status": status,
            "turn_index": turn_index,
            "run_dir": str(self.logger.run_dir),
            "turns_log": str(self.logger.turns_path),
            "pgn_path": str(self.logger.pgn_path),
            "boards_dir": str(self.logger.boards_dir),
            "selected_move_uci": selected_move_uci,
            "selected_move_san": selected_move_san,
            "observed_piece_placement": recognition_result.piece_placement,
            "post_fen_observed": post_fen_observed,
            "synthetic_full_fen": post_fen_expected,
        }

    def _recognise_with_retries(self, image_path: str) -> tuple[RecognitionResult, int]:
        retries = max(1, int(self.max_vision_retries_per_turn))
        last_result: RecognitionResult | None = None

        for attempt in range(retries):
            result = self.recogniser.recognise(image_path)
            last_result = result
            confidence = result.confidence
            if confidence is None or confidence >= self.legal_match_min_confidence:
                return result, attempt

        return last_result, retries - 1

    @staticmethod
    def _find_matching_legal_moves(
        board: chess.Board,
        observed_piece_placement: str,
    ) -> list[chess.Move]:
        matches: list[chess.Move] = []
        for move in board.legal_moves:
            trial = board.copy(stack=False)
            trial.push(move)
            if trial.board_fen() == observed_piece_placement:
                matches.append(move)
        return matches

    def _synthetic_full_fen(self, piece_placement: str) -> str:
        side_to_move = str(self.full_fen_defaults.get("side_to_move", "w"))
        castling = str(self.full_fen_defaults.get("castling", "-"))
        en_passant = str(self.full_fen_defaults.get("en_passant", "-"))
        halfmove = int(self.full_fen_defaults.get("halfmove", 0))
        fullmove = int(self.full_fen_defaults.get("fullmove", 1))
        return f"{piece_placement} {side_to_move} {castling} {en_passant} {halfmove} {fullmove}"

    @staticmethod
    def _recent_moves_san(initial_fen: str, moves_uci: list[str]) -> list[str]:
        board = chess.Board(initial_fen)
        san_moves: list[str] = []
        for move_uci in moves_uci:
            move = chess.Move.from_uci(move_uci)
            san_moves.append(board.san(move))
            board.push(move)
        return san_moves[-10:]
