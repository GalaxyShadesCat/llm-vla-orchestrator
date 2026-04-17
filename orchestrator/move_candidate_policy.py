"""Shared policy for Elo-aware Stockfish candidate targeting and shortlist selection."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from orchestrator.chess_types import EngineCandidate


@dataclass
class CandidateDiversityMetrics:
    """Summary metrics describing a candidate shortlist."""

    style_variety_count: int
    cp_loss_spread: int


def target_cp_loss_from_elo(*, player_estimated_elo: int, policy_mode: str) -> int:
    """Map estimated player Elo to a target centipawn loss for candidate targeting."""
    elo = int(player_estimated_elo)

    if elo <= 900:
        baseline = 300
    elif elo <= 1100:
        baseline = 240
    elif elo <= 1300:
        baseline = 190
    elif elo <= 1500:
        baseline = 150
    elif elo <= 1700:
        baseline = 115
    elif elo <= 1900:
        baseline = 85
    else:
        baseline = 60

    if str(policy_mode).strip() == "soft_mode":
        baseline += 40
    elif str(policy_mode).strip() == "parity_mode":
        baseline += 0
    else:
        baseline -= 25

    return max(20, min(350, int(baseline)))


def categorise_move_style(*, board: chess.Board, move: chess.Move) -> str:
    """Categorise a move into one of four benchmark style labels."""
    if board.gives_check(move):
        return "forcing_check"
    if board.is_capture(move):
        return "material_capture"
    if board.is_castling(move):
        return "king_safety_castle"
    return "quiet_positional"


def shortlist_candidates_for_target(
    *,
    candidates: list[EngineCandidate],
    target_cp_loss: int,
    shortlist_size: int,
) -> list[EngineCandidate]:
    """Build a deterministic shortlist centred around the target centipawn loss."""
    if shortlist_size <= 0:
        raise ValueError("shortlist_size must be positive")
    if not candidates:
        raise ValueError("candidates must not be empty")
    if len(candidates) <= shortlist_size:
        return list(candidates)

    target = int(target_cp_loss)
    ranked = sorted(
        candidates,
        key=lambda candidate: (abs(int(candidate.cp_loss) - target), int(candidate.cp_loss)),
    )
    shortlist = ranked[:shortlist_size]
    shortlist_uci = {candidate.uci for candidate in shortlist}

    below = [candidate for candidate in candidates if int(candidate.cp_loss) <= target]
    above = [candidate for candidate in candidates if int(candidate.cp_loss) >= target]
    anchor_below = max(below, key=lambda candidate: int(candidate.cp_loss)) if below else None
    anchor_above = min(above, key=lambda candidate: int(candidate.cp_loss)) if above else None

    def _replace_furthest_with(anchor: EngineCandidate | None) -> None:
        if anchor is None or anchor.uci in shortlist_uci:
            return
        furthest = max(
            shortlist,
            key=lambda candidate: abs(int(candidate.cp_loss) - target),
        )
        shortlist.remove(furthest)
        shortlist_uci.remove(furthest.uci)
        shortlist.append(anchor)
        shortlist_uci.add(anchor.uci)

    _replace_furthest_with(anchor_below)
    _replace_furthest_with(anchor_above)

    shortlist.sort(key=lambda candidate: int(candidate.cp_loss))
    return shortlist


def compute_diversity_metrics(
    *,
    board: chess.Board,
    candidates: list[EngineCandidate],
) -> CandidateDiversityMetrics:
    """Compute style diversity and cp-loss spread for a candidate set."""
    if not candidates:
        return CandidateDiversityMetrics(style_variety_count=0, cp_loss_spread=0)

    styles = {
        categorise_move_style(board=board, move=chess.Move.from_uci(candidate.uci))
        for candidate in candidates
    }
    cp_losses = [int(candidate.cp_loss) for candidate in candidates]
    return CandidateDiversityMetrics(
        style_variety_count=len(styles),
        cp_loss_spread=max(cp_losses) - min(cp_losses),
    )

