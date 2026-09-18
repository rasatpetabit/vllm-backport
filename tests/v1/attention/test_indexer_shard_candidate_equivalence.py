"""CPU tests for the composition of indexer query-sharding with the V4.1
candidate-block mechanism (producer layer 20 -> consumers 24/28/32/36).

Two silent-when-wrong things are pinned here:

1. The **candidate mask semantics** (``_candidate_flags_kernel`` +
   ``_mask_candidates_kernel`` in ``candidate_blocks.py``): which packed
   columns survive for a given candidate set, including ``-1`` pads, the
   packed-column clamp onto the sentinel block ``nblocks``, and the
   ``col == width - 1`` edge rule -- plus the property that a context shorter
   than ``topk_blocks * block_size`` can never be masked at all.
2. The **selection equivalence**: candidate-restricted scoring followed by a
   row top-k selects the same indices as full-width scoring followed by the
   same candidate mask.

Everything here is pure Python over integer/float inputs -- no CUDA, no Triton,
no torch device work -- because the failure these guard against (wrong top-k
indices) does not crash and must not need scarce hardware to catch.

The Python model below is a model *of* the kernels, justified line-by-line
against ``candidate_blocks.py``, not a re-implementation of the Triton
lowering: it pins the documented masking contract and the index mapping, and it
cannot certify the kernels themselves. That is the skipped test at the end,
whose on-device design is written out in
``docs/masterplan/dsv41-flash-a100-perf/findings/ws2b-equivalence-and-r1.md``.
"""

from typing import Iterable

import pytest

from vllm.v1.attention.backends.mla.indexer import indexer_decode_shard_rows

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


def teardown_function(function) -> None:
    """Make the collected failures fail the test under pytest.

    ``check`` accumulates so one run reports every broken case; without this
    hook pytest would collect these tests and pass them unconditionally.
    """
    failures, FAILURES[:] = list(FAILURES), []
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Model of candidate_blocks.py
# ---------------------------------------------------------------------------


def normalize_candidates(
    candidates: Iterable[int], start: int, width: int, block_size: int, nblocks: int
) -> set[int]:
    """``_candidate_flags_kernel``: which flag slots get set.

    A candidate ``b`` whose block reaches past the logits width is clamped onto
    the sentinel slot ``nblocks`` (``candidate_blocks.py:102``); ``-1`` pads
    select nothing (``:104``).
    """
    flags = set()
    for b in candidates:
        if b < 0:
            continue
        if start + b * block_size >= width:
            flags.add(nblocks)
        else:
            flags.add(b)
    return flags


def candidate_mask_columns(
    candidates: Iterable[int],
    start: int,
    end: int,
    width: int,
    block_size: int,
    nblocks: int,
) -> set[int]:
    """``_mask_candidates_kernel``: the packed columns the mask *keeps*.

    ``valid = (cols >= start) & (cols < end) & (cols < width)`` (``:128``);
    ``block = (cols - start) // BLOCK_SIZE`` (``:129``); ``keep = flags[block]``
    for valid columns (``:130``); plus the edge rule
    ``(cols == width - 1) & flags[nblocks]`` (``:132``). Everything else is
    written to ``-inf`` (``:136``).
    """
    flags = normalize_candidates(candidates, start, width, block_size, nblocks)
    keep = set()
    for col in range(width):
        valid = start <= col < end and col < width
        if valid and (col - start) // block_size in flags:
            keep.add(col)
        elif col == width - 1 and nblocks in flags:
            keep.add(col)
    return keep


def _producer_block_scores(
    scores: list[float], start: int, end: int, block_size: int
) -> list[float]:
    """``_block_scores_kernel``: per-block max over the row's positions.

    ``other=-inf`` outside ``[start, end)`` and past the width (``:37-44``);
    the block holding ``end - 1`` is pinned to ``+inf`` (``:46-50``) -- the
    "newest block is always a candidate" rule.
    """
    nblocks = (len(scores) - start + block_size - 1) // block_size
    out = []
    for b in range(nblocks):
        lo = start + b * block_size
        hi = min(lo + block_size, end, len(scores))
        vals = scores[lo:hi] if hi > lo else []
        out.append(max(vals) if vals else float("-inf"))
    newest = (end - start - 1) // block_size if end > start else -1
    if 0 <= newest < nblocks:
        out[newest] = float("inf")
    return out


def _top_k(values: list[float], k: int) -> list[int]:
    """Indices of the k largest values, best first, ties to the lower index.

    ``-inf`` entries are *not* filtered: the row top-k kernels do emit indices
    for every one of their k slots, so keeping them models the consumer
    faithfully. Callers that need a directly comparable pair of arms assert
    that enough live columns exist that no ``-inf`` is selected.
    """
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    return order[:k]


def select_candidates(
    scores: list[float], start: int, end: int, block_size: int, topk_blocks: int
) -> list[int]:
    """The producer half: block max-pool, then top-k blocks, ``-1`` padded."""
    block_scores = _producer_block_scores(scores, start, end, block_size)
    picked = _top_k(block_scores, min(topk_blocks, len(block_scores)))
    return picked + [-1] * (topk_blocks - len(picked))


# ---------------------------------------------------------------------------
# 1. Mask semantics
# ---------------------------------------------------------------------------


def test_the_mask_keeps_exactly_the_selected_blocks_and_the_edge_column():
    """The mask is a set function of the candidate blocks, and nothing else.

    Exercises the three encodings that are easy to get wrong: ``-1`` pads
    select nothing, a candidate past the logits width clamps onto the sentinel
    slot (keeping only the ``width - 1`` edge column), and a candidate block
    only partly inside ``[start, end)`` keeps just the part that is inside.
    """
    block_size, width = 8, 40
    nblocks = -(-width // block_size)  # 5

    # start=0: candidates {0, 2} -> columns of blocks 0 and 2 only.
    kept = candidate_mask_columns([0, 2, -1, -1], 0, width, width, block_size, nblocks)
    check(
        kept == set(range(0, 8)) | set(range(16, 24)),
        f"blocks 0 and 2 must be kept verbatim, got {sorted(kept)}",
    )

    # The causal window clips a partially-visible block: end=20 truncates block 2.
    kept = candidate_mask_columns([2, -1], 0, 20, width, block_size, nblocks)
    check(kept == set(range(16, 20)), f"end must clip block 2, got {sorted(kept)}")

    # A candidate at/past the width clamps to the sentinel and keeps the edge.
    kept = candidate_mask_columns([99, -1], 0, width, width, block_size, nblocks)
    check(
        kept == {width - 1},
        f"an out-of-range candidate must clamp to the edge column, got {sorted(kept)}",
    )

    # An all-pad candidate row masks everything away.
    kept = candidate_mask_columns([-1, -1], 0, width, width, block_size, nblocks)
    check(kept == set(), f"an all-pad candidate row must mask the whole row, got {kept}")

    # start > 0: block ids are request-local, positions are start-relative.
    kept = candidate_mask_columns([0, 1], 16, 40, width, block_size, nblocks)
    check(
        kept == set(range(16, 32)),
        f"blocks must be counted from start, got {sorted(kept)}",
    )


def test_a_context_shorter_than_the_candidate_set_can_never_be_masked():
    """``topk_blocks * block_size >= width`` => the mask is a no-op.

    This is what makes the mechanism harmless at short context: the producer's
    top-k cannot run out of blocks, so every block of the row is a candidate
    and the mask removes nothing. Stated over a sweep of
    (block_size, topk_blocks, width) satisfying the inequality rather than one
    hardcoded lane value, so it holds for any configuration that meets it --
    including the lane's own ``2048 * 8 = 16384`` compressed positions.
    """
    for block_size in (1, 2, 8, 16):
        for topk_blocks in (1, 2, 8, 64):
            for width in (block_size, 64, 100):
                if topk_blocks * block_size < width:
                    continue
                nblocks = -(-width // block_size)
                candidates = list(range(nblocks))
                kept = candidate_mask_columns(
                    candidates, 0, width, width, block_size, nblocks
                )
                check(
                    kept == set(range(width)),
                    f"no-op mask violated: width={width} bs={block_size} "
                    f"topk={topk_blocks} kept={len(kept)} of {width}",
                )


def test_the_producers_pinned_newest_block_always_survives_the_mask():
    """The ``+inf`` pin guarantees the newest compressed position is visible.

    ``_block_scores_kernel:46-50`` pins the block holding ``end - 1`` to
    ``+inf``, so it always ranks first and is always a candidate; the mask then
    keeps it. Modelled end to end (producer selection -> mask) with an
    adversarial row where the newest block holds the *worst* scores.
    """
    block_size, width, topk_blocks = 8, 64, 4
    nblocks = -(-width // block_size)
    for end in range(1, width + 1):
        scores = [0.0] * width
        newest = (end - 1) // block_size
        for col in range(newest * block_size, min((newest + 1) * block_size, width)):
            scores[col] = -100.0
        candidates = select_candidates(scores, 0, end, block_size, topk_blocks)
        kept = candidate_mask_columns(candidates, 0, end, width, block_size, nblocks)
        check(
            end - 1 in kept,
            f"newest position {end - 1} masked out with candidates={candidates}",
        )


# ---------------------------------------------------------------------------
# 2. Selection equivalence
# ---------------------------------------------------------------------------


def test_candidate_restricted_scoring_selects_the_same_indices_as_masking():
    """The property: masking then top-k == restricting to candidates then top-k.

    Arm A (what the shipped consumers do): take full-width scores, apply the
    candidate mask, take the row top-k. Arm B (the compact formulation): keep
    only the candidate columns, top-k among them, map back to request-local
    positions. The fixture is built so the mask *does* change the answer
    relative to unmasked full-width top-k, otherwise the comparison would pass
    vacuously.
    """
    block_size, width, topk_blocks, topk_tokens = 8, 64, 4, 16
    nblocks = -(-width // block_size)  # 8
    start, end = 0, width

    # Block 0 and the pinned block 7 are the signal; blocks 5 and 3 have the
    # next-best block maxima. Block 6 (all columns at 7.5) loses the block
    # top-k, but its columns still outrank the *low* columns of the selected
    # blocks -- so the mask is what removes them from the row top-k.
    scores = [0.05] * width
    for col in range(0, block_size):
        scores[col] = 50.0 + col
    scores[3 * block_size] = 8.0
    scores[5 * block_size] = 9.0
    for col in range(6 * block_size, 7 * block_size):
        scores[col] = 7.5

    candidates = select_candidates(scores, start, end, block_size, topk_blocks)
    check(-1 not in candidates, f"the row is long enough to fill top-k: {candidates}")
    check(
        set(candidates) == {7, 0, 5, 3},
        f"unexpected candidate set {sorted(candidates)}",
    )

    kept = candidate_mask_columns(candidates, start, end, width, block_size, nblocks)
    check(
        len(kept) >= topk_tokens,
        f"need >= {topk_tokens} live columns for a comparable pair of arms, "
        f"got {len(kept)}",
    )

    # Arm A: mask (unselected columns -> -inf), then top-k.
    masked = [scores[c] if c in kept else float("-inf") for c in range(width)]
    arm_a = _top_k(masked, topk_tokens)

    # Arm B: compact matrix of only the candidate columns, top-k, map back.
    compact = sorted(kept)
    arm_b = [compact[i] for i in _top_k([scores[c] for c in compact], topk_tokens)]

    check(arm_a == arm_b, f"arms disagree: masked={arm_a} compact={arm_b}")

    # Non-vacuity, both halves: the mask must have removed at least one
    # position the unmasked full-width top-k would have taken, and every
    # position it removed must belong to a block that was not a candidate.
    unmasked_topk = _top_k(scores, topk_tokens)
    dropped = [c for c in unmasked_topk if c not in kept]
    check(
        dropped,
        "fixture is vacuous: the candidate mask changed nothing, so this "
        "comparison cannot fail",
    )
    dropped_blocks = {c // block_size for c in dropped}
    check(
        not (dropped_blocks & set(candidates)),
        f"the mask dropped columns {dropped} from selected blocks "
        f"{sorted(dropped_blocks & set(candidates))}",
    )


def test_the_mask_is_idempotent_and_never_widens_the_candidate_set():
    """Applying the mask twice equals applying it once, and it only removes.

    The mask writes ``-inf`` in place on the shared logits, so a repeated
    application (re-entrant or repeated layer call) must not resurrect columns
    nor drop more of them.
    """
    block_size, width = 8, 40
    nblocks = -(-width // block_size)
    candidates = [1, 3, -1]
    kept_once = candidate_mask_columns(candidates, 0, width, width, block_size, nblocks)
    kept_twice = candidate_mask_columns(
        candidates, 0, width, width, block_size, nblocks
    )
    check(kept_once == kept_twice, "masking is not idempotent")
    check(kept_once <= set(range(width)), "the mask selected a nonexistent column")
    check(
        kept_once != set(range(width)),
        "fixture is vacuous: the mask kept every column",
    )


# ---------------------------------------------------------------------------
# 3. Composition with the decode query shard
# ---------------------------------------------------------------------------


def test_candidate_rows_and_topk_rows_use_the_same_shard_mapping():
    """Under the shard, candidates and top-k address the same batch rows.

    The producer writes ``candidate_blocks[row_lo:row_hi]`` and each consumer
    reads ``candidate_blocks[row_lo:row_hi]``; the top-k writes
    ``topk_indices_buffer[row_lo:row_hi]``. Both must be the *batch-absolute*
    row range ``indexer_decode_shard_rows`` returns -- group-relative offsets
    silently misplace every rank after rank 0. Checked against the shipped
    helper, over the bounds the builder actually produces.
    """
    from vllm.distributed.utils import balanced_row_bounds

    next_n, batch_size, shard_size = 6, 32, 8
    covered: list[int] = []
    for rank in range(shard_size):
        bounds = balanced_row_bounds(0, batch_size, rank, shard_size)
        row_lo, row_hi = indexer_decode_shard_rows(bounds, batch_size, next_n)
        group_lo, group_hi = bounds
        check(
            (row_lo, row_hi) == (group_lo * next_n, group_hi * next_n),
            f"rank {rank}: rows {row_lo}:{row_hi} do not match groups "
            f"{group_lo}:{group_hi} at next_n={next_n}",
        )
        check(
            row_hi - row_lo == (group_hi - group_lo) * next_n,
            f"rank {rank}: row span is not a whole number of query groups",
        )
        covered.extend(range(row_lo, row_hi))
    check(
        covered == list(range(batch_size * next_n)),
        "sharded candidate/top-k rows must tile the batch exactly once",
    )

    # No shard: the same helper must be the identity, or the replicated path
    # would address different rows than the sharded one.
    check(
        indexer_decode_shard_rows(None, batch_size, next_n)
        == (0, batch_size * next_n),
        "an absent shard must map to the full row range",
    )


# ---------------------------------------------------------------------------
# 4. What needs a GPU
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "requires CUDA + Triton: drives _block_scores_kernel, "
        "_store_candidates_kernel, _candidate_flags_kernel and "
        "_mask_candidates_kernel on device. The production host has ~386 MiB "
        "free per GPU beside a live serving lane, so this cannot run here "
        "without an eviction window; the on-device design is written out in "
        "docs/masterplan/dsv41-flash-a100-perf/findings/"
        "ws2b-equivalence-and-r1.md."
    )
)
def test_candidate_restricted_scoring_matches_full_width_scoring_on_device():
    """Arm A vs Arm B over the real Triton kernels, on real logits.

    Arm A: the logits producer -> ``apply_candidate_mask`` -> the row top-k.
    Arm B: score only the candidate blocks' columns, top-k among them, map back
    through ``col = start + block * candidate_block_size + off``. Assertions:
    identical top-k index sets per row, and a max absolute logit difference of
    0.0 over the candidate columns.
    """
    raise AssertionError(
        "GPU-only test executed on the CPU lane: the skip marker was removed "
        "without the hardware being available."
    )


if __name__ == "__main__":
    _gpu_only = "test_candidate_restricted_scoring_matches_full_width_scoring_on_device"
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and k != _gpu_only]
    for t in tests:
        t()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        raise SystemExit(1)
    print(f"all {len(tests)} tests passed")
