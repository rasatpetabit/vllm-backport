"""CPU tests for the seam between the TP indexer query shard and DeepSeek-V4.1's
two-level candidate-block buffer: both mechanisms must agree on who owns which
top-k rows.

Why this file exists. The shard splits the decode indexer's query groups across
TP ranks (``indexer_decode_shard_bounds`` -> ``indexer_decode_shard_rows``), and
the V4.1 candidate buffer is written by one layer (``candidate_source_layer``,
``candidate_write=True``) and masked with by later layers (``uses_candidates``).
Neither mechanism has a test that composes them: the shard tests pin the row
arithmetic on its own, and the candidate buffer is never partitioned at all.
So a regression that left the two disagreeing -- a rank writing candidate rows
it does not own, or reading another rank's -- would produce *valid-looking*
indices and a perf A/B would show nothing but a score wobble.

Every assertion here is on shipped pure functions composed in the order the
runner composes them. Where a side is re-derived it is re-derived from the one
partition rule (``balanced_row_counts``) rather than from the helper under test,
because two calls to one helper agreeing with each other is not evidence.

Runs without a GPU on purpose, like its siblings: the regressions guarded here
are silent, so their guards must not depend on scarce hardware.
"""

import re
from typing import Any, cast

import pytest

from vllm.distributed.utils import balanced_row_counts
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillChunkMetadata,
    indexer_decode_shard_bounds,
    indexer_decode_shard_rows,
    indexer_q_row_ranges,
)

FAILURES: list[str] = []

# The standing DSv4 decode shape: next_n 6 (num_speculative_tokens=5), 8 TP
# ranks, and the priced crossover for the decode half.
NEXT_N = 6
SHARD_SIZE = 8
MIN_REQS = 4

# The V4.1 candidate buffer's layer roles (vllm/models/deepseek_v4_1/nvidia/
# model.py:439-448 and attention.py:398-399). Layer 20 publishes, the ratio-1
# indexers 24/28/32/36 mask with it.
CANDIDATE_SOURCE_LAYER = 20
CONSUMER_LAYERS = (24, 28, 32, 36)

# The batch shapes the lane captures for cudagraph replay, all with next_n=6.
GRAPH_PAD_BATCHES = (6, 12, 24, 48, 96, 128, 192, 256, 384)


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


def check_exact_line(src: str, line: str, msg: str) -> None:
    """Anchor a shipped statement as a whole source line, not a substring.

    `line in src` passes when the pinned statement was deleted from the real
    site but an identical string survives elsewhere -- a comment, a docstring,
    a second overload. Anchoring the whole line (indentation aside) makes the
    pin exact, so one occurrence is one statement.
    """
    check(
        re.search(rf"^\s*{re.escape(line)}\s*$", src, re.MULTILINE) is not None,
        msg,
    )


def _line_span(src: str, line: str) -> tuple[int, int] | None:
    """Span of the whole-line match for ``line``, or None if absent."""
    match = re.search(rf"^[ \t]*{re.escape(line)}[ \t]*$", src, re.MULTILINE)
    return match.span() if match else None


def check_gap_statement_set(
    src: str, first: str, last: str, identifiers: tuple, expected: tuple, msg: str
) -> None:
    """Anchor what happens *between* two shipped statements.

    Two line anchors pin two statements and nothing in between, so a new
    statement inserted in the gap leaves both green while changing behaviour.
    This pins the gap's content: the set of non-comment source lines between
    ``first`` and ``last`` that mention any of ``identifiers`` must be exactly
    ``expected`` (each a stripped source line). Comment lines are ignored so
    rewording a comment is not a false failure.

    If either endpoint line is missing, the line anchors already report it and
    this check stays silent rather than double-reporting a derived failure.
    """
    first_span = _line_span(src, first)
    last_span = _line_span(src, last)
    if first_span is None or last_span is None:
        return
    gap = src[first_span[1] : last_span[0]]
    found = {
        line.strip()
        for line in gap.splitlines()
        if not line.strip().startswith("#")
        and any(name in line for name in identifiers)
    }
    check(
        found == set(expected),
        f"{msg}; the gap between the two anchored statements now contains "
        f"{sorted(found)}, expected {sorted(expected)}",
    )


def teardown_function(function) -> None:
    """Make the collected failures fail the test under pytest.

    `check` accumulates so that one run reports every broken case, which the
    __main__ runner below prints in a batch. Without this hook a pytest run
    would collect these tests and pass them unconditionally.
    """
    failures, FAILURES[:] = list(FAILURES), []
    assert not failures, "\n".join(failures)


def _layer_roles(layer_id: int, source_layer: int = CANDIDATE_SOURCE_LAYER) -> dict:
    """Mirror of the three predicates the decoder layer sets per layer.

    Kept as a local restatement of the two shipped one-liners
    (``layer_id == candidate_source_layer`` / ``0 <= candidate_source_layer <
    layer_id``) so the roles are visible in the assertions below; the
    ``test_layer_roles_match_the_shipped_predicates`` case pins it against the
    source so this cannot drift into a fiction.
    """
    is_source = layer_id == source_layer
    return {
        "candidate_write": is_source,
        "uses_candidates": 0 <= source_layer < layer_id,
        "has_buffer": is_source or 0 <= source_layer < layer_id,
    }


def _bounds(batch_size, rank, shard_size=SHARD_SIZE, num_decodes=None, min_reqs=MIN_REQS):
    return indexer_decode_shard_bounds(
        batch_size,
        batch_size if num_decodes is None else num_decodes,
        rank,
        shard_size,
        min_reqs,
    )


def _composed_rows(batch_size, rank, next_n=NEXT_N, **kw) -> tuple[int, int]:
    """The row range the runner derives: bounds, then rows (sparse_attn_indexer
    lines 785-786). This is the composition under test, not a helper."""
    return indexer_decode_shard_rows(_bounds(batch_size, rank, **kw), batch_size, next_n)


def _independent_rows(batch_size, rank, next_n=NEXT_N, shard_size=SHARD_SIZE):
    """The same ownership re-derived from the single partition rule.

    ``balanced_row_counts`` is the one rule every row-sharding feature must
    derive from, and the candidate buffer is indexed by batch token, so query
    group ``g`` owns rows ``[g*next_n, (g+1)*next_n)``. Nothing here calls the
    helper under test -- that is the point.

    Below the shard size the shipped gate declines and every rank owns the
    whole buffer; that case is modelled here rather than special-cased at each
    call site, so the independent derivation stays the single source of truth
    for what ownership *should* be.
    """
    if batch_size < shard_size:
        return 0, batch_size, 0, batch_size * next_n
    counts = balanced_row_counts(batch_size, shard_size)
    lo = sum(counts[:rank])
    hi = lo + counts[rank]
    return lo, hi, lo * next_n, hi * next_n


def _consumer_rows(
    batch_size: int,
    rank: int,
    layer: int,
    next_n: int = NEXT_N,
    shard_size: int = SHARD_SIZE,
) -> tuple[int, int]:
    """The candidate-buffer rows consumer ``layer`` reads on ``rank``.

    This is the consumer side, and it is derived from a different source than
    the producer's ``_composed_rows``: it goes to the partition primitive
    (``balanced_row_counts``) directly and reconstructs the read range in the
    batch-absolute token axis, which is the axis ``candidate_blocks`` is
    indexed by (``vllm/model_executor/layers/sparse_attn_indexer.py:863``).
    Nothing here calls ``indexer_decode_shard_bounds`` or
    ``indexer_decode_shard_rows`` -- the two helpers the producer side is
    composed from. Reusing them on both sides is what made the comparison at
    this call site unfalsifiable.

    ``layer`` is load-bearing in two ways. A layer that does not consume
    candidates has no read range at all, and one constructed without a
    candidate buffer has nothing to read; both gates are asserted here against
    the per-layer role predicates, so a change to them
    (``vllm/models/deepseek_v4_1/attention.py:398-399``) surfaces at this call
    site rather than silently leaving the consumers' comparison vacuous.
    """
    roles = _layer_roles(layer)
    assert roles["uses_candidates"], f"layer {layer} does not consume candidates"
    assert roles["has_buffer"], f"layer {layer} has no candidate buffer to read"
    if batch_size < shard_size:
        # Below the shard size the gate declines and every rank reads the whole
        # buffer -- the replicated path, stated rather than special-cased.
        return 0, batch_size * next_n
    counts = balanced_row_counts(batch_size, shard_size)
    group_lo = sum(counts[:rank])
    group_hi = group_lo + counts[rank]
    # Batch-absolute: the buffer row axis is batch tokens, so a rank's rows
    # begin at its first owned group's token offset, not at 0. A group-relative
    # slice here would read another request's candidate rows on every rank but
    # the first, and would land inside the buffer extent asserted below --
    # which is why the source anchor pins the shipped expression separately.
    return group_lo * next_n, group_hi * next_n


def _chunk(token_start: int, token_end: int, shard_row_counts) -> Any:
    none = cast(Any, None)
    return DeepseekV32IndexerPrefillChunkMetadata(
        block_table=none,
        cu_seqlen_ks=none,
        cu_seqlen_ke=none,
        cu_seq_lens=none,
        token_to_seq=none,
        total_seq_lens=token_end - token_start,
        token_start=token_start,
        token_end=token_end,
        num_reqs=1,
        shard_row_counts=shard_row_counts,
    )


def test_layer_roles_match_the_shipped_predicates():
    """Anchor: the layer roles this file reasons about are the shipped ones.

    Read straight out of the model source, so a renamed or inverted predicate
    fails here rather than leaving the ownership tests asserting about a role
    split that no longer exists.
    """
    import inspect
    from pathlib import Path

    import vllm

    # Read the source text rather than importing the module: importing it pulls
    # in the Triton kernels, which are unavailable on this CPU host. The anchor
    # is the same either way -- a changed predicate changes these lines.
    v41_attention_path = (
        Path(inspect.getfile(vllm)).parent / "models" / "deepseek_v4_1" / "attention.py"
    )
    check(
        v41_attention_path.is_file(),
        f"deepseek_v4_1 attention source not found at {v41_attention_path}",
    )
    src = v41_attention_path.read_text()
    check_exact_line(
        src,
        "is_candidate_source = layer_id == self.candidate_source_layer",
        "candidate source predicate moved or changed shape",
    )
    check_exact_line(
        src,
        "uses_candidates = 0 <= self.candidate_source_layer < layer_id",
        "candidate consumer predicate moved or changed shape",
    )
    # The consumer's read range is the one shipped expression
    # `candidate_blocks[row_lo:row_hi]` at sparse_attn_indexer.py:863. Pinned as
    # source text because that expression is the consumer side's only anchor in
    # the shipped tree: both roles reach it through the single call site, so no
    # CPU-visible signature distinguishes them.
    indexer_path = Path(inspect.getfile(vllm)).parent / (
        "model_executor/layers/sparse_attn_indexer.py"
    )
    check(
        indexer_path.is_file(),
        f"sparse_attn_indexer source not found at {indexer_path}",
    )
    indexer_src = indexer_path.read_text()
    check_exact_line(
        indexer_src,
        "decode_candidates = candidate_blocks[row_lo:row_hi]",
        "the consumer's candidate slice moved or changed shape -- the read "
        "range `_consumer_rows` models is no longer what layer 24/28/32/36 "
        "actually reads",
    )
    # ...and the provenance of the two bounds that slice consumes. The anchors
    # above pin the bounds' *computation* (the shard-bounds call) and their
    # *consumption* (the slice), but nothing between them: an offset, clamp,
    # reorder, or added per-layer/write-flag branch inserted in that gap leaves
    # both pinned strings intact. The arithmetic side cannot catch it either --
    # this module is never imported on a CPU host, so only source text pins
    # the chain. Between the two statements below, `row_lo`/`row_hi` are read
    # exactly once (`shard_weights = weights[row_lo:row_hi]`, an unmodified
    # read) and never reassigned, so the chain is these two assignments.
    check_exact_line(
        indexer_src,
        "row_lo, row_hi = indexer_decode_shard_rows(shard_bounds, batch_size, next_n)",
        "the decode rows the consumer slices are no longer derived from the "
        "shard bounds in one assignment -- the bounds were offset, clamped, "
        "reordered, or produced by a new branch between the shard-bounds call "
        "and the candidate slice, and neither the slice anchor nor the "
        "arithmetic comparison covers that gap",
    )
    check_exact_line(
        indexer_src,
        "shard_weights = weights[row_lo:row_hi]",
        "the only read of row_lo/row_hi between the shard-bounds call and the "
        "candidate slice moved or changed shape -- a new statement in that gap "
        "is where an unowned row range would be introduced",
    )
    # The two anchors above pin the endpoints of the chain, and this pins the
    # gap between them: `row_lo`/`row_hi` are assigned once from the shard
    # bounds and referenced exactly once before the candidate slice. Anything
    # else in the gap touching them -- an offset, a clamp, a reorder, a
    # per-layer or write-flag branch -- changes which rows the consumer reads
    # while leaving both adjacent anchors intact.
    check_gap_statement_set(
        indexer_src,
        "row_lo, row_hi = indexer_decode_shard_rows(shard_bounds, batch_size, next_n)",
        "decode_candidates = candidate_blocks[row_lo:row_hi]",
        ("row_lo", "row_hi"),
        ("shard_weights = weights[row_lo:row_hi]",),
        "row_lo/row_hi are no longer carried unchanged from the shard-bounds "
        "call to the candidate slice -- an unowned row range was introduced "
        "in the gap, and neither the slice anchor nor the bounds anchor sees it",
    )
    # ...and the partition it reads by is the one rule, not a second policy.
    # `balanced_row_counts` is the single authoritative partition primitive
    # (vllm/distributed/utils.py:127-165: "Every row-sharding feature ... must
    # derive from this one rule"), and the shard bounds are a thin call into
    # it -- so deriving the consumer side from it directly is derivation from
    # the source of truth, not from the helper under test.
    shard_src = (
        Path(inspect.getfile(vllm)).parent
        / "v1/attention/backends/mla/indexer.py"
    ).read_text()
    check_exact_line(
        shard_src,
        "return balanced_row_bounds(0, batch_size, shard_rank, shard_size)",
        "the decode shard bounds no longer derive from the one partition rule",
    )
    check(
        _layer_roles(CANDIDATE_SOURCE_LAYER)["candidate_write"]
        and not _layer_roles(CANDIDATE_SOURCE_LAYER)["uses_candidates"],
        "layer 20 must write candidates and must not consume them",
    )
    for layer in CONSUMER_LAYERS:
        roles = _layer_roles(layer)
        check(
            roles["uses_candidates"] and not roles["candidate_write"],
            f"layer {layer} must read candidates and must not write them",
        )
        check(roles["has_buffer"], f"layer {layer} must have a candidate buffer")
    for layer in range(0, CANDIDATE_SOURCE_LAYER):
        check(
            not _layer_roles(layer)["has_buffer"],
            f"layer {layer} precedes the source and must not touch candidates",
        )


def test_producer_and_consumer_agree_on_row_ownership():
    """Case 1: the writer's rows are the readers' rows, at every rank.

    For batch sizes 1..40 the range layer 20 writes into ``candidate_blocks``
    (``_select_candidate_blocks`` over ``candidate_blocks[row_lo:row_hi]``)
    equals the range each of layers 24/28/32/36 reads back
    (``decode_candidates = candidate_blocks[row_lo:row_hi]``). Both sides are
    the composed shipped helpers, and both are additionally pinned against the
    independently derived ownership, so a change to either side that the other
    did not follow fails here.
    """
    producer = _layer_roles(CANDIDATE_SOURCE_LAYER)
    check(producer["candidate_write"], "the producer role must write")
    # From the shard size up, so the producer/consumer comparison is against a
    # real partition; the sub-shard replicated case is case 4's boundary.
    for batch in range(SHARD_SIZE, 41):
        for rank in range(SHARD_SIZE):
            write_lo, write_hi = _composed_rows(batch, rank)
            gl, gh, exp_lo, exp_hi = _independent_rows(batch, rank)
            check(
                (write_lo, write_hi) == (exp_lo, exp_hi),
                f"batch={batch} rank={rank}: producer writes [{write_lo},"
                f"{write_hi}) but owns groups [{gl},{gh}) -> [{exp_lo},{exp_hi})",
            )
            for layer in CONSUMER_LAYERS:
                roles = _layer_roles(layer)
                check(roles["uses_candidates"], f"layer {layer} must consume")
                # The consumer side comes from its own derivation
                # (`_consumer_rows`: the partition primitive plus the
                # batch-absolute buffer axis), not from the producer's
                # `_composed_rows`. Comparing one pure function with itself
                # cannot fail, which is the defect this case used to carry.
                read_lo, read_hi = _consumer_rows(batch, rank, layer)
                check(
                    (read_lo, read_hi) == (write_lo, write_hi),
                    f"batch={batch} rank={rank}: layer {layer} reads "
                    f"[{read_lo},{read_hi}) but layer {CANDIDATE_SOURCE_LAYER} "
                    f"wrote [{write_lo},{write_hi})",
                )
                check(
                    (read_lo, read_hi) == (exp_lo, exp_hi),
                    f"batch={batch} rank={rank}: layer {layer} reads "
                    f"[{read_lo},{read_hi}), not the owned [{exp_lo},{exp_hi})",
                )
                # New content: the read must land inside the buffer's
                # batch-absolute row extent and stay request-aligned, so a
                # consumer that read past the buffer or straddled a request
                # fails even where the range happened to be the owned one.
                check(
                    read_lo < read_hi <= batch * NEXT_N
                    and read_lo % NEXT_N == 0
                    and read_hi % NEXT_N == 0,
                    f"batch={batch} rank={rank}: layer {layer} reads "
                    f"[{read_lo},{read_hi}) outside the batch-absolute buffer "
                    f"extent [0, {batch * NEXT_N})",
                )


def test_row_ownership_is_batch_absolute_not_group_relative():
    """The offset guard, stated against the candidate buffer.

    ``candidate_blocks`` is indexed by batch token, so a rank's rows start at
    its first owned group's batch-absolute offset. Group-relative offsets are
    exactly right on rank 0 and put every later rank's candidate rows on
    another request's rows -- which is why the independent derivation above
    multiplies by ``next_n`` rather than trusting the helper.
    """
    for batch in (24, 132, 133, 512):
        for rank in range(SHARD_SIZE):
            lo, hi = _composed_rows(batch, rank)
            gl, _, exp_lo, _ = _independent_rows(batch, rank)
            check(
                lo == gl * NEXT_N,
                f"batch={batch} rank={rank}: rows start at {lo}, not at the "
                f"batch-absolute {gl * NEXT_N}",
            )
            if gl > 0:
                check(
                    lo != 0,
                    f"batch={batch} rank={rank}: row range starts at 0 for a "
                    "rank that does not own the first group",
                )


def test_uneven_partitions_tile_the_buffer_exactly():
    """Case 2: where batch % shard_size != 0 the ranks tile the rows exactly.

    The candidate buffer and ``topk_indices_buffer`` are reduced by summing
    across ranks, so a row owned twice is written twice and a row owned by none
    arrives as zeros -- both valid-looking. Exactness is what makes the
    reduction legal.
    """
    uneven = [b for b in range(SHARD_SIZE, 41) if b % SHARD_SIZE != 0]
    check(
        set(b % SHARD_SIZE for b in uneven) == set(range(1, SHARD_SIZE)),
        "the uneven set must exercise every non-zero remainder, got "
        f"{sorted(set(b % SHARD_SIZE for b in uneven))}",
    )
    for batch in uneven:
        rows: list[int] = []
        groups: list[int] = []
        for rank in range(SHARD_SIZE):
            bounds = _bounds(batch, rank)
            if bounds is None:
                continue
            groups.extend(range(*bounds))
            lo, hi = _composed_rows(batch, rank)
            rows.extend(range(lo, hi))
        check(
            groups == list(range(batch)),
            f"batch={batch}: query groups do not tile [0, {batch})",
        )
        check(
            rows == list(range(batch * NEXT_N)),
            f"batch={batch}: rows do not tile [0, {batch * NEXT_N})",
        )
        check(len(set(rows)) == len(rows), f"batch={batch}: overlapping rows")


def test_unowned_rows_are_never_read_by_a_rank():
    """Case 3: a rank reads only rows inside its own range.

    "Never read" is expressed over the pure slicing arithmetic: for rank r the
    buffer slice ``[row_lo, row_hi)`` contains no row outside the independently
    derived owned set, and every row outside it is owned by exactly one other
    rank. So no rank reads a row it does not own, and no row goes unwritten.
    """
    for batch in (24, 48, 133, 256):
        # The exclusive-ownership map is only meaningful once the gate opens;
        # below the shard size every rank deliberately owns the whole buffer.
        owner_of: dict[int, int] = {}
        for rank in range(SHARD_SIZE):
            lo, hi = _composed_rows(batch, rank)
            for row in range(lo, hi):
                check(
                    row not in owner_of,
                    f"batch={batch} row={row} owned by ranks "
                    f"{owner_of.get(row)} and {rank}",
                )
                owner_of[row] = rank
        check(
            sorted(owner_of) == list(range(batch * NEXT_N)),
            f"batch={batch}: some row is owned by no rank",
        )
        for rank in range(SHARD_SIZE):
            lo, hi = _composed_rows(batch, rank)
            _, _, exp_lo, exp_hi = _independent_rows(batch, rank)
            for row in range(batch * NEXT_N):
                in_slice = lo <= row < hi
                owned_by_rank = exp_lo <= row < exp_hi
                check(
                    in_slice == owned_by_rank,
                    f"batch={batch} rank={rank}: row {row} is "
                    f"{'read' if in_slice else 'not read'} but is "
                    f"{'owned' if owned_by_rank else 'not owned'}",
                )
                check(
                    in_slice == (owner_of[row] == rank),
                    f"batch={batch} rank={rank}: row {row} slice disagrees "
                    "with the ownership map",
                )
    # The sub-shard case, stated separately: there the gate declines and each
    # rank's slice is the whole buffer, so no rank reads a row outside what the
    # replicated path would read.
    for batch in (1, 7):
        for rank in range(SHARD_SIZE):
            lo, hi = _composed_rows(batch, rank)
            check(
                (lo, hi) == (0, batch * NEXT_N),
                f"sub-shard batch={batch} rank={rank}: replicated path must own "
                f"the whole buffer, got [{lo},{hi})",
            )


def test_threshold_transitions_are_exact():
    """Case 4: None exactly when num_decodes < min_reqs or batch < shard_size.

    Both boundaries are exercised either side: 3 vs 4 decode requests, and a
    batch of 7 vs 8. "Exactly" means the equivalence, not just that the gate
    closes somewhere -- a gate that stayed open one request early would
    partition a batch the collective was not sized for, and one that closed
    late would launch kernels over an empty row range.
    """
    for batch in range(1, 41):
        for num_decodes in range(0, 41):
            bounds = _bounds(batch, 0, num_decodes=num_decodes)
            expected_none = num_decodes < MIN_REQS or batch < SHARD_SIZE
            check(
                (bounds is None) == expected_none,
                f"batch={batch} decodes={num_decodes}: bounds={bounds}, "
                f"expected {'None' if expected_none else 'a real range'}",
            )
    # The named boundaries, both directions.
    check(_bounds(8, 0, num_decodes=3) is None, "3 decodes must not shard")
    check(_bounds(8, 0, num_decodes=4) is not None, "4 decodes must shard")
    check(_bounds(7, 0, num_decodes=8) is None, "batch 7 < shard 8 must not shard")
    check(_bounds(8, 0, num_decodes=8) is not None, "batch 8 == shard 8 must shard")
    # A real range is a real range at both boundaries: strictly inside.
    for batch, num_decodes in ((8, 4), (9, 4), (40, 40)):
        for rank in range(SHARD_SIZE):
            bounds = _bounds(batch, rank, num_decodes=num_decodes)
            assert bounds is not None
            lo, hi = bounds
            check(
                0 <= lo < hi <= batch,
                f"batch={batch} rank={rank}: degenerate bounds {bounds}",
            )
    # shard_size == 1 (tp=1 / DCP / PCP / flag off) and min_reqs == 0 (the
    # documented opt-out) both mean "compute every group".
    for batch in (8, 132):
        for rank in range(SHARD_SIZE):
            check(
                _bounds(batch, rank, shard_size=1) is None,
                f"batch={batch} rank={rank}: shard_size 1 must be replicated",
            )
            check(
                _bounds(batch, rank, min_reqs=0) is None,
                f"batch={batch} rank={rank}: min_reqs 0 must be replicated",
            )


def test_flattened_next_n_6_maps_groups_to_token_rows():
    """Case 5: on the flattening path batch_size counts tokens, not requests.

    SM80 cannot pass next_n=6 Q rows per request to the decode kernel
    (``_supports_native_decode(6)`` is False at capability 8.0), so the path
    flattens and each query group is one token row. The row mapping is then
    group ``g`` -> rows ``[g*next_n, (g+1)*next_n)`` over a batch_size that is
    the flattened token count, and the candidate buffer's row ownership follows
    it unchanged.
    """
    # The flattening decision is capability-dependent and cannot be reproduced
    # on a CPU host (see test_sm80_native_decode_decision_is_gpu_only); what is
    # CPU-testable is the mapping the flattened path then relies on.
    for batch in (24, 48, 132, 133, 384):
        for rank in range(SHARD_SIZE):
            lo, hi = _composed_rows(batch, rank)
            gl, gh, exp_lo, exp_hi = _independent_rows(batch, rank)
            check(
                (lo, hi) == (exp_lo, exp_hi),
                f"flattened batch={batch} rank={rank}: rows [{lo},{hi}) != "
                f"token rows of groups [{gl},{gh}) = [{exp_lo},{exp_hi})",
            )
            check(
                (hi - lo) == (gh - gl) * NEXT_N,
                f"flattened batch={batch} rank={rank}: {hi - lo} rows for "
                f"{gh - gl} groups is not a whole number of next_n blocks",
            )
            check(
                lo % NEXT_N == 0 and hi % NEXT_N == 0,
                f"flattened batch={batch} rank={rank}: a request's next_n rows "
                "would straddle two ranks",
            )


def test_graph_padded_batch_shapes_partition_coherently():
    """Case 6: every captured shape the lane replays partitions coherently.

    A full-cudagraph decode replays with the padded row count baked in, so each
    captured shape must partition exactly and at most one group's worth
    unevenly, or capture and replay would disagree about who owns which rows.
    Shapes below the shard size stay replicated, and that is asserted rather
    than skipped: the replicated path owns the whole buffer.
    """
    for batch in GRAPH_PAD_BATCHES:
        replicated = batch < SHARD_SIZE
        rows: list[int] = []
        counts: list[int] = []
        for rank in range(SHARD_SIZE):
            bounds = _bounds(batch, rank)
            if replicated:
                check(
                    bounds is None,
                    f"padded batch={batch} rank={rank}: must be replicated",
                )
                lo, hi = indexer_decode_shard_rows(None, batch, NEXT_N)
                check(
                    (lo, hi) == (0, batch * NEXT_N),
                    f"padded batch={batch} rank={rank}: replicated path must "
                    f"own the whole buffer, got [{lo},{hi})",
                )
                rows = list(range(batch * NEXT_N))
                counts = [batch] * SHARD_SIZE
                break
            assert bounds is not None
            rows.extend(range(*_composed_rows(batch, rank)))
            counts.append(bounds[1] - bounds[0])
        check(
            rows == list(range(batch * NEXT_N)),
            f"padded batch={batch}: rows do not tile [0, {batch * NEXT_N})",
        )
        check(
            max(counts) - min(counts) <= 1,
            f"padded batch={batch}: imbalance > 1 group, counts={counts}",
        )


def test_mixed_prefill_and_decode_halves_cannot_disagree():
    """Case 7: the Q path declines on a mixed batch, and that is the safe side.

    ``indexer_q_row_ranges`` returns None when ``num_decodes > 0``: the decode
    branch reads ``q_quant[:num_decode_tokens]`` and ``weights[:batch*next_n]``,
    ranges no chunk names. The decode half may still shard. The two halves can
    only disagree if the declining half covers *less* than the sharding half
    writes, so what is asserted is that the Q path's full-range answer is a
    superset of every rank's decode rows -- no row is left written by neither,
    and the Q path cannot claim a partition it did not compute.
    """
    batch = 132
    chunks = [_chunk(64, 64 + 4096, balanced_row_counts(4096, SHARD_SIZE))]
    check(
        indexer_q_row_ranges(chunks, 1, 64 + 4096) is None,
        "a batch with decode requests must run the replicated Q path",
    )
    check(
        indexer_q_row_ranges(chunks, 0, 64 + 4096) is not None,
        "the same batch without decode must shard -- otherwise this case "
        "proves nothing about the decline being decode-specific",
    )
    # The Q path's answer when it declines is "every row".
    q_rows = set(range(64 + 4096))
    for rank in range(SHARD_SIZE):
        lo, hi = _composed_rows(batch, rank)
        check(
            set(range(lo, hi)) <= q_rows,
            f"rank={rank}: decode rows [{lo},{hi}) escape the Q path's "
            "full-range answer",
        )
    # And the decode shard is still a proper subrange, i.e. it did not silently
    # fall back to the whole buffer when the Q half declined.
    check(
        any(_composed_rows(batch, r) != (0, batch * NEXT_N) for r in range(SHARD_SIZE)),
        "the decode half must still shard while the Q half declines",
    )
    # A batch with no decode requests must never produce decode bounds for a
    # rank that owns no group (that rank would enter the collective with an
    # empty range); the gate covers it, and the Q half shards over the same
    # partition of rows the chunks name.
    for batch_size in (8, 9, 132):
        for rank in range(SHARD_SIZE):
            check(
                _bounds(batch_size, rank, num_decodes=0) is None,
                f"batch={batch_size} rank={rank}: no decode requests means no "
                "decode shard",
            )


def test_sm80_native_decode_decision_is_gpu_only():
    """Skip-with-reason: the capability-8.0 flattening decision needs a GPU.

    ``_supports_native_decode(6)`` reads ``current_platform.is_cuda()`` and
    ``has_deep_gemm()`` before it ever consults the capability family, so on
    this CPU host it returns False for a reason unrelated to SM80 and asserting
    it here would pass without testing the branch. The CPU-testable consequence
    -- the group-to-token-row mapping the flattened path relies on -- is
    ``test_flattened_next_n_6_maps_groups_to_token_rows``.

    GPU design: on an A100 (capability 8.0) with deep_gemm available, assert
    ``_supports_native_decode(6) is False`` and ``_supports_native_decode(1) is
    True``, and that ``_use_flattening`` is True for a VllmConfig with
    num_speculative_tokens=5 and no adaptive verification; then capture a
    decode batch and assert the metadata builder's ``batch_size`` equals the
    flattened token count rather than the request count.
    """
    pytest.skip(
        "capability-8.0 native-decode decision requires a CUDA device and "
        "deep_gemm; CPU host cannot reach that branch"
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    skipped = 0
    for t in tests:
        try:
            t()
        except pytest.skip.Exception as e:
            skipped += 1
            print(f"SKIPPED {t.__name__}: {e}")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        raise SystemExit(1)
    print(f"all {len(tests) - skipped} tests passed ({skipped} skipped)")
