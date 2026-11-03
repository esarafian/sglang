"""Chain rejection sampling with q kept in a reduced draft vocabulary.

``chain_speculative_sampling_triton(..., target_to_draft_vocab=...)`` reads q
through a target-id -> draft-column map instead of a q scattered into the target
vocabulary. Both layouts hold the same number for every target id (0.0 off the
draft vocabulary), so every output must be bitwise equal to the scattered run:
accept lengths, accept indices and every predicted token, including the
resampled bonus token. A change to either kernel path that reads q differently
turns this red.
"""

import os
import sys

import pytest
import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.speculative.spec_utils import (
    build_target_to_draft_vocab,
    scatter_draft_probs_to_target_vocab,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-small")

DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


def _run(target, draft, candidates, coins, final_coins, block, target_to_draft):
    batch, slots = candidates.shape
    indices = torch.arange(batch * slots, device=DEVICE).view(batch, slots)
    predicts = torch.full((batch * slots,), -1, dtype=torch.int32, device=DEVICE)
    accepted = torch.full((batch, slots), -1, dtype=torch.int32, device=DEVICE)
    lengths = torch.full((batch,), -1, dtype=torch.int32, device=DEVICE)
    chain_speculative_sampling_triton(
        predicts,
        accepted,
        lengths,
        candidates,
        indices,
        None,
        None,
        coins,
        final_coins,
        target,
        draft,
        1.0,
        1.0,
        True,
        block_verification=block,
        target_to_draft_vocab=target_to_draft,
    )
    return predicts, accepted, lengths


def _make_case(batch, steps, target_vocab, draft_vocab, seed, odd_q):
    gen = torch.Generator().manual_seed(seed)
    hot = torch.randperm(target_vocab, generator=gen)[:draft_vocab]
    q = torch.rand(batch, steps, draft_vocab, generator=gen).pow(8)
    q /= q.sum(-1, keepdim=True)
    cols = torch.multinomial(q.flatten(0, 1), 1, generator=gen).view(batch, steps)
    candidates = torch.zeros(batch, steps + 1, dtype=torch.int64)
    candidates[:, 1:] = hot[cols]
    # A draft token outside the draft vocabulary reads q = 0 and is rejected.
    outside = torch.ones(target_vocab, dtype=torch.bool)
    outside[hot] = False
    candidates[0, 1] = outside.nonzero()[0, 0]
    # Target close to q on the draft vocabulary, so that accepted, rejected and
    # fully accepted rows all occur.
    q_dense = scatter_draft_probs_to_target_vocab(q, hot, target_vocab)
    noise = torch.rand(batch, steps + 1, target_vocab, generator=gen)
    target = noise / noise.sum(-1, keepdim=True)
    target[:, :steps] = 0.8 * q_dense + 0.2 * target[:, :steps]
    if odd_q:
        # Values the kernel must treat as "not a probability", on proposed
        # tokens and inside the residual scans.
        odd = [float("nan"), float("inf"), -1.0, 2.0, 0.0, -0.0, 1e-45, 1.0]
        for i, value in enumerate(odd):
            row, step = i % batch, i % steps
            q[row, step, cols[row, step]] = value
            q[(row + 1) % batch, step, (7 * i) % draft_vocab] = value
    coins = torch.rand(batch, steps + 1, generator=gen)
    final_coins = torch.rand(batch, generator=gen)
    return hot, q, target, candidates, coins, final_coins


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("odd_q", [False, True])
@pytest.mark.parametrize(
    "batch,steps,target_vocab,draft_vocab",
    [(1, 1, 17, 5), (7, 3, 4099, 1000), (65, 3, 151936, 32000), (16, 5, 248320, 32000)],
)
def test_mapped_q_matches_scattered_q(
    batch, steps, target_vocab, draft_vocab, odd_q, block
):
    hot, q, target, candidates, coins, final_coins = _make_case(
        batch, steps, target_vocab, draft_vocab, seed=batch * 131 + steps, odd_q=odd_q
    )
    hot, q, target, candidates, coins, final_coins = (
        x.to(DEVICE) for x in (hot, q, target, candidates, coins, final_coins)
    )
    target_to_draft = build_target_to_draft_vocab(hot, target_vocab)
    assert target_to_draft is not None

    dense = _run(
        target,
        scatter_draft_probs_to_target_vocab(q, hot, target_vocab),
        candidates,
        coins,
        final_coins,
        block,
        None,
    )
    mapped = _run(
        target, q, candidates, coins, final_coins, block, target_to_draft
    )
    for name, expected, actual in zip(("predicts", "accepted", "lengths"), dense, mapped):
        assert torch.equal(expected, actual), name
    # Row 0 proposes a token outside the draft vocabulary, so it rejects at once;
    # the other rows must reach the accept branch.
    lengths = dense[2]
    assert lengths[0] == 0
    if batch > 1:
        assert (lengths[1:] > 0).any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
