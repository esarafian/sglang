"""Benchmark the rejection-sampling verify step for a reduced draft vocabulary.

``scatter`` lifts the draft-vocabulary q into the target vocabulary (zeros +
index_copy_) and verifies it; ``mapped`` verifies the draft-vocabulary q
directly through a target-to-draft id map. Both run the same chain kernel on the
same inputs and produce identical outputs.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.speculative.spec_utils import (
    build_target_to_draft_vocab,
    scatter_draft_probs_to_target_vocab,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

NUM_STEPS = 3
DRAFT_VOCAB = 32000


def _make_case(batch_size: int, target_vocab: int):
    torch.manual_seed(0)
    hot = torch.randperm(target_vocab, device="cuda")[:DRAFT_VOCAB]
    q = torch.rand(batch_size, NUM_STEPS, DRAFT_VOCAB, device="cuda").pow(8)
    q /= q.sum(-1, keepdim=True)
    cols = torch.multinomial(q.flatten(0, 1), 1).view(batch_size, NUM_STEPS)
    candidates = torch.zeros(
        batch_size, NUM_STEPS + 1, dtype=torch.int64, device="cuda"
    )
    candidates[:, 1:] = hot[cols]
    target = torch.rand(batch_size, NUM_STEPS + 1, target_vocab, device="cuda")
    target /= target.sum(-1, keepdim=True)
    target[:, :NUM_STEPS] = (
        0.5 * scatter_draft_probs_to_target_vocab(q, hot, target_vocab)
        + 0.5 * target[:, :NUM_STEPS]
    )
    coins = torch.rand(batch_size, NUM_STEPS + 1, device="cuda")
    final_coins = torch.rand(batch_size, device="cuda")
    return hot, q, target, candidates, coins, final_coins


@marker.parametrize("target_vocab", [151936, 248320], [151936])
@marker.parametrize("batch_size", [1, 4, 16, 32, 64, 128], [16, 64])
@marker.benchmark("impl", ["scatter", "mapped"])
def benchmark(target_vocab: int, batch_size: int, impl: str):
    hot, q, target, candidates, coins, final_coins = _make_case(
        batch_size, target_vocab
    )
    target_to_draft = build_target_to_draft_vocab(hot, target_vocab)
    slots = NUM_STEPS + 1
    indices = torch.arange(batch_size * slots, device="cuda").view(batch_size, slots)
    predicts = torch.empty(batch_size * slots, dtype=torch.int32, device="cuda")
    accepted = torch.empty(batch_size, slots, dtype=torch.int32, device="cuda")
    lengths = torch.empty(batch_size, dtype=torch.int32, device="cuda")

    def verify(q: torch.Tensor, target: torch.Tensor) -> None:
        mapping = target_to_draft
        if impl == "scatter":
            q = scatter_draft_probs_to_target_vocab(q, hot, target_vocab)
            mapping = None
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
            q,
            1.0,
            1.0,
            True,
            target_to_draft_vocab=mapping,
        )

    return marker.do_bench(verify, input_args=(q, target))


if __name__ == "__main__":
    benchmark.run()
