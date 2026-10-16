"""slime rollout function for offline GRPO over a SAH replay file.

slime's training loop asks a rollout function for the samples of each step.
Online, that function generates responses; here every response already exists
— they are the proposer trajectories a finished outer round produced — so
this function only tokenises them, computes the loss mask, and attaches the
advantage that ``outer/reward/rewards.py`` already assigned.

Pass it to slime as::

    --rollout-function-path trainer.slime_replay_rollout.generate_rollout
    --prompt-data <replay.jsonl>  --input-key messages
    --tool-key tools --metadata-key metadata
    --loss-mask-type qwen3_5 --n-samples-per-prompt 1
    --advantage-estimator grpo --disable-rewards-normalization

The row's ``metadata.advantage`` becomes ``sample.reward`` because slime's
GRPO estimator broadcasts that field per token; the untouched scalar reward
is kept as ``metadata.raw_reward`` for diagnostics.  Rows are refused rather
than silently skipped when they carry no trainable assistant tokens or a
non-finite advantage — a silently dropped row would change the group
composition that the advantage was normalised over.
"""
from __future__ import annotations

import logging
import math

from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

from trainer.qwen35_mask import Qwen35MultiTurnLossMaskGenerator

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)
TOKENIZER = None
PROCESSOR = None
MASK_GENERATOR = None
SAMPLE_PRINTED = False


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Build trainable Samples from replayed rows without generating anything."""
    assert not evaluation, "offline GRPO replay is training-only"
    assert args.rollout_global_dataset

    global TOKENIZER, PROCESSOR, MASK_GENERATOR, SAMPLE_PRINTED
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)
    if MASK_GENERATOR is None:
        if args.loss_mask_type != "qwen3_5":
            raise ValueError(
                "Qwen3.5 offline GRPO requires --loss-mask-type qwen3_5"
            )
        MASK_GENERATOR = Qwen35MultiTurnLossMaskGenerator(
            TOKENIZER, tokenizer_type=args.loss_mask_type
        )

    grouped_samples = data_buffer.get_samples(args.rollout_batch_size)
    samples: list = []
    for group in grouped_samples:
        if len(group) != 1:
            raise ValueError(
                "offline replay requires --n-samples-per-prompt 1; "
                f"received group size {len(group)}"
            )
        sample = group[0]
        messages = sample.prompt
        tools = sample.metadata.get("tools") or None
        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages, tools=tools)
        response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]
        if response_length <= 0 or sum(loss_mask) <= 0:
            raise ValueError(
                f"offline GRPO row {sample.index} has no trainable assistant tokens"
            )

        try:
            advantage = float(sample.metadata["advantage"])
            raw_reward = float(sample.metadata["reward"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"offline GRPO row {sample.index} needs numeric metadata.advantage "
                "and metadata.reward"
            ) from exc
        if not math.isfinite(advantage) or not math.isfinite(raw_reward):
            raise ValueError(
                f"offline GRPO row {sample.index} has non-finite reward metadata"
            )

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.loss_mask = loss_mask[-response_length:]
        sample.reward = advantage
        sample.metadata["raw_reward"] = raw_reward
        sample.status = Sample.Status.COMPLETED
        samples.append(sample)

    if samples and not SAMPLE_PRINTED:
        first = samples[0]
        logger.info(
            "sah_offline_grpo_replay example rollout_id=%s index=%s tokens=%s "
            "response_length=%s loss_tokens=%s advantage=%s raw_reward=%s",
            rollout_id,
            first.index,
            len(first.tokens),
            first.response_length,
            sum(first.loss_mask or ()),
            first.reward,
            first.metadata.get("raw_reward"),
        )
        SAMPLE_PRINTED = True
    return samples
