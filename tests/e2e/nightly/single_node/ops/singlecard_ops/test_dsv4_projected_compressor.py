# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile


@pytest.mark.skipif(
    not get_current_hardware_profile().supports(HardwareCapability.DSV4_PROJECTED_COMPRESSOR),
    reason="Requires the packaged projected Compressor kernel",
)
@pytest.mark.parametrize(("ratio", "width", "coff"), [(4, 512, 2), (4, 128, 2), (128, 512, 1)])
@torch.inference_mode()
def test_compressor_from_projected_matches_hidden_chunks(ratio, width, coff):
    """Check projection/restore, both slot formats, RoPE and residual state.

    Mixed requests and successive chunks cross core-group and state-block
    boundaries. This catches projected workspace writes to the wrong group,
    while the native hidden-input path supplies an independent numerical oracle.
    """
    torch.manual_seed(1024)
    weight_kv = (torch.randn(coff * width, 4096, device="npu") * 0.02).bfloat16()
    weight_score = (torch.randn(coff * width, 4096, device="npu") * 0.02).bfloat16()
    ape = torch.randn(ratio, coff * width, device="npu") * 0.02
    norm = 0.5 + torch.rand(width, device="npu")
    state_block_size = 8 if ratio == 4 else 16
    pages_per_request = 1024 // state_block_size
    state_a = torch.zeros((1 + 4 * pages_per_request, state_block_size, 2 * coff * width), device="npu")
    state_b = state_a.clone()
    table = torch.arange(1, 1 + 4 * pages_per_request, dtype=torch.int32, device="npu").view(4, pages_per_request)
    cache_table = torch.arange(1, 9, dtype=torch.int32, device="npu").view(4, 2)
    angles = torch.linspace(0, 1, 1024 * 64, device="npu").view(1024, 64)
    rope_cos, rope_sin = angles.cos(), angles.sin()
    starts = [0, 0, 0, 0]
    for step, lengths in enumerate([[131, 9, 256, 0], [139, 13, 255, 0], [1, 1, 1, 0], [0, 0, 0, 0]]):
        rows = sum(lengths)
        hidden = torch.randn(rows, 4096, device="npu").bfloat16()
        prefix = [0]
        for length in lengths:
            prefix.append(prefix[-1] + length)
        cu = torch.tensor(prefix, dtype=torch.int32, device="npu")
        positions = torch.tensor(starts, dtype=torch.int32, device="npu")
        count = min(rows, rows // ratio + len(lengths))
        expected_slots, rope_positions = [], []
        for rank, (start, length) in enumerate(zip(starts, lengths)):
            for compressed_pos in range(start // ratio, (start + length) // ratio):
                expected_slots.append([1 + rank * 2 + compressed_pos // 128, compressed_pos % 128])
                rope_positions.append(compressed_pos * ratio)
        if count:
            cos, sin, slots = torch.ops._C_ascend.compressor_metadata(
                rope_cos, rope_sin, cu, positions, cache_table, 128, 2, ratio, count, 4
            )
            expected_slots += [[-1, 127]] * (count - len(expected_slots))
            torch.testing.assert_close(slots.cpu(), torch.tensor(expected_slots, dtype=torch.int32), rtol=0, atol=0)
            if rope_positions:
                torch.testing.assert_close(
                    cos.view(count, 64)[: len(rope_positions)], rope_cos[rope_positions], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    sin.view(count, 64)[: len(rope_positions)], rope_sin[rope_positions], rtol=0, atol=0
                )
            _, _, flat_slots = torch.ops._C_ascend.compressor_metadata(
                rope_cos, rope_sin, cu, positions, cache_table, 128, 1, ratio, count, 4
            )
            expected_flat = [block * 128 + offset if block >= 0 else -1 for block, offset in expected_slots]
            torch.testing.assert_close(flat_slots.cpu(), torch.tensor(expected_flat, dtype=torch.int32), rtol=0, atol=0)
            cos, sin = cos.view(count, 64), sin.view(count, 64)
        else:
            cos = hidden.new_empty((0, 64), dtype=torch.float32)
            sin = torch.empty_like(cos)
        kwargs = dict(
            state_block_table=table,
            cu_seqlens=cu,
            seqused=None,
            start_pos=positions,
            rope_head_dim=64,
            cmp_ratio=ratio,
            coff=coff,
            norm_eps=1e-6,
            rotary_mode=2,
            cache_mode=1,
        )
        reference = torch.ops._C_ascend.compressor(
            hidden, weight_kv, weight_score, state_a, ape, norm, sin, cos, **kwargs
        )
        chunks = []
        for local in torch.tensor_split(hidden, 4, dim=0):
            if local.shape[0] == 0:
                chunks.append(hidden.new_empty((0, 2 * coff * width), dtype=torch.float32))
            else:
                kv = torch.mm(local, weight_kv.t(), out_dtype=torch.float32)
                score = torch.mm(local, weight_score.t(), out_dtype=torch.float32)
                chunks.append(torch.cat([kv, score], dim=-1))
        projected = torch.cat(chunks, dim=0)
        # Simulate rank-order exchange plus restoration before the native op.
        swapped = torch.cat([chunks[2], chunks[0], chunks[3], chunks[1]], dim=0)
        order = torch.cat(
            [
                torch.arange(sum(c.shape[0] for c in [chunks[2]]), sum(c.shape[0] for c in [chunks[2], chunks[0]])),
                torch.arange(sum(c.shape[0] for c in [chunks[2], chunks[0], chunks[3]]), rows),
                torch.arange(0, chunks[2].shape[0]),
                torch.arange(
                    sum(c.shape[0] for c in [chunks[2], chunks[0]]),
                    sum(c.shape[0] for c in [chunks[2], chunks[0], chunks[3]]),
                ),
            ]
        ).to("npu")
        restored = swapped.index_select(0, order)
        torch.testing.assert_close(restored, projected, rtol=0, atol=0)
        candidate = torch.ops._C_ascend.compressor_from_projected(
            restored, weight_kv, weight_score, state_b, ape, norm, sin, cos, **kwargs
        )
        torch.npu.synchronize()
        valid = sum((start + length) // ratio - start // ratio for start, length in zip(starts, lengths))
        torch.testing.assert_close(state_a, state_b, rtol=3e-4, atol=3e-4)
        torch.testing.assert_close(reference[:valid].float(), candidate[:valid].float(), rtol=0.02, atol=0.03)
        starts = [start + length for start, length in zip(starts, lengths)]
