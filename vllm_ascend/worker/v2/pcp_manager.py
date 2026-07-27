# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

_UPSTREAM_PCP_VALIDATE_CONFIG = PCPManager.validate_config


def validate_ascend_pcp_config(
    vllm_config: VllmConfig,
    supports_mm_inputs: bool,
) -> None:
    """Validate the PCP subset implemented by the Ascend MRV2 runner."""
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    if parallel_config.prefill_context_parallel_size <= 1:
        return

    # Preserve the upstream MLA validation semantics. The additional path
    # below is intentionally limited to unquantized BF16 GQA in eager mode.
    if model_config.use_mla:
        _UPSTREAM_PCP_VALIDATE_CONFIG(vllm_config, supports_mm_inputs)
        return

    if parallel_config.decode_context_parallel_size > 1:
        raise NotImplementedError("Ascend MRV2 GQA PCP does not support PCP and DCP simultaneously yet.")
    if parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError("Ascend MRV2 GQA PCP does not support PP yet.")
    if model_config.is_encoder_decoder:
        raise NotImplementedError("Ascend MRV2 GQA PCP does not support encoder-decoder models yet.")
    if supports_mm_inputs:
        raise NotImplementedError("Ascend MRV2 GQA PCP does not support MM inputs yet.")
    if vllm_config.lora_config is not None:
        raise NotImplementedError("Ascend MRV2 GQA PCP does not support LoRA yet.")
    if vllm_config.speculative_config is not None:
        raise NotImplementedError("Ascend MRV2 GQA PCP does not support speculative decoding yet.")
    if model_config.quantization is not None:
        raise NotImplementedError("Ascend MRV2 GQA PCP does not support quantized models yet.")
    if model_config.dtype != torch.bfloat16:
        raise NotImplementedError("Ascend MRV2 GQA PCP currently supports BF16 models only.")
    if vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
        raise NotImplementedError(
            "Ascend MRV2 GQA PCP currently supports eager mode only. Set -cc.cudagraph_mode=NONE."
        )

    text_config = model_config.hf_text_config
    num_heads = getattr(text_config, "num_attention_heads", None)
    num_kv_heads = getattr(text_config, "num_key_value_heads", None)
    if (
        not isinstance(num_heads, int)
        or not isinstance(num_kv_heads, int)
        or num_kv_heads <= 0
        or num_heads <= num_kv_heads
        or num_heads % num_kv_heads != 0
    ):
        raise NotImplementedError(
            "Ascend MRV2 GQA PCP requires num_attention_heads to be an "
            "integer multiple greater than num_key_value_heads."
        )


class AscendPCPManager(PCPManager):
    """PCP manager that refreshes Ascend-only local-batch metadata."""

    def __init__(self, *args, vllm_config: VllmConfig, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vllm_config = vllm_config

    def partition_batch(self, input_batch: AscendInputBatch) -> AscendInputBatch:
        """Partition the batch and update Ascend-specific local metadata."""
        local_batch = super().partition_batch(input_batch)
        assert isinstance(local_batch, AscendInputBatch)

        local_seq_lens_np = local_batch.num_computed_tokens_np + local_batch.num_scheduled_tokens
        local_batch.seq_lens_np = local_seq_lens_np
        local_batch.attn_state = build_attn_state(
            self.vllm_config,
            local_seq_lens_np,
            local_batch.num_reqs,
            local_batch.num_scheduled_tokens,
            local_batch.num_scheduled_tokens,
        )
        return local_batch


def maybe_build_ascend_pcp_manager(
    vllm_config: VllmConfig,
    device: torch.device,
    supports_mm_inputs: bool,
    req_states: RequestState,
    block_tables: BlockTables,
) -> AscendPCPManager | None:
    """Build the Ascend PCP manager after validating the supported subset."""
    parallel_config = vllm_config.parallel_config
    pcp_size = parallel_config.prefill_context_parallel_size
    if pcp_size <= 1:
        return None

    validate_ascend_pcp_config(vllm_config, supports_mm_inputs)
    dcp_size = parallel_config.decode_context_parallel_size
    return AscendPCPManager(
        pcp_world_size=pcp_size,
        pcp_rank=get_pcp_group().rank_in_group,
        device=device,
        req_states=req_states,
        max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
        max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        block_tables=block_tables,
        dcp_world_size=dcp_size,
        dcp_rank=get_dcp_group().rank_in_group if dcp_size > 1 else 0,
        cp_interleave=parallel_config.cp_kv_cache_interleave_size,
        vllm_config=vllm_config,
    )
