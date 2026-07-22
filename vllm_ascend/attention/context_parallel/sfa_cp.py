from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, TypeVar

import torch
import torch.distributed as dist
import torch_npu
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.context_parallel.common_cp import (
    DCPImplMixin,
    DCPMetadataBuilderMixin,
    get_dcp_local_seq_lens,
)
from vllm_ascend.attention.sfa_v1 import (
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.distributed.utils import all_gather_async

M = TypeVar("M", bound=AscendSFAMetadata)


class DCPQueryGatherContext(NamedTuple):
    """State needed to finish the async fused DCP query all-gather."""

    gathered: torch.Tensor
    handle: torch.distributed.Work | None
    restore_perm: tuple[int, ...] | None
    ql_nope_dim: int
    q_pe_dim: int


@dataclass
class DCPContext:
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    query_gather_context: DCPQueryGatherContext | None = None


@dataclass
class AscendSFADCPMetadata(AscendSFAMetadata):
    """SFA metadata fields used only by the DCP execution path."""

    dcp_context: DCPContext | None = None


# SFA DCP replicated-indexer layout:
#
# - LightningIndexer cache is replicated on every DCP rank so index selection
#   can run against the full sequence and keep the same sparse topk semantics as
#   non-DCP SFA.
# - SFA KV cache remains DCP-local to preserve the KV memory saving. The sparse
#   topk indices produced from the replicated indexer view are remapped to local
#   KV indices before calling sparse flash attention.
# - BlockTable only owns the DCP-local physical layout. This builder derives the
#   replicated block table and slot mapping on demand, temporarily builds the
#   indexer-facing metadata with that replicated view, and then stores the
#   original DCP-local view in metadata.dcp_context for KV writes and SFA reads.
# - The replicated view uses the same logical/kernel block size as BlockTable,
#   including hybrid block splitting.
class AscendSFADCPMetadataBuilder(
    DCPMetadataBuilderMixin,
    AscendSFAMetadataBuilder,
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        metadata_cls = metadata_cls or AscendSFADCPMetadata
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls,
            supports_dcp_with_varlen,
        )
        self.cp_kv_cache_interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
        assert self.dcp_size > 1, "AscendSFADCPMetadataBuilder requires DCP world size > 1."
        if self.cp_kv_cache_interleave_size <= 0:
            raise RuntimeError(f"Invalid cp_kv_cache_interleave_size: {self.cp_kv_cache_interleave_size}")

        # Full-graph FIA padding can append one dummy request.
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs + 1
        self.dcp_local_seq_lens_buf = torch.empty(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self.replicated_view_block_size = self.kernel_block_size
        if kv_cache_spec.block_size % self.replicated_view_block_size != 0:
            raise RuntimeError(
                "SFA replicated view requires the KV cache block size "
                f"({kv_cache_spec.block_size}) to be divisible by "
                f"{self.replicated_view_block_size}."
            )
        self.blocks_per_phys_block = kv_cache_spec.block_size // self.replicated_view_block_size
        max_num_input_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_model_len = vllm_config.model_config.max_model_len
        total_cp_size = self.dcp_size
        # Match BlockTable's local logical width, then expand it to the
        # replicated view seen by the SFA indexer.
        max_local_block_table_cols = (
            cdiv(max_model_len, kv_cache_spec.block_size * total_cp_size) * self.blocks_per_phys_block
        )
        max_replicated_block_table_cols = max_local_block_table_cols * total_cp_size
        self.block_table_replicated_view_buf: torch.Tensor = torch.empty(
            (max_num_reqs, max_replicated_block_table_cols),
            dtype=torch.int32,
            device=device,
        )
        self.arange_buffer: torch.Tensor = torch.arange(
            max_replicated_block_table_cols,
            dtype=torch.int32,
            device=device,
        )
        self.slot_mapping_replicated_view_buf: torch.Tensor = torch.empty(
            (max_num_input_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def _get_dcp_local_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        return get_dcp_local_seq_lens(
            seq_lens,
            self.dcp_size,
            self.cp_kv_cache_interleave_size,
        )[:, self.dcp_rank]

    def _ensure_replicated_view_buffers(
        self,
        num_reqs: int,
        num_input_tokens: int,
        local_block_table_cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_table_cols = local_block_table_cols * self.dcp_size
        if (
            self.block_table_replicated_view_buf.shape[0] < num_reqs
            or self.block_table_replicated_view_buf.shape[1] < block_table_cols
        ):
            raise RuntimeError(
                f"Replicated view buffer is too small: "
                f"block_table_replicated_view_buf.shape={self.block_table_replicated_view_buf.shape}, "
                f"num_reqs={num_reqs}, block_table_cols={block_table_cols}"
            )
        if self.slot_mapping_replicated_view_buf.shape[0] < num_input_tokens:
            raise RuntimeError(
                f"Replicated view buffer is too small: "
                f"slot_mapping_replicated_view_buf.shape={self.slot_mapping_replicated_view_buf.shape}, "
                f"num_input_tokens={num_input_tokens}"
            )
        return (
            self.block_table_replicated_view_buf[:num_reqs, :block_table_cols],
            self.arange_buffer[:block_table_cols],
            self.slot_mapping_replicated_view_buf[:num_input_tokens],
        )

    def _build_block_table_replicated_view(
        self,
        dcp_block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = dcp_block_table.shape[0]
        local_block_table_cols = dcp_block_table.shape[1]
        block_table_replicated_view, replicated_col_idx, _ = self._ensure_replicated_view_buffers(
            num_reqs,
            0,
            local_block_table_cols,
        )

        total_cp_size = self.dcp_size
        blocks_per_phys_block = self.blocks_per_phys_block
        local_col_idx = (
            replicated_col_idx // (total_cp_size * blocks_per_phys_block) * blocks_per_phys_block
            + replicated_col_idx % blocks_per_phys_block
        )
        rank_in_replicated_view = (replicated_col_idx // blocks_per_phys_block) % total_cp_size

        local_logical_blocks = torch.index_select(dcp_block_table, 1, local_col_idx)
        if blocks_per_phys_block == 1:
            replicated_blocks = local_logical_blocks * total_cp_size + rank_in_replicated_view
        else:
            local_sub_blocks = local_logical_blocks % blocks_per_phys_block
            local_phys_blocks = local_logical_blocks // blocks_per_phys_block
            replicated_blocks = (
                local_phys_blocks * total_cp_size + rank_in_replicated_view
            ) * blocks_per_phys_block + local_sub_blocks

        valid_req_mask = (seq_lens[:num_reqs].to(device=self.device) > 0).to(replicated_blocks.dtype).view(-1, 1)
        replicated_blocks = replicated_blocks * valid_req_mask
        block_table_replicated_view.copy_(replicated_blocks)
        return block_table_replicated_view

    def _build_slot_mapping_replicated_view(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        block_table_replicated_view: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        num_actual_tokens = min(common_attn_metadata.num_actual_tokens, num_input_tokens)
        _, _, slot_mapping_replicated_view = self._ensure_replicated_view_buffers(
            num_reqs,
            num_input_tokens,
            common_attn_metadata.block_table_tensor.shape[1],
        )
        slot_mapping_replicated_view.fill_(-1)
        if num_actual_tokens == 0:
            return slot_mapping_replicated_view

        query_lens = (
            common_attn_metadata.query_start_loc[1 : num_reqs + 1] - common_attn_metadata.query_start_loc[:num_reqs]
        )
        req_indices = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int32, device=self.device),
            query_lens.to(device=self.device),
            output_size=num_input_tokens,
        )[:num_actual_tokens]
        if req_indices.numel() == 0:
            return slot_mapping_replicated_view

        num_actual_tokens = min(num_actual_tokens, req_indices.shape[0])
        req_indices = req_indices[:num_actual_tokens]
        positions = common_attn_metadata.positions[:num_actual_tokens].to(
            device=self.device,
            dtype=torch.int32,
        )
        logical_block_idx = positions // self.replicated_view_block_size
        block_offsets = positions % self.replicated_view_block_size
        block_table_indices = req_indices * block_table_replicated_view.shape[1] + logical_block_idx
        block_numbers = block_table_replicated_view.flatten()[block_table_indices]
        slot_mapping_replicated_view[:num_actual_tokens] = (
            block_numbers * self.replicated_view_block_size + block_offsets
        )
        return slot_mapping_replicated_view

    def _update_dsa_cp_slot_mapping_for_dcp(
        self,
        metadata: AscendSFAMetadata,
        dcp_slot_mapping: torch.Tensor,
        num_input_tokens: int,
    ) -> None:
        if metadata.dsa_cp_context is None:
            return

        dsa_cp_context = metadata.dsa_cp_context
        slot_mapping = dcp_slot_mapping[:num_input_tokens]
        if dsa_cp_context.num_tokens_pad > slot_mapping.shape[0]:
            slot_mapping = torch.nn.functional.pad(
                slot_mapping,
                (0, dsa_cp_context.num_tokens_pad - slot_mapping.shape[0]),
                value=-1,
            )
        else:
            slot_mapping = slot_mapping[: dsa_cp_context.num_tokens_pad]
        dsa_cp_context.slot_mapping_cp = slot_mapping[dsa_cp_context.local_start : dsa_cp_context.local_end_with_pad]

    def _build_with_metadata_view(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        build_metadata: Callable[[], AscendSFAMetadata],
    ) -> AscendSFAMetadata:
        dcp_slot_mapping = common_attn_metadata.slot_mapping
        dcp_block_table = common_attn_metadata.block_table_tensor
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        block_table_replicated_view = self._build_block_table_replicated_view(
            dcp_block_table[:num_reqs],
            common_attn_metadata.seq_lens,
        )
        slot_mapping_replicated_view = self._build_slot_mapping_replicated_view(
            common_attn_metadata,
            block_table_replicated_view,
        )

        common_attn_metadata.slot_mapping = slot_mapping_replicated_view
        common_attn_metadata.block_table_tensor = block_table_replicated_view
        try:
            metadata = build_metadata()
        finally:
            common_attn_metadata.slot_mapping = dcp_slot_mapping
            common_attn_metadata.block_table_tensor = dcp_block_table

        assert isinstance(metadata, AscendSFADCPMetadata)
        dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens
        if dcp_local_seq_lens is None:
            dcp_local_seq_lens = self._get_dcp_local_seq_lens(metadata.seq_lens)
        local_seq_lens_src = dcp_local_seq_lens[:num_reqs].to(
            device=self.device,
            dtype=torch.int32,
            non_blocking=True,
        )
        self.dcp_local_seq_lens_buf[:num_reqs].copy_(local_seq_lens_src, non_blocking=True)
        local_seq_lens = self.dcp_local_seq_lens_buf[:num_reqs]

        metadata.dcp_context = DCPContext(
            slot_mapping=dcp_slot_mapping[:num_input_tokens],
            block_table=dcp_block_table[:num_reqs],
            seq_lens=local_seq_lens,
        )
        self._update_dsa_cp_slot_mapping_for_dcp(metadata, dcp_slot_mapping, num_input_tokens)
        return metadata


class AscendSFADCPImpl(DCPImplMixin, AscendSFAImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ):
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **kwargs,
        )
        # DCP shards only the SFA KV cache. MLAPO writes the SFA KV cache
        # internally, so keep DCP on the native path where we pass the DCP
        # slot mapping explicitly.
        self.enable_mlapo = False
        self._dcp_interleave_size = self.vllm_config.parallel_config.cp_kv_cache_interleave_size
        if self._dcp_interleave_size <= 0:
            raise RuntimeError(f"Invalid cp_kv_cache_interleave_size: {self._dcp_interleave_size}")
        self._dcp_index_topk = 0
        for config in (
            getattr(self.vllm_config.model_config, "hf_text_config", None),
            getattr(self.vllm_config.model_config, "hf_config", None),
        ):
            index_topk = getattr(config, "index_topk", None)
            if isinstance(index_topk, int) and index_topk > 0:
                self._dcp_index_topk = index_topk
                break
        if self._dcp_index_topk <= 0:
            raise RuntimeError("index_topk must be set in the model config for DCP SFA.")
        device = self.q_proj.weight.device
        self._remap_order = torch.arange(self._dcp_index_topk, dtype=torch.float32, device=device)
        self._remap_invalid_index = torch.tensor(-1.0, dtype=torch.float32, device=device)

    def _all_gather_dim_async(
        self,
        x: torch.Tensor,
        dim: int,
    ) -> tuple[torch.Tensor, torch.distributed.Work | None, tuple[int, ...] | None]:
        assert self.dcp_group is not None
        if dim == 0:
            gathered, handle = all_gather_async(x.contiguous(), self.dcp_group)
            return gathered, handle, None

        perm = (dim, *[i for i in range(x.dim()) if i != dim])
        restore_perm = tuple(perm.index(i) for i in range(x.dim()))
        gathered, handle = all_gather_async(x.permute(perm).contiguous(), self.dcp_group)
        return gathered, handle, restore_perm

    def _remap_sparse_indices(self, topk_indices: torch.Tensor) -> torch.Tensor:
        if self.dcp_size <= 1:
            return topk_indices

        topk_count = topk_indices.shape[-1]
        if topk_count > self._dcp_index_topk:
            raise RuntimeError(
                f"topk_indices last dimension ({topk_count}) exceeds configured index_topk ({self._dcp_index_topk})."
            )

        # Remap the topk indices from the replicated view to the DCP-local KV cache view.
        # We use float32 for better performance on Ascend.
        topk_indices_fp32 = topk_indices.to(torch.float32)
        interleave_size = self._dcp_interleave_size
        local_block_indices = torch.floor(topk_indices_fp32 / interleave_size)
        local_owner_base = torch.floor(local_block_indices / self.dcp_size) * self.dcp_size
        local_owner = local_block_indices - local_owner_base
        local_owner_mask = (topk_indices_fp32 >= 0) & (local_owner == self.dcp_rank)
        if interleave_size == 1:
            remapped_indices_fp32 = torch.floor(topk_indices_fp32 / self.dcp_size)
        else:
            local_offsets = topk_indices_fp32 - local_block_indices * interleave_size
            remapped_indices_fp32 = torch.floor(topk_indices_fp32 / (self.dcp_size * interleave_size))
            remapped_indices_fp32 = remapped_indices_fp32 * interleave_size + local_offsets
        remapped_indices = torch.where(
            local_owner_mask,
            remapped_indices_fp32,
            self._remap_invalid_index,
        ).to(topk_indices.dtype)

        # Compact local indices to the front without changing their top-k order.
        original_order = self._remap_order[:topk_count].expand_as(topk_indices)
        pack_keys = original_order + (~local_owner_mask).to(torch.float32) * topk_count
        _, pack_order = torch.sort(pack_keys, dim=-1)
        return torch.gather(remapped_indices, dim=-1, index=pack_order.to(torch.int32))

    def _merge_dsa_cp_dcp_outputs(
        self,
        sfa_output: torch.Tensor,
        softmax_lse: torch.Tensor,
    ) -> torch.Tensor:
        # DSA-CP keeps the head dimension replicated/full.  Only DCP shards KV,
        # so merge the per-DCP partial outputs explicitly instead of using the
        # common CP helper, which assumes DCP also shards heads.
        num_tokens, num_heads, head_size = sfa_output.shape
        assert self.dcp_group is not None
        out_flat = self.dcp_group.all_gather(sfa_output.contiguous(), dim=0)
        lse_flat = self.dcp_group.all_gather(softmax_lse.contiguous(), dim=0)
        out_flat = out_flat.view(
            self.dcp_size,
            num_tokens,
            num_heads,
            head_size,
        )
        lse_flat = lse_flat.view(
            self.dcp_size,
            num_tokens,
            num_heads,
            1,
        )
        out_flat = out_flat.to(torch.float32).flatten(1, 2)
        lse_flat = lse_flat.to(torch.float32).flatten(1, -1)
        output, _ = torch_npu.npu_attention_update(lse_flat.unbind(0), out_flat.unbind(0), 0)
        return output.view(num_tokens, num_heads, head_size)

    @staticmethod
    def _merge_dcp_outputs_with_torch(
        output_recv: torch.Tensor,
        lse_recv: torch.Tensor,
    ) -> torch.Tensor:
        lse_recv = lse_recv.masked_fill(torch.isnan(lse_recv) | torch.isinf(lse_recv), float("-inf"))
        lse_max = torch.amax(lse_recv, dim=0)
        lse_max = lse_max.masked_fill(lse_max == float("-inf"), 0.0)
        weights = torch.exp(lse_recv - lse_max.unsqueeze(0))
        weights = weights.masked_fill(torch.isnan(weights), 0.0)
        weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-10)

        output = (output_recv.to(torch.float32) * weights.unsqueeze(-1)).sum(dim=0)
        return output.permute(1, 0, 2).contiguous()

    def _merge_dcp_outputs(
        self,
        sfa_output: torch.Tensor,
        softmax_lse: torch.Tensor,
    ) -> torch.Tensor:
        assert self.dcp_group is not None, "DCP output merge requires dcp_group when dcp_size > 1."
        num_tokens, num_heads, head_size = sfa_output.shape
        assert num_heads % self.dcp_size == 0
        local_num_heads = num_heads // self.dcp_size

        output_send = sfa_output.permute(1, 0, 2).contiguous()
        output_recv = torch.empty_like(output_send)
        dist.all_to_all_single(
            output_recv,
            output_send,
            group=self.dcp_group.device_group,
        )
        output_recv = output_recv.view(self.dcp_size, local_num_heads, num_tokens, head_size)

        lse_send = softmax_lse.to(torch.float32).permute(1, 0, 2).contiguous()
        lse_recv = torch.empty_like(lse_send)
        dist.all_to_all_single(
            lse_recv,
            lse_send,
            group=self.dcp_group.device_group,
        )
        lse_recv = lse_recv.view(self.dcp_size, local_num_heads, num_tokens)

        return self._merge_dcp_outputs_with_torch(output_recv, lse_recv)

    def _start_dcp_query_gather(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> DCPQueryGatherContext:
        query_gather_dim = 0 if self.enable_dsa_cp else 1
        assert self.dcp_group is not None, "DCP query gather requires dcp_group when dcp_size > 1."
        if ql_nope.shape[:-1] != q_pe.shape[:-1] or ql_nope.dtype != q_pe.dtype:
            raise RuntimeError(
                "Cannot fuse DCP query gather for ql_nope/q_pe with "
                f"shapes {tuple(ql_nope.shape)} / {tuple(q_pe.shape)} "
                f"and dtypes {ql_nope.dtype} / {q_pe.dtype}."
            )

        ql_nope_dim = ql_nope.shape[-1]
        q_pe_dim = q_pe.shape[-1]
        # Avoid back-to-back DCP all_gather calls for the two SFA query
        # fragments. On Ascend the separate gathers can leave SFA with an
        # incomplete stream dependency on the first prefill. DSA-CP restores
        # token shards on dim 0; native DCP restores query shards on dim 1.
        fused_q = torch.cat([ql_nope, q_pe], dim=-1).contiguous()
        gathered, handle, restore_perm = self._all_gather_dim_async(fused_q, query_gather_dim)
        return DCPQueryGatherContext(gathered, handle, restore_perm, ql_nope_dim, q_pe_dim)

    def _record_query_gather_context(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        attn_metadata: M,
    ) -> None:
        assert isinstance(attn_metadata, AscendSFADCPMetadata)
        assert attn_metadata.dcp_context is not None, "DCP SFA requires attn_metadata.dcp_context."
        attn_metadata.dcp_context.query_gather_context = self._start_dcp_query_gather(ql_nope, q_pe)

    def _get_sfa_kv_slot_mapping(
        self,
        attn_metadata: M,
    ) -> torch.Tensor:
        assert isinstance(attn_metadata, AscendSFADCPMetadata)
        assert attn_metadata.dcp_context is not None
        return attn_metadata.dcp_context.slot_mapping

    def _finish_all_gather_query_for_dcp(
        self,
        context: DCPQueryGatherContext,
    ) -> tuple[torch.Tensor, ...]:
        if context.handle is not None:
            context.handle.wait()
        gathered = context.gathered
        if context.restore_perm is not None:
            gathered = gathered.permute(context.restore_perm).contiguous()
        return torch.split(gathered, [context.ql_nope_dim, context.q_pe_dim], dim=-1)

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
    ):
        assert attn_metadata.dcp_context is not None, "DCP SFA requires attn_metadata.dcp_context."
        assert self.dcp_group is not None, "DCP SFA requires dcp_group when dcp_size > 1."
        dcp_context = attn_metadata.dcp_context
        query_gather_context = dcp_context.query_gather_context
        dcp_context.query_gather_context = None
        if query_gather_context is None:
            query_gather_context = self._start_dcp_query_gather(ql_nope, q_pe)
        if self.enable_dsa_cp:
            # DSA-CP shards the token sequence. Restore the flat token order for
            # SFA, and use the original full query lengths for varlen metadata.
            actual_seq_lengths_query = attn_metadata.cum_query_lens
            # topk_indices are in per-request global token coordinates. Gather
            # the DSA token shards first, then remap for this receiver rank's
            # DCP-local KV shard.
            topk_indices = self.dcp_group.all_gather(topk_indices.contiguous(), dim=0)
        topk_indices = self._remap_sparse_indices(topk_indices)
        ql_nope, q_pe = self._finish_all_gather_query_for_dcp(query_gather_context)
        kv = kv_cache[0]
        key_rope = kv_cache[1]
        sfa_output, softmax_max, softmax_sum = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=ql_nope,
            key=kv,
            value=kv,
            sparse_indices=topk_indices,
            scale_value=self.scale,
            sparse_block_size=1,
            block_table=attn_metadata.dcp_context.block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=attn_metadata.dcp_context.seq_lens,
            query_rope=q_pe,
            key_rope=key_rope,
            layout_query="TND",
            layout_kv="PA_BSND",
            # The replicated-view indexer already applies the causal visibility rule.
            # After DCP remaps topk indices to local KV positions, local KV
            # length no longer shares the same coordinate system as global
            # query length, so SFA must not apply its right-down causal crop.
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=True,
        )
        output_dtype = sfa_output.dtype
        # SFA returns softmax max/sum separately. Convert them to LSE so the
        # existing CP merge helper can combine local-KV partial outputs.
        softmax_lse = softmax_max.to(torch.float32) + torch.log(softmax_sum.to(torch.float32))
        softmax_lse = softmax_lse.permute(1, 0, 2).reshape(softmax_lse.shape[1], -1, 1)
        if self.enable_dsa_cp:
            output = self._merge_dsa_cp_dcp_outputs(sfa_output, softmax_lse)
            assert attn_metadata.dsa_cp_context is not None, "DSA-CP DCP output selection requires dsa_cp_context."
            dsa_cp_context = attn_metadata.dsa_cp_context
            output = output[dsa_cp_context.local_start : dsa_cp_context.local_end_with_pad]
            return output.to(output_dtype)
        output = self._merge_dcp_outputs(sfa_output, softmax_lse)
        return output.to(output_dtype)
