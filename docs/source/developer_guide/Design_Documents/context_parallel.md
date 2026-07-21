# Decode Context Parallel (DCP)

Decode Context Parallel shards the KV cache along the sequence dimension across devices in a Tensor Parallel (TP) group. It eliminates redundant KV-cache storage without adding devices to the process world.

Prefill Context Parallel is not supported by vLLM Ascend. This document describes only DCP and the separate DSA-CP sparse-attention path.

## KV-cache layout

DCP stores tokens in an interleaved layout across ranks. The interleaving granularity is controlled by `cp_kv_cache_interleave_size`, whose default value is `1`.

For a DCP size of `dcp_size`, a virtual block contains `block_size * dcp_size` tokens. For token `x`:

- `virtual_block_index = x // (block_size * dcp_size)`
- `offset_in_virtual_block = x % (block_size * dcp_size)`
- `local_block_index = offset_in_virtual_block // cp_kv_cache_interleave_size`
- `target_rank = local_block_index % dcp_size`

The slot-mapping calculation uses this layout so each DCP rank stores only its local sequence shard. The current implementation requires `block_size % cp_kv_cache_interleave_size == 0`.

![DCP block table](../../assets/cp/blocktable.png)

## Attention execution

### Backend structure

DCP is implemented as a specialization of the corresponding v1 attention
backend rather than as a parallel copy of it:

- `DCPMetadataBuilderMixin` owns DCP group/rank discovery and access to the
  per-rank context-length matrix.
- `DCPImplMixin` owns DCP collectives and the common partial-output/LSE merge.
- GQA, MLA, and SFA DCP builders inherit their v1 metadata builders. They only
  add DCP metadata fields or temporarily expose the DCP-specific cache view.
- GQA, MLA, and SFA DCP implementations inherit their v1 implementations and
  override only the kernel stages whose communication or cache layout differs.

The normal v1 builders remain the source of truth for request classification,
padding, masks, graph metadata, and common KV-cache metadata. DCP-specific
metadata is kept out of the normal v1 metadata schemas.

### Prefill and chunked prefill

During chunked or cached prefill, the local query must attend to KV-cache shards distributed across the DCP group.

- MLA gathers the context KV cache, restores request-contiguous order, and computes attention for the current query chunk.
- GQA gathers query heads across the DCP group, computes attention against each local KV shard, and combines partial outputs and LSE values.

![DCP prefill](../../assets/cp/dcp-prefill.png)

### Decode

Decode gathers the query heads required by each DCP rank, computes attention against the local KV-cache shard, and combines partial outputs and LSE values across the DCP group.

![DCP decode](../../assets/cp/dcp-decode.png)

## SFA DSA-CP mixed `o_proj` path

SFA DSA-CP mixed execution reuses the normal TP-sharded `o_proj`. DSA-CP is controlled independently through `additional_config.enable_dsa_cp`; it is not Prefill Context Parallel.

- Decode-only batches keep the decode TP path. SFA outputs are exchanged with an all-to-all in the TP group, then the TP-sharded `o_proj` runs normally.
- Prefill or mixed batches all-gather the TP-sharded `o_proj` weight and input-sharded quantization parameters into reusable temporary full-weight buffers before the projection.

The original TP-sharded parameter remains the only persistent source of truth. Temporary full-weight buffers must not become a second persistent parameter copy.

## Related files

- Slot mapping: `vllm_ascend/worker/block_table.py`
- Input and attention metadata: `vllm_ascend/worker/model_runner_v1.py`
- Shared DCP backend capabilities: `vllm_ascend/attention/context_parallel/common_cp.py`
- GQA DCP backend: `vllm_ascend/attention/context_parallel/attention_cp.py`
- MLA DCP backend: `vllm_ascend/attention/context_parallel/mla_cp.py`
- SFA DCP backend: `vllm_ascend/attention/context_parallel/sfa_cp.py`
- DSA-CP backend: `vllm_ascend/attention/context_parallel/dsa_cp.py`
