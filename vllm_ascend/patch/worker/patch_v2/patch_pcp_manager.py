# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.v1.worker.gpu.pcp_manager import PCPManager

from vllm_ascend.worker.v2.pcp_manager import validate_ascend_pcp_config

# GPUModelRunner builds its PCP manager before NPUModelRunner can replace it
# with AscendPCPManager. Install the platform-specific validation at worker
# startup so GQA configurations reach that replacement point.
PCPManager.validate_config = staticmethod(validate_ascend_pcp_config)
