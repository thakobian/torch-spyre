# Copyright 2026 The Torch-Spyre Authors.
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

import os

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_spyre._C import (  # type: ignore[attr-defined]
    PoolConfig,
    PoolStatus,
    SpyreTensorLayout,
    attach_pool,
    create_pool,
    destroy_pool,
    detach_pool,
    open_pool,
    copy_kv_page_raw,
    get_composite_address,
    get_device_dtype,
    get_elem_in_stick,
)


# Skip all tests if RANK is not defined, or WORLD_SIZE is not set or less than 2
if "RANK" not in os.environ:
    pytest.skip(
        "RANK environment variable not defined, skipping distributed tests",
        allow_module_level=True,
    )

if "WORLD_SIZE" not in os.environ:
    pytest.skip(
        "WORLD_SIZE environment variable not defined, skipping distributed tests",
        allow_module_level=True,
    )

try:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size < 2:
        pytest.skip(
            f"WORLD_SIZE is {world_size}, need at least 2 for distributed tests",
            allow_module_level=True,
        )
except ValueError:
    pytest.skip(
        "WORLD_SIZE environment variable is not a valid integer, skipping distributed tests",
        allow_module_level=True,
    )

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"

# Geometry from ibm-ai-platform/micro-g3.3-8b-instruct-1b config.json.
# head_dim is hidden_size 4096 / num_attention_heads 32.
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_BLOCKS = 4
BLOCK_SIZE = 16
BLOCK_ID = 1


def slot_major_layout(num_slots, num_kv_heads, head_size, dtype=torch.float16):
    """Mirror of spyre-inference slot_major_kv_layout, which is how the real
    decoder cache is allocated. The slot axis is outermost, so each page is one
    contiguous physical range. stride_map is row major over device_size, which
    is what copy_kv_page_raw requires.
    """
    eps = get_elem_in_stick(dtype)
    sticks = (head_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_slots, num_kv_heads, sticks, eps],
        stride_map=[num_kv_heads * sticks * eps, sticks * eps, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


def make_cache(kv_cache_shape, fill_data):
    """Allocate a KV cache with the production layout. Host allocated then
    transferred, since only .to() accepts a device_layout.
    """
    num_blocks, block_size, num_kv_heads, head_dim = kv_cache_shape
    layout = slot_major_layout(num_blocks * block_size, num_kv_heads, head_dim)
    host_cache = torch.zeros(kv_cache_shape, dtype=torch.float16)
    if fill_data:
        # Each page holds its own index so both ranks build the same data.
        for block in range(num_blocks):
            host_cache[block] = block + 1
    return host_cache.to(DEVICE, device_layout=layout)


class TestKVOffloadCrossProcess(TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the distributed environment once for all tests."""
        if not dist.distributed_c10d.is_backend_available(C10D_BACKEND):
            raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
        if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
            raise RuntimeError(
                f"Error: Missing a C10 Backend for 'spyre'! Expected {C10D_BACKEND}"
            )

        if not dist.is_initialized():
            dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")

        cls.comm_size = dist.get_world_size()
        cls.comm_rank = dist.get_rank()

    @classmethod
    def tearDownClass(cls):
        """Clean up the distributed environment after all tests."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_cross_process_reload(self):
        """
        Test that a KV page offloaded in one process can be reloaded in another.
        """
        kv_cache_shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
        cache = make_cache(kv_cache_shape, fill_data=True)
        name = self.id()

        if self.comm_rank == 0:
            # Process 0: create the pool as the delegator would, offload one
            # page into slot 0, then detach. The pool keeps the page.
            page_bytes = get_composite_address(cache).total_size // NUM_BLOCKS
            config = PoolConfig(name, num_slots=1, slot_bytes=page_bytes)
            status, descriptor = create_pool(config)
            self.assertEqual(status, PoolStatus.Ok)
            status, pool = attach_pool(descriptor)
            self.assertEqual(status, PoolStatus.Ok)
            copy_kv_page_raw(cache, BLOCK_ID, pool, 0, to_device=False)
            detach_pool(pool)

        dist.barrier()  # Ensure process 0 has completed offloading

        # Process 1: open the pool by name and reload the page from it
        if self.comm_rank == 1:
            status, descriptor = open_pool(name)
            self.assertEqual(status, PoolStatus.Ok)
            status, pool = attach_pool(descriptor)
            self.assertEqual(status, PoolStatus.Ok)
            reloaded_cache = make_cache(kv_cache_shape, fill_data=False)
            copy_kv_page_raw(reloaded_cache, BLOCK_ID, pool, 0, to_device=True)
            detach_pool(pool)

            # Verify the reloaded page matches byte for byte, with zero tolerance
            self.assertEqual(
                reloaded_cache.to("cpu")[BLOCK_ID].contiguous().view(torch.uint8),
                cache.to("cpu")[BLOCK_ID].contiguous().view(torch.uint8),
                rtol=0,
                atol=0,
            )

        dist.barrier()  # Ensure process 1 has detached

        if self.comm_rank == 0:
            # Nobody is attached anymore, so the pool can be destroyed.
            self.assertEqual(destroy_pool(descriptor), PoolStatus.Ok)


if __name__ == "__main__":
    run_tests()
