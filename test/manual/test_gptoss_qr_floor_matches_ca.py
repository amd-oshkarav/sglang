"""gpt-oss TP4: a tuned QuickReduce floor matches custom-all-reduce-first.

The two chunk sizes are the ones from sgl-project/sglang#34473.

  8192 x 2880 bf16 = 45 MiB. Custom all-reduce still takes this. Untuned
  QuickReduce-first does not: the INT8 cast-on floor is 16 MiB, so the
  chunk is quantized and one value above 65504 spoils its 32-value block.
  16384 x 2880 bf16 = 90 MiB. Custom all-reduce stops at 64 MiB, so both
  orders already use QuickReduce.

ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB=64 puts the 45 MiB chunk back on custom
all-reduce and leaves the 90 MiB chunk on QuickReduce. Passing means the
tuned QuickReduce-first choice equals the custom-all-reduce-first choice
on both chunks, and the 45 MiB result equals the exact sum.

Run on the four GPUs the model uses:

  HIP_VISIBLE_DEVICES=0,1,2,3 python test/manual/test_gptoss_qr_floor_matches_ca.py
"""

import multiprocessing
import os
import socket
import time
import unittest

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    dispatch_custom_allreduce,
)
from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    QuickAllReduce,
    create_quick_allreduce,
    qr_rocm_arch_available,
)

_HIDDEN = 2880
_WORLD_SIZE = 4
_CASES = (
    ("45MiB", (8192, _HIDDEN), "ca"),
    ("90MiB", (16384, _HIDDEN), "qr"),
)


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _choose(order: str, qr_ok: bool, ca_ok: bool) -> str:
    if order == "ca":
        if ca_ok:
            return "ca"
        if qr_ok:
            return "qr"
        return "pynccl"
    if qr_ok:
        return "qr"
    if ca_ok:
        return "ca"
    return "pynccl"


def _run(rank: int, world_size: int, port: int) -> None:
    os.environ["SGLANG_USE_AITER"] = "1"
    os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = "INT8"
    os.environ["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "1"
    os.environ["ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB"] = "64"
    os.environ["ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB"] = "512"

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    qr = None
    ca = None
    try:
        qr = create_quick_allreduce(group=dist.group.WORLD, device=device)
        assert qr is not None and not qr.disabled
        assert isinstance(qr._communicator, QuickAllReduce)
        assert qr._communicator.qr_min_size_override == 64 * 1024 * 1024

        ca_class = dispatch_custom_allreduce(group=dist.group.WORLD, device=device)
        ca = ca_class(group=dist.group.WORLD, device=device)
        assert not ca.disabled, "custom all-reduce did not initialize"

        for name, shape, expected in _CASES:
            # One value above the fp16 ceiling. The same index on every rank
            # makes the reduced sum 280000, which bf16 can hold and fp16 cannot.
            inp = torch.zeros(shape, dtype=torch.bfloat16, device=device)
            inp.view(-1)[0] = 70_000.0
            nbytes = inp.numel() * inp.element_size()
            reference = (inp.float() * world_size).to(torch.bfloat16)

            tuned_qr = qr.should_quick_allreduce(inp)
            ca_ok = bool(ca.should_custom_ar(inp))
            saved = qr._communicator.qr_min_size_override
            qr._communicator.qr_min_size_override = 0
            try:
                untuned_qr = qr.should_quick_allreduce(inp)
            finally:
                qr._communicator.qr_min_size_override = saved

            ca_first = _choose("ca", untuned_qr, ca_ok)
            tuned = _choose("qr", tuned_qr, ca_ok)
            untuned = _choose("qr", untuned_qr, ca_ok)
            if rank == 0:
                print(
                    f"{name} {tuple(shape)} {nbytes / (1024 * 1024):.0f} MiB: "
                    f"ca-first={ca_first} tuned-qr-first={tuned} "
                    f"untuned-qr-first={untuned}",
                    flush=True,
                )
            assert tuned == expected, (
                f"{name}: tuned QuickReduce-first chose {tuned}, expected {expected}"
            )
            assert tuned == ca_first, (
                f"{name}: tuned QuickReduce-first chose {tuned}, "
                f"custom-all-reduce-first chose {ca_first}"
            )

            if name == "45MiB":
                assert untuned == "qr" and tuned == "ca"
                dist.barrier()
                ca_out = ca.custom_all_reduce(inp)
                torch.cuda.synchronize()
                assert ca_out is not None
                assert torch.equal(ca_out, reference), (
                    f"{name}: custom all-reduce disagreed with the exact sum"
                )
                dist.barrier()
                qr_out = qr.quick_all_reduce(inp)
                torch.cuda.synchronize()
                assert not torch.equal(qr_out, reference), (
                    f"{name}: untuned QuickReduce matched the exact sum, "
                    "so this host did not reproduce the cast corruption"
                )
                if rank == 0:
                    print(
                        f"{name}: tuned path matches the exact sum; "
                        f"untuned QuickReduce nonfinite="
                        f"{int((~torch.isfinite(qr_out)).sum().item())}",
                        flush=True,
                    )
    finally:
        if qr is not None:
            qr.close()
        if ca is not None:
            ca.close()
        dist.destroy_process_group()


class TestGptOssQuickReduceFloor(unittest.TestCase):
    def test_tuned_floor_matches_custom_allreduce_first(self):
        if not (qr_rocm_arch_available() and torch.cuda.device_count() >= _WORLD_SIZE):
            self.skipTest("gpt-oss floor test needs four supported ROCm GPUs")
        port = _get_open_port()
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(target=_run, args=(rank, _WORLD_SIZE, port))
            for rank in range(_WORLD_SIZE)
        ]
        for process in processes:
            process.start()
        deadline = time.monotonic() + 180
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        if any(process.is_alive() for process in processes):
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
            self.fail("gpt-oss QuickReduce floor test timed out")
        for rank, process in enumerate(processes):
            self.assertEqual(
                process.exitcode,
                0,
                f"gpt-oss QuickReduce floor test failed on rank {rank}",
            )


if __name__ == "__main__":
    unittest.main()
