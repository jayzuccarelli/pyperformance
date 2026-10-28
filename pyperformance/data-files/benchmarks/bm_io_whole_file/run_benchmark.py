"""
Benchmark reading and writing a whole file in a single call.

Each result measures a single API. The binary pattern
``open(path, "rb").read()`` / ``open(path, "wb").write(data)`` is measured at
4 KiB, where per-call overhead and the number of system calls dominate, and
at 1 MiB, where buffer allocation and copying dominate.
``Path.read_text`` / ``Path.write_text`` are measured at 1 MiB.
``Path.read_bytes`` / ``Path.write_bytes`` are thin wrappers over the same
``open()`` calls, so they are not measured separately.

The file lives in a fresh temporary directory and stays in the OS page cache,
so this measures Python's I/O stack and system call overhead, not the storage
device. Writes are not fsync'ed and always create a new file: the file is
removed outside the timed region, because truncating and rewriting an
existing file can make the filesystem start writeback on close (ext4
``auto_da_alloc``), which would measure the disk. Larger sizes are left out
on purpose: they scale linearly from 1 MiB and get closer to dirty page
throttling (``vm.dirty_*``), which measures the kernel's writeback policy
rather than Python.

Before the benchmarks run, an 8 MiB block is allocated and freed once. On
glibc this raises malloc's dynamic mmap and trim thresholds well above the
1 MiB buffers used here; otherwise ``read_text_1m``, which frees two 1 MiB
blocks per call, can sit right at the trim threshold and, depending on
earlier heap history, give the memory back to the kernel and fault it in
again on every call (~3x slower).
"""

import io
import os
import pathlib
import shutil
import tempfile

import pyperf


SIZES = {
    "4k": 4 * 1024,
    "1m": 1024 * 1024,
}
TEXT_SIZE = "1m"


def make_bytes(size):
    return (bytes(range(256)) * (size // 256 + 1))[:size]


def make_text(size):
    line = "abcdefghijklmnopqrstuvwxyz0123456789" * 2 + "\n"
    return (line * (size // len(line) + 1))[:size]


def bench_read_bytes(loops, path):
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    for _ in range_it:
        with open(path, "rb") as f:
            f.read()
    return pyperf.perf_counter() - t0


def bench_write_bytes(loops, path, data):
    elapsed = 0
    for _ in range(loops):
        t0 = pyperf.perf_counter()
        with open(path, "wb") as f:
            f.write(data)
        elapsed += pyperf.perf_counter() - t0
        os.unlink(path)
    return elapsed


def bench_read_text(loops, path):
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    for _ in range_it:
        path.read_text(encoding="utf-8")
    return pyperf.perf_counter() - t0


def bench_write_text(loops, path, text):
    elapsed = 0
    for _ in range(loops):
        t0 = pyperf.perf_counter()
        path.write_text(text, encoding="utf-8")
        elapsed += pyperf.perf_counter() - t0
        os.unlink(path)
    return elapsed


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.metadata["description"] = (
        "Read and write a whole file in a single call (page cache warm)"
    )
    runner.metadata["io_default_buffer_size"] = io.DEFAULT_BUFFER_SIZE

    # Pin glibc's dynamic mmap/trim thresholds above the working set, so the
    # results do not depend on heap history (see the module docstring).
    bytes(8 * 1024 * 1024)

    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_path = pathlib.Path(tmp_dir)
        for name, size in SIZES.items():
            data = make_bytes(size)
            path = tmp_path / f"read_{name}.bin"
            path.write_bytes(data)
            runner.bench_time_func(
                f"io_whole_file_read_bytes_{name}", bench_read_bytes, path
            )
            runner.bench_time_func(
                f"io_whole_file_write_bytes_{name}",
                bench_write_bytes,
                tmp_path / f"write_{name}.bin",
                data,
            )
            os.unlink(path)

        text = make_text(SIZES[TEXT_SIZE])
        path = tmp_path / f"read_{TEXT_SIZE}.txt"
        path.write_text(text, encoding="utf-8")
        runner.bench_time_func(
            f"io_whole_file_read_text_{TEXT_SIZE}", bench_read_text, path
        )
        runner.bench_time_func(
            f"io_whole_file_write_text_{TEXT_SIZE}",
            bench_write_text,
            tmp_path / f"write_{TEXT_SIZE}.txt",
            text,
        )
    finally:
        shutil.rmtree(tmp_dir)
