"""OpenCL cosine similarity via L2-normalized dot products (one reusable context + program)."""

from __future__ import annotations

from typing import Any

import numpy as np

_KERNEL_SRC = r"""
__kernel void dot_scores(
    __global const float *Q,
    __global const float *D,
    __global float *out,
    const int dim,
    const int n_docs
) {
    const int gid = get_global_id(0);
    if (gid >= n_docs) return;
    float acc = 0.0f;
    const int base = gid * dim;
    for (int j = 0; j < dim; j++) {
        acc += Q[j] * D[base + j];
    }
    out[gid] = acc;
}
"""

_ocl_ctx: Any = None
_ocl_queue: Any = None
_ocl_prg: Any = None


def _create_opencl_context() -> Any:
    """Build a single-device ``Context`` per ``OPENCL_DEVICE`` (cpu / gpu / auto)."""
    from paraverrag.config import OPENCL_DEVICE

    import pyopencl as cl

    pref = OPENCL_DEVICE
    if pref == "auto":
        return cl.create_some_context(interactive=False)
    if pref == "cpu":
        chosen: list[Any] = []
        for plat in cl.get_platforms():
            chosen.extend(plat.get_devices(device_type=cl.device_type.CPU))
        if not chosen:
            raise RuntimeError(
                "PARAVERAG_OPENCL_DEVICE=cpu but no OpenCL CPU device was found (install a CPU ICD, "
                "e.g. Ubuntu: sudo apt install pocl-opencl-icd). If you only have GPU OpenCL, set "
                "PARAVERAG_OPENCL_DEVICE=auto or gpu."
            )
        return cl.Context([chosen[0]])
    if pref == "gpu":
        chosen = []
        for plat in cl.get_platforms():
            chosen.extend(plat.get_devices(device_type=cl.device_type.GPU))
        if not chosen:
            raise RuntimeError(
                "PARAVERAG_OPENCL_DEVICE=gpu but no OpenCL GPU device was found. "
                "Try PARAVERAG_OPENCL_DEVICE=auto or cpu (with pocl-opencl-icd)."
            )
        return cl.Context([chosen[0]])
    raise RuntimeError(f"Invalid OPENCL_DEVICE={pref!r}")


def _opencl_runtime() -> tuple[Any, Any, Any]:
    global _ocl_ctx, _ocl_queue, _ocl_prg
    if _ocl_ctx is None:
        import pyopencl as cl

        ctx = _create_opencl_context()
        queue = cl.CommandQueue(ctx)
        prg = cl.Program(ctx, _KERNEL_SRC).build()
        _ocl_ctx, _ocl_queue, _ocl_prg = ctx, queue, prg
    return _ocl_ctx, _ocl_queue, _ocl_prg


def opencl_probe_error() -> str | None:
    """Return ``None`` if OpenCL is usable; otherwise a short error string (missing ICD, no platforms, etc.)."""
    try:
        import pyopencl  # noqa: F401

        _opencl_runtime()
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def opencl_available() -> bool:
    """True only if PyOpenCL imports and at least one CL platform/context works (ICD/driver present)."""
    return opencl_probe_error() is None


def dot_scores_opencl(q_norm: np.ndarray, d_matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity scores: dot product of ``q_norm`` with each row of ``d_matrix``.

    Caller must supply **L2-unit** ``q_norm`` (dim,) and **row-L2-unit** ``d_matrix`` (n_docs, dim);
    then dot equals cosine similarity in [-1, 1].

    Returns float32 shape (n_docs,).
    """
    import pyopencl as cl

    ctx, queue, prg = _opencl_runtime()

    q = np.asarray(q_norm, dtype=np.float32).reshape(-1)
    d = np.asarray(d_matrix, dtype=np.float32, order="C")
    if d.ndim != 2:
        raise ValueError("d_matrix must be 2-D")
    n_docs, dim = d.shape
    if q.shape[0] != dim:
        raise ValueError("dim mismatch")

    mf = cl.mem_flags
    q_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    d_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
    out_np = np.empty((n_docs,), dtype=np.float32)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, out_np.nbytes)

    prg.dot_scores(
        queue,
        (n_docs,),
        None,
        q_buf,
        d_buf,
        out_buf,
        np.int32(dim),
        np.int32(n_docs),
    )
    cl.enqueue_copy(queue, out_np, out_buf).wait()
    return out_np
