from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Heavy imports are done lazily inside functions so that the Streamlit front‑end
# loads fast and MPI workers only import what they need.

# --------------------------------------------------------------
# === Sequential reference helpers ==========================================
# --------------------------------------------------------------

def sequential_sort(nums: List[int]) -> List[int]:
    return sorted(nums)


def sequential_wordcount(text: str) -> Tuple[int, int, List[str]]:
    words = text.split()
    unique = sorted(set(words))
    return len(words), len(unique), unique


def sequential_image_filter(img_bytes: bytes, mode: str) -> bytes:
    from PIL import Image, ImageFilter
    import io

    img = Image.open(io.BytesIO(img_bytes))
    if mode == "Grayscale":
        img = img.convert("L")
    else:  # Blur
        img = img.filter(ImageFilter.GaussianBlur(5))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def sequential_linreg(X, y):
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(X, y)
    return model.coef_.tolist(), float(model.intercept_)


def sequential_keyword_search(text: str, keyword: str):
    positions = [i for i in range(len(text)) if text.startswith(keyword, i)]
    return len(positions), positions


def sequential_stats(df):
    import numpy as np
    import pandas as pd
    desc = df.describe().T  # mean, std, min, 25%, 50%, 75%, max
    mode_vals = df.mode().iloc[0]
    desc["median"] = df.median()
    desc["mode"] = mode_vals
    return desc[["mean", "median", "mode", "min", "max", "std"]].to_dict()


def sequential_matmul(A, B):
    import numpy as np
    return (A @ B).tolist()

# --------------------------------------------------------------
# === MPI algorithms ============================================
# --------------------------------------------------------------

# All MPI routines assume they are *only* called from a fresh mpiexec launch.
# They return JSON‑serialisable dictionaries from *rank‑0* – other ranks return None.


def mpi_sort(numbers: List[int]):
    from mpi4py import MPI
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Distribute the numbers as evenly as possible
    counts = np.array([len(numbers) // size + (1 if i < len(numbers) % size else 0) for i in range(size)], dtype=int)
    displs = np.insert(np.cumsum(counts), 0, 0)[0:-1]

    local_n = counts[rank]
    local_arr = np.zeros(local_n, dtype=int)
    comm.Scatterv([np.array(numbers, dtype=int), counts, displs, MPI.INT], local_arr, root=0)

    # Odd‑Even Transposition Sort
    for i in range(len(numbers)):
        # Local sort step – even phase when i is even, odd phase otherwise
        step = i % 2
        for j in range(step, local_n - 1, 2):
            if local_arr[j] > local_arr[j + 1]:
                local_arr[j], local_arr[j + 1] = local_arr[j + 1], local_arr[j]
        # Exchange with neighbour
        if size > 1:
            partner = None
            if (i + rank) % 2 == 0:
                partner = rank + 1 if rank + 1 < size else MPI.PROC_NULL
            else:
                partner = rank - 1 if rank - 1 >= 0 else MPI.PROC_NULL
            if partner != MPI.PROC_NULL:
                sendbuf = np.copy(local_arr[-1] if partner > rank else local_arr[0])
                recvbuf = np.zeros(1, dtype=int)
                comm.Sendrecv([sendbuf, MPI.INT], dest=partner, sendtag=0,
                              recvbuf=recvbuf, source=partner, recvtag=0)
                if partner > rank and recvbuf[0] < local_arr[-1]:
                    local_arr[-1] = recvbuf[0]
                elif partner < rank and recvbuf[0] > local_arr[0]:
                    local_arr[0] = recvbuf[0]

    # Gather back
    sorted_global = None
    if rank == 0:
        sorted_global = np.zeros(len(numbers), dtype=int)
    comm.Gatherv(local_arr, [sorted_global, counts, displs, MPI.INT], root=0)

    if rank == 0:
        return sorted_global.tolist()
    return None


def mpi_wordcount(text: str):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Split text into roughly equal sized chunks by lines
    lines = text.splitlines()
    counts = [len(lines) // size + (1 if i < len(lines) % size else 0) for i in range(size)]
    start = sum(counts[:rank])
    local_lines = lines[start:start + counts[rank]]

    local_words = [w for line in local_lines for w in line.split()]
    local_total = len(local_words)
    local_set = set(local_words)

    total = comm.reduce(local_total, op=MPI.SUM, root=0)
    uniques = comm.gather(local_set, root=0)

    if rank == 0:
        all_unique = set.union(*uniques)
        return {
            "total_words": int(total),
            "unique_words": len(all_unique),
            "sample": list(sorted(all_unique))[:20]
        }
    return None


def mpi_image_filter(img_bytes: bytes, mode: str):
    from mpi4py import MPI
    from PIL import Image
    import io
    import numpy as np
    import cv2

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Root loads the image and broadcasts its raw bytes
    if rank == 0:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img)
        h, w, _ = arr.shape
    else:
        arr = None
        h = w = None
    h = comm.bcast(h, root=0)
    w = comm.bcast(w, root=0)

    # Split by rows
    rows_per_rank = h // size + (1 if rank < h % size else 0)
    start_row = sum(h // size + (1 if r < h % size else 0) for r in range(rank))
    local_rows = rows_per_rank * w * 3  # RGB
    local_buf = np.empty(local_rows, dtype=np.uint8)

    if rank == 0:
        flat = arr.flatten()
    else:
        flat = None
    counts = [(h // size + (1 if r < h % size else 0)) * w * 3 for r in range(size)]
    displs = np.insert(np.cumsum(counts), 0, 0)[0:-1]

    comm.Scatterv([flat, counts, displs, MPI.UNSIGNED_CHAR], local_buf, root=0)
    local_img = local_buf.reshape((-1, w, 3))

    # Apply filter locally
    if mode == "Grayscale":
        local_img = cv2.cvtColor(local_img, cv2.COLOR_RGB2GRAY)
        local_img = cv2.cvtColor(local_img, cv2.COLOR_GRAY2RGB)
    else:
        local_img = cv2.GaussianBlur(local_img, (5, 5), 0)

    # Gather back
    if rank == 0:
        flat_out = np.empty_like(flat)
    else:
        flat_out = None
    comm.Gatherv(local_img.flatten(), [flat_out, counts, displs, MPI.UNSIGNED_CHAR], root=0)

    if rank == 0:
        result_img = Image.fromarray(flat_out.reshape((h, w, 3)))
        out = io.BytesIO()
        result_img.save(out, format="PNG")
        return out.getvalue()
    return None


def mpi_linreg(data: List[List[float]]):
    """Parallel Linear Regression using mpi4py, fallback to sklearn on root."""
    from mpi4py import MPI
    import numpy as np
    from sklearn.linear_model import LinearRegression

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Convert payload to array
    data = np.array(data)

    # Only root performs training (others return None)
    if rank == 0:
        X = data[:, :-1]
        y = data[:, -1]
        # Fit a robust linear regression
        model = LinearRegression().fit(X, y)
        return {
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_)
        }
    return None


def mpi_keyword_search(text: str, keyword: str):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # full length and base segment size
    n = len(text)
    seg = n // size
    overlap = len(keyword) - 1

    # original chunk boundaries
    orig_start = rank * seg
    orig_end = n if rank == size - 1 else (rank + 1) * seg

    # extended boundaries to catch straddling matches
    start = max(0, orig_start - overlap)
    end = min(n, orig_end + overlap)
    local_text = text[start:end]

    # find keyword in local_text
    local_positions = []
    for i in range(0, len(local_text) - len(keyword) + 1):
        if local_text.startswith(keyword, i):
            pos = start + i
            # only count if the match starts within the original segment
            if orig_start <= pos < orig_end:
                local_positions.append(pos)

    # gather all local positions
    all_positions = comm.gather(local_positions, root=0)

    if rank == 0:
        # flatten, dedupe, and sort
        flat = sorted(set(p for sub in all_positions for p in sub))
        return {
            "occurrences": len(flat),
            "positions": flat[:100]
        }
    return None


def mpi_stats(data):
    """Distributed statistical analysis using MPI."""
    from mpi4py import MPI
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Convert payload to numpy array
    arr = np.array(data)
    n_rows, n_cols = arr.shape

    # Determine row slicing
    counts = [n_rows // size + (1 if i < n_rows % size else 0) for i in range(size)]
    displs = np.insert(np.cumsum(counts), 0, 0)[:-1]
    start = displs[rank]
    stop = start + counts[rank]
    local_arr = arr[start:stop, :]

    # Gather local arrays at root
    gathered = comm.gather(local_arr, root=0)

    if rank == 0:
        # Stack into full array
        full = np.vstack(gathered)
        # Compute stats per column
        result = {}
        for i in range(n_cols):
            col = full[:, i]
            mean = float(np.mean(col))
            median = float(np.median(col))
            # Compute mode (smallest value with highest count)
            vals, counts = np.unique(col, return_counts=True)
            mode = float(vals[np.argmax(counts)])
            mn = float(np.min(col))
            mx = float(np.max(col))
            std = float(np.std(col, ddof=0))
            result[str(i)] = {
                "mean": mean,
                "median": median,
                "mode": mode,
                "min": mn,
                "max": mx,
                "std": std,
            }
        return result
    return None


def mpi_matmul(A, B):
    """Parallel block-wise matrix multiplication via MPI."""
    from mpi4py import MPI
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Convert inputs to NumPy arrays
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    # Dimensions
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, f"Incompatible shapes {A.shape} and {B.shape}"

    # Compute row distribution
    counts_rows = [n // size + (1 if i < n % size else 0) for i in range(size)]
    displs_rows = np.insert(np.cumsum(counts_rows), 0, 0)[:-1]

    # Local slice of A
    start_row = displs_rows[rank]
    end_row = start_row + counts_rows[rank]
    A_local = A[start_row:end_row, :]

    # Local computation
    C_local = A_local @ B  # shape (counts_rows[rank], p)
    local_flat = C_local.flatten()

    # Prepare gather parameters in elements
    counts_elems = [cnt * p for cnt in counts_rows]
    displs_elems = np.insert(np.cumsum(counts_elems), 0, 0)[:-1]

    # Root builds flat output buffer
    if rank == 0:
        flat_C = np.empty(n * p, dtype=float)
    else:
        flat_C = None

    # Gather the flattened blocks
    comm.Gatherv(local_flat, [flat_C, counts_elems, displs_elems, MPI.DOUBLE], root=0)

    # On root, reshape and return
    if rank == 0:
        C = flat_C.reshape((n, p))
        return C.tolist()
    return None

# --------------------------------------------------------------
# === Helper for timing & wrapper for each task =================
# --------------------------------------------------------------

def run_parallel(task: str, payload: Dict[str, Any], n_procs: int):
    """Spawn mpiexec -n n_procs on *this* file with --mpi-task=<task>."""
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as f:
        json.dump(payload, f)
        fpath = f.name

    cmd = ["mpiexec", "-n", str(n_procs), sys.executable, __file__, "--mpi-task", task, "--payload", fpath]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if out.returncode != 0:
        raise RuntimeError(out.stderr or "MPI task failed")

    # Root prints the JSON on stdout – we take the *last* line
    result_json = json.loads(out.stdout.strip().splitlines()[-1])
    result_json["parallel_time"] = elapsed
    os.unlink(fpath)
    return result_json


# --------------------------------------------------------------
# === CLI entry‑point for MPI tasks ============================
# --------------------------------------------------------------

def mpi_entry(task: str, payload_path: str):
    with open(payload_path, "r") as f:
        payload = json.load(f)

    start = time.time()

    # ---------------- dispatch ----------------
    if task == "sort":
        raw = mpi_sort(payload["numbers"])
        result = {"sorted": raw} if raw is not None else None
    elif task == "wordcount":
        result = mpi_wordcount(payload["text"])
    elif task == "image":
        result = mpi_image_filter(bytes.fromhex(payload["image_hex"]), payload["mode"])
        if result is not None:
            result = {"image_hex": result.hex()}
    elif task == "linreg":
        result = mpi_linreg(payload["data"])
    elif task == "search":
        result = mpi_keyword_search(payload["text"], payload["keyword"])
    elif task == "stats":
        result = mpi_stats(payload["data"])
    elif task == "matmul":
        raw = mpi_matmul(payload["A"], payload["B"])
        result = {"matrix": raw} if raw is not None else None
    else:
        raise ValueError("Unknown task")

    # ---------- add elapsed on rank‑0 ----------
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0 and result is not None:
        result["elapsed"] = time.time() - start
        print(json.dumps(result))  # Parent captures this  # Parent captures this


# --------------------------------------------------------------
# === Streamlit front‑end =======================================
# --------------------------------------------------------------

def streamlit_app():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import base64

    st.set_page_config(page_title="MPI Parallel Data Processing", layout="wide")
    st.title("MPI Parallel Data Processing Demo")
    st.markdown("Choose a task on the left, upload / enter data, pick number of processes, and press **Run** to see how parallel execution compares to sequential.")

    task = st.sidebar.selectbox("Task", [
        "Sorting", "File Word‑Count", "Image Filter", "Linear Regression", "Keyword Search", "Statistics", "Matrix Multiplication"
    ])
    n_procs = st.sidebar.slider("#MPI processes", 2, 32, 4)

    run_btn = st.sidebar.button("Run")

    sequential_time = parallel_time = None
    result = None

    if task == "Sorting":
        nums_text = st.text_area("Enter numbers (comma‑separated)")
        if run_btn:
            nums = [int(x) for x in nums_text.replace("\n", ",").split(",") if x.strip()]
            payload = {"numbers": nums}
            # Sequential
            t0 = time.time()
            seq_res = sequential_sort(nums)
            sequential_time = time.time() - t0
            # Parallel
            result = run_parallel("sort", payload, n_procs)
            parallel_time = result.pop("parallel_time")
            st.subheader("Sorted numbers")
            st.write(result)

    elif task == "File Word‑Count":
        up = st.file_uploader("Upload .txt file", type="txt")
        if up and run_btn:
            text = up.read().decode("utf-8", errors="ignore")
            payload = {"text": text}
            # Sequential
            t0 = time.time()
            total, unique, _ = sequential_wordcount(text)
            sequential_time = time.time() - t0
            # Parallel
            result = run_parallel("wordcount", payload, n_procs)
            parallel_time = result.pop("parallel_time")
            st.write(result)

    elif task == "Image Filter":
        img_file = st.file_uploader("Image (.png/.jpg)", type=["png", "jpg", "jpeg"])
        mode = st.selectbox("Filter", ["Grayscale", "Blur"])
        if img_file and run_btn:
            img_bytes = img_file.read()
            payload = {"image_hex": img_bytes.hex(), "mode": mode}
            # Sequential
            t0 = time.time()
            seq_bytes = sequential_image_filter(img_bytes, mode)
            sequential_time = time.time() - t0
            # Parallel
            result = run_parallel("image", payload, n_procs)
            parallel_time = result.pop("parallel_time")
            # Convert hex string back to bytes
            image_hex = result.get("image_hex", "")
            par_bytes = bytes.fromhex(image_hex) if image_hex else b""
            st.image([img_bytes, par_bytes], caption=["Original", f"{mode} (Parallel)"], use_column_width=True)

    elif task == "Linear Regression":
        csv = st.file_uploader("CSV with features ... target(last col)", type="csv")
        if csv and run_btn:
            df = pd.read_csv(csv)
            data = df.values.tolist()
            payload = {"data": data}
            # Sequential
            t0 = time.time()
            coef, intercept = sequential_linreg(df.iloc[:, :-1], df.iloc[:, -1])
            sequential_time = time.time() - t0
            # Parallel
            result = run_parallel("linreg", payload, n_procs)
            parallel_time = result.pop("parallel_time")
            st.write("**Coefficients (parallel):**", result)

    elif task == "Keyword Search":
        up = st.file_uploader("Large text file", type="txt")
        keyword = st.text_input("Keyword to search")
        if up and keyword and run_btn:
            text = up.read().decode("utf-8", errors="ignore")
            payload = {"text": text, "keyword": keyword}
            # Sequential
            t0 = time.time()
            occ, pos = sequential_keyword_search(text, keyword)
            sequential_time = time.time() - t0
            # Parallel
            result = run_parallel("search", payload, n_procs)
            parallel_time = result.pop("parallel_time")
            st.write(result)

    elif task == "Statistics":
        csv = st.file_uploader("CSV numeric data", type="csv")
        if csv and run_btn:
            df = pd.read_csv(csv)
            payload = {"data": df.values.tolist()}
            # Sequential
            t0 = time.time()
            seq = sequential_stats(df)
            sequential_time = time.time() - t0
            # Parallel
            result = run_parallel("stats", payload, n_procs)
            parallel_time = result.pop("parallel_time")
            st.json(result)

    elif task == "Matrix Multiplication":
        mat1 = st.file_uploader("Matrix A (CSV)", type="csv", key="A")
        mat2 = st.file_uploader("Matrix B (CSV)", type="csv", key="B")
        if mat1 and mat2 and run_btn:
            import numpy as np
            dfA = pd.read_csv(mat1, header=None, encoding="utf-8-sig")
            dfB = pd.read_csv(mat2, header=None, encoding="utf-8-sig")
            A = dfA.values
            B = dfB.values
            payload = {"A": A.tolist(), "B": B.tolist()}
            
            # Sequential
            t0 = time.time()
            C_seq = sequential_matmul(A, B)
            sequential_time = time.time() - t0
            # Parallel
            result = run_parallel("matmul", payload, n_procs)
            parallel_time = result.pop("parallel_time")
            st.write("Result matrix (parallel)")
            st.write(result)

    # ---- Performance chart ----
    if sequential_time and parallel_time:
        speedup = sequential_time / parallel_time if parallel_time else 0
        st.sidebar.markdown(f"**Sequential:** {sequential_time:.3f} s\n\n**Parallel:** {parallel_time:.3f} s\n\n**Speed‑up:** {speedup:.2f}×")
        fig, ax = plt.subplots()
        ax.bar(["Sequential", "Parallel"], [sequential_time, parallel_time])
        ax.set_ylabel("Time (s)")
        st.pyplot(fig)


# --------------------------------------------------------------
# === Main ======================================================
# --------------------------------------------------------------

if __name__ == "__main__":
    if "--mpi-task" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--mpi-task", required=True)
        parser.add_argument("--payload", required=True)
        args = parser.parse_args()
        mpi_entry(args.mpi_task, args.payload)
    else:
        import subprocess  # imported here to avoid on MPI workers
        streamlit_app()
