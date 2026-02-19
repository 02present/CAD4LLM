#!/usr/bin/env python3
import argparse, h5py, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5a")
    ap.add_argument("h5b")
    ap.add_argument("--dataset", default="vec")
    args = ap.parse_args()

    with h5py.File(args.h5a, "r") as f:
        A = np.array(f[args.dataset])
    with h5py.File(args.h5b, "r") as f:
        B = np.array(f[args.dataset])

    print("shape A", A.shape, "shape B", B.shape)
    same = (A.shape == B.shape) and np.array_equal(A, B)
    print("[SAME_VEC]" if same else "[DIFF_VEC]")
    if not same:
        if A.shape == B.shape:
            idx = np.argwhere(A != B)
            print("diff count:", idx.shape[0])
            print("first 10 diffs:", idx[:10].tolist())
            if idx.shape[0] > 0:
                i, j = idx[0]
                print("A[i,j], B[i,j] =", A[i, j], B[i, j])

if __name__ == "__main__":
    main()
