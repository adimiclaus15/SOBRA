import argparse
import pathlib
import time
import math
import random

import numpy as np


def read_target(file_path):
    tokens = []
    with open(file_path, "r") as f:
        for line in f:
            for tok in line.split():
                tokens.append(int(tok))

    if len(tokens) >= 2 and tokens[0] == len(tokens) - 1:
        return tokens[1:]
    return tokens


def best_tau_and_mask(residual):
    N = len(residual)
    idx = np.argsort(residual)[::-1]
    r_sorted = np.array(residual)[idx]
    scores = r_sorted * np.arange(1, N+1)
    k_star = int(np.argmax(scores)) + 1
    tau_star = r_sorted[k_star-1]
    P = [1 if residual[n] >= tau_star else 0 for n in range(N)]
    return tau_star, P


def greedy_fast(hat_d, T_max):
    N = len(hat_d)
    delivered = [0] * N
    tau_list = []
    mask_list = []

    for _ in range(T_max):
        residual = [hat_d[n] - delivered[n] for n in range(N)]
        if max(residual) == 0:
            break
        tau, P = best_tau_and_mask(residual)
        for n in range(N):
            if P[n]:
                delivered[n] += tau
        tau_list.append(tau)
        mask_list.append(P)

    error = sum(abs(hat_d[n] - delivered[n]) for n in range(N))
    return error, tau_list, mask_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch fast greedy (O(T N log N)) for MinFixMasks."
    )
    parser.add_argument(
        "input_dir",
        help="Folder containing target .txt files"
    )
    parser.add_argument(
        "--out_prefix",
        default="greedy2",
        help="Prefix for results, e.g. greedy2_patient_Tk.txt"
    )
    args = parser.parse_args()

    in_base = pathlib.Path(args.input_dir)
    for file_path in sorted(in_base.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() != ".txt":
            continue

        hat_d = read_target(file_path)
        # if len(hat_d) > 250:
        #     random.seed(22)
        #     hat_d = random.sample(hat_d, 250)
        N = len(hat_d)
        Ts = sorted({
            max(1, int(math.log2(N))),
            max(1, int(math.sqrt(N))),
            max(1, N // 4),
            max(1, N // 2),
            max(1, (3 * N) // 4)
        })
        print(f"\n{file_path.name}: N={N}, T_max={Ts}")

        for T_max in Ts:
            start = time.time()
            error, taus, masks = greedy_fast(hat_d, T_max)
            elapsed = time.time() - start
            print(f"  T_max={T_max:<3} error={error:<6} time={elapsed:.6f}s")

            out = f"{args.out_prefix}_{file_path.stem}_T{T_max}.txt"
            with open(out, "w") as f:
                f.write(f"{error} {elapsed:.6f}\n")
            print(f"    → wrote {out}")

