import argparse
import pathlib
import math
import random

import gurobipy as gp
from gurobipy import GRB
import numpy as np


def read_target(file_path):
    with open(file_path, "r") as f:
        tokens = []
        for line in f:
            for tok in line.split():
                tokens.append(int(tok))
    
    if len(tokens) >= 2 and tokens[0] == len(tokens) - 1:
        return tokens[1:]
    return tokens


def solve_min_fix_masks_indicators(hat_d, T_max, time_limit=None, mip_gap=None):
    N = len(hat_d)
    model = gp.Model(f"MinFixMasks_T{T_max}")
    model.setParam("OutputFlag", 0)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if mip_gap is not None:
        model.setParam("MIPGap", mip_gap)

    hat_max = max(hat_d)

    P = {}
    tau = {}
    dnt = {}
    delta = {}

    for n in range(N):
        for t in range(T_max):
            P[n, t] = model.addVar(vtype=GRB.BINARY, name=f"P_{n}_{t}")
    for t in range(T_max):
        tau[t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=hat_max, name=f"tau_{t}")
    for n in range(N):
        for t in range(T_max):
            dnt[n, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"dnt_{n}_{t}")
    for n in range(N):
        delta[n] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"delta_{n}")

    model.update()

    
    for n in range(N):
        for t in range(T_max):
            model.addGenConstrIndicator(P[n, t], 1, dnt[n, t] - tau[t], GRB.EQUAL, 0,
                                        name=f"ind_pos_{n}_{t}")
            model.addGenConstrIndicator(P[n, t], 0, dnt[n, t], GRB.EQUAL, 0,
                                        name=f"ind_zero_{n}_{t}")

    
    for n in range(N):
        expr = gp.LinExpr()
        for t in range(T_max):
            expr.add(dnt[n, t])
        model.addConstr(delta[n] >= hat_d[n] - expr, name=f"abspos_{n}")
        model.addConstr(delta[n] >= expr - hat_d[n], name=f"absneg_{n}")

    
    model.setObjective(gp.quicksum(delta[n] for n in range(N)), GRB.MINIMIZE)

    
    model.optimize()

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        obj_val = model.getAttr(GRB.Attr.ObjVal)
        
        P_sol = {
            (n, t): int(P[n, t].getAttr(GRB.Attr.X) + 0.5)
            for n in range(N) for t in range(T_max)
        }
        tau_sol = {
            t: int(tau[t].getAttr(GRB.Attr.X) + 0.5)
            for t in range(T_max)
        }
        return model, obj_val, P_sol, tau_sol
    else:
        return model, None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply MinFixMasks_BOUND to all files in a folder. "
                    "For each file, uses T_max ∈ {n/4, n/2, 3n/4}. "
                    "Outputs only error and runtime.")
    parser.add_argument(
        "input_dir",
        help="Folder containing target files (one file = one list of integers)."
    )
    parser.add_argument(
        "--out_prefix",
        default="solution",
        help="Prefix for output files (e.g. solution_filename_T{T}.txt)."
    )
    parser.add_argument(
        "--timelimit",
        type=float,
        default=None,
        help="Gurobi TimeLimit (seconds)."
    )
    parser.add_argument(
        "--mipgap",
        type=float,
        default=None,
        help="Gurobi MIPGap (relative)."
    )

    args = parser.parse_args()
    in_base = pathlib.Path(args.input_dir)
    if not in_base.is_dir():
        raise FileNotFoundError(f"Not a folder: {in_base}")

    
    for file_path in sorted(in_base.iterdir()):
        if not file_path.is_file():
            continue
        hat_d = read_target(file_path)
        sz = len(hat_d)
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
        print(f"\nFile {file_path.name}: original size={sz}, used N={N}, T_max values = {Ts}")

        for T_max in Ts:
            print(f"  Solving T_max = {T_max}...", end="", flush=True)
            model, obj_val, P_sol, tau_sol = solve_min_fix_masks_indicators(
                hat_d, T_max,
                time_limit=args.timelimit,
                mip_gap=args.mipgap
            )
            if obj_val is None:
                print(" no solution.")
                continue
            runtime = model.getAttr(GRB.Attr.Runtime)
            print(f" done. Error = {obj_val:.6f}, Runtime = {runtime:.6f}s")

            out_name = f"{args.out_prefix}_{file_path.stem}_T{T_max}.txt"
            with open(out_name, "w") as f:
                f.write(f"{obj_val:.6f} {runtime:.6f}\n")
            print(f"    Wrote {out_name}")
