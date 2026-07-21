#!/usr/bin/env python
"""
Finite-size scaling of the absolute crystal free energy A_sol/NkT.

The per-particle solid free energy has leading ln(N)/N and 1/N size effects associated with the
fixed-center-of-mass construction and intrinsic crystal phonons [Polson et al., JCP 112, 5339
(2000)]. ``run_ti.py`` also reports the historical Frenkel-Ladd proxy
``A_sol_FL_NkT = A_sol_NkT + (2/N) ln N``. Vega et al. emphasize that this is not the exact free
energy of the finite system; it is an empirical finite-size prescription that is often closer to
the thermodynamic limit. This tool therefore reports both a 1/N fit of that proxy and, when at least
three sizes are available, a raw-data cross-check with fitted ln(N)/N and 1/N coefficients.

Why this matters here: CsCl and Th3P4 are compared at DIFFERENT N (2 vs 7 atoms/cell). If their
finite-size corrections differ, the fixed-N comparison is biased; extrapolating each to N->infinity
gives a fair selection and quantifies the residual bias.

It reports:
  * a table of a_sol_raw, the historical a_sol_FL proxy, and the decomposition A0/N, dA1/N, dA2/N vs N,
    which localizes the size dependence: in practice dA1 (essentially the lattice energy) is
    intensive/N-independent, and the size dependence lives in A0 (the ideal-Einstein reference, mostly
    the center-of-mass ln N/N term) and in dA2 (the interacting-solid vibrational/anharmonic part);
  * a weighted fit a_sol_FL(N) = a_inf + b/N (and a cross-check fitting the RAW value with an explicit
    +c*ln N/N term), giving a_inf +/- uncertainty;
  * the finite-size bias a_sol_FL(N) - a_inf at each N (how wrong each fixed-N number is);
  * a figure fss_<label>.png.

Usage:
    python finite_size_scaling.py <ti_dir_N1> <ti_dir_N2> ... [--label CsCl] [--no-errors]
"""

import argparse
import json
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def _load_run(output_dir):
    r = json.load(open(os.path.join(output_dir, "free_energy.json")))
    return {"dir": output_dir, "N": int(r["n_particles"]),
            "a_raw": float(r["A_sol_NkT"]), "a_fl": float(r["A_sol_FL_NkT"]),
            "A0": float(r["A0_NkT"]), "dA1": float(r["dA1_NkT"]), "dA2": float(r["dA2_NkT"]),
            "json": r}


def _dA2_error(output_dir, r):
    """Statistical error of dA2/NkT: sum_k (c_k se_k)^2, c_k = GL weight_k*ds/dw_k/(N kT),
    se_k = block standard error of window k's mean bare spring energy. Returns nan if windows absent."""
    try:
        from free_energy import FrenkelLaddSchedule
        from colloids.units import energy_unit, length_unit, temperature_unit
        import mbar_analysis
        _KB = 0.0083144626
        n = r["n_particles"]; kt = _KB * r["temperature_K"]
        rows = sorted(r["windows_data"], key=lambda x: x["lambda_ein"])
        bare = [mbar_analysis.read_bare_energies(
            os.path.join(output_dir, "windows", f"window_{int(x['window']):02d}", "trajectory.gsd"),
            x["lambda_ein"]) for x in rows]
        sched = FrenkelLaddSchedule(n_points=len(rows), temperature=r["temperature_K"] * temperature_unit,
                                    spring_constant=r["lambda_E_kJ_per_mol_nm2"] * energy_unit / (length_unit ** 2))
        coeff = sched._gl_weights * sched._ds_dw / (n * kt)
        se = np.array([_block_se(b) for b in bare])
        return float(np.sqrt(np.sum((coeff * se) ** 2)))
    except Exception:
        return float("nan")


def _dA1_error(output_dir, n_particles, n_boot=200):
    """Bootstrap error of dA1/NkT from the saved ideal-Einstein reweighting samples. nan if absent."""
    try:
        d = np.load(os.path.join(output_dir, "delta_a1_samples.npz"))
        _KB = 0.0083144626
        beta = 1.0 / (_KB * float(d["temperature_K"]))
        n = int(d["n_mobile"]); du = np.asarray(d["u_sol_samples"]) - float(d["u_lattice"])
        w = -beta * du
        rng = np.random.default_rng(0); m = w.size
        vals = []
        for _ in range(n_boot):
            s = w[rng.integers(0, m, m)]
            mx = s.max()
            vals.append(-(mx + np.log(np.mean(np.exp(s - mx)))) / n)   # -(1/N) ln <exp(w)>
        return float(np.std(vals))
    except Exception:
        return float("nan")


def _block_se(x, n_blocks=5):
    x = np.asarray(x); m = (x.size // n_blocks) * n_blocks
    if m < n_blocks:
        return float(x.std() / max(1.0, np.sqrt(x.size)))
    means = x[:m].reshape(n_blocks, -1).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(n_blocks))


def _weighted_line_fit(x, y, yerr):
    """Weighted least squares y = a + b x. Returns (a, a_err, b, b_err)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = 1.0 / np.asarray(yerr, float) ** 2 if np.all(np.isfinite(yerr)) and np.all(np.asarray(yerr) > 0) \
        else np.ones_like(x)
    S = w.sum(); Sx = (w * x).sum(); Sy = (w * y).sum(); Sxx = (w * x * x).sum(); Sxy = (w * x * y).sum()
    denom = S * Sxx - Sx * Sx
    b = (S * Sxy - Sx * Sy) / denom
    a = (Sy - b * Sx) / S
    a_err = np.sqrt(Sxx / denom); b_err = np.sqrt(S / denom)
    return a, a_err, b, b_err


def _fit_raw_logN(Ns, a_raw, aerr):
    """Fit raw a_sol(N) = a_inf + c*(lnN/N) + b*(1/N); return a_inf, a_inf_err (design-matrix WLS)."""
    Ns = np.asarray(Ns, float)
    A = np.column_stack([np.ones_like(Ns), np.log(Ns) / Ns, 1.0 / Ns])
    W = np.diag(1.0 / np.asarray(aerr, float) ** 2) if np.all(np.isfinite(aerr)) else np.eye(len(Ns))
    cov = np.linalg.inv(A.T @ W @ A)
    coef = cov @ (A.T @ W @ np.asarray(a_raw, float))
    return float(coef[0]), float(np.sqrt(cov[0, 0]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="run_ti output dirs at different N (same conditions)")
    p.add_argument("--label", default="crystal")
    p.add_argument("--no-errors", action="store_true", help="skip the per-size error estimate (faster; unweighted fit)")
    p.add_argument("--plot", default=None, help="output png path (default ./fss_<label>.png)")
    args = p.parse_args()

    runs = sorted((_load_run(d) for d in args.dirs), key=lambda r: r["N"])
    if len(runs) < 2:
        p.error("finite-size scaling requires at least two run directories")
    if len({r["N"] for r in runs}) < len(runs):
        print("WARNING: repeated N values among the inputs.")
    Ns = np.array([r["N"] for r in runs], float)

    a_err = np.full(len(runs), np.nan)
    if not args.no_errors:
        for i, r in enumerate(runs):
            e2 = _dA2_error(r["dir"], r["json"]); e1 = _dA1_error(r["dir"], r["N"])
            a_err[i] = np.sqrt(np.nansum([e2 ** 2, e1 ** 2])) if np.isfinite(e2) or np.isfinite(e1) else np.nan

    print(f"Finite-size scaling: {args.label}   ({len(runs)} sizes)")
    print(f"  {'N':>6} {'a_raw':>10} {'a_FL proxy':>11} {'+/-':>8} {'A0/N':>9} {'dA1/N':>10} {'dA2/N':>9}")
    for r, e in zip(runs, a_err):
        es = f"{e:8.4f}" if np.isfinite(e) else f"{'--':>8}"
        print(f"  {r['N']:>6} {r['a_raw']:>10.4f} {r['a_fl']:>11.4f} {es} "
              f"{r['A0']:>9.4f} {r['dA1']:>10.4f} {r['dA2']:>9.4f}")

    a_fl = np.array([r["a_fl"] for r in runs])
    a_raw = np.array([r["a_raw"] for r in runs])
    a_inf, a_inf_err, b, b_err = _weighted_line_fit(1.0 / Ns, a_fl, a_err)
    print(f"\n  Fit a_FL(N) = a_inf + b/N :   a_inf = {a_inf:.4f} +/- {a_inf_err:.4f} NkT   (b = {b:+.2f})")
    if len(runs) >= 3:
        a_inf_raw, a_inf_raw_err = _fit_raw_logN(Ns, a_raw, a_err)
        print(f"  Cross-check (raw + c lnN/N + b/N): a_inf = {a_inf_raw:.4f} +/- {a_inf_raw_err:.4f} NkT")
    else:
        print("  (cross-check raw+lnN/N fit needs >=3 sizes; skipped)")
    print(f"  Finite-size bias a_FL(N) - a_inf:")
    for r in runs:
        print(f"    N={r['N']:>6}:  {r['a_fl'] - a_inf:+.4f} NkT")

    out = args.plot
    if out is None:
        out = f"fss_{args.label}.png"
    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8))
        x = 1.0 / Ns
        yerr = a_err if np.all(np.isfinite(a_err)) else None
        ax[0].errorbar(x, a_fl, yerr=yerr, fmt='o', color="#1f77b4", capsize=3, label="historical $a_{\\rm FL}$ proxy")
        xs = np.linspace(0, x.max() * 1.05, 50)
        ax[0].plot(xs, a_inf + b * xs, '-', color="#1f77b4", lw=1)
        ax[0].plot(0, a_inf, '*', color="crimson", ms=13, label=f"$a_\\infty$ = {a_inf:.3f}")
        ax[0].set_xlabel("$1/N$"); ax[0].set_ylabel(r"$a_{\rm sol}/Nk_BT$")
        ax[0].set_title(f"{args.label}: extrapolation to $N\\to\\infty$"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        for key, c, mk in [("A0", "#2ca02c", "s"), ("dA1", "#d62728", "^"), ("dA2", "#9467bd", "v")]:
            ax[1].plot(x, [r[key] for r in runs], mk + '-', color=c, ms=5, label=key.replace("dA", r"$\Delta A_").__add__("$") if key != "A0" else "$A_0$")
        ax[1].set_xlabel("$1/N$"); ax[1].set_ylabel(r"component$/Nk_BT$")
        ax[1].set_title("Decomposition: $A_0$ (COM) and $\\Delta A_2$ vary; $\\Delta A_1$ is intensive"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out, dpi=140)
        print(f"\n  wrote {out}")
    except Exception as exc:
        print(f"  (plot skipped: {exc})")


if __name__ == "__main__":
    main()
