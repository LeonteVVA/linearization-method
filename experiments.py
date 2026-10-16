# Эксперименты: решаем тестовые задачи методом линеаризации и проверяем
# ответ библиотечной функцией scipy.optimize.minimize (метод SLSQP).
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import problems
from linearization import linearization

FIG_DIR = Path(__file__).resolve().parent / "figures"

# задача, начальная точка, начальное N
EXPERIMENTS = [
    (problems.example_quadratic, [0.0, 0.0], 2.0),
    (problems.example_ellipse, [-2.0, 2.0], 1.0),
    (problems.example_circle, [2.0, 0.5], 1.0),
    (problems.example_rosenbrock, [-1.5, 1.5], 1.0),
    (problems.example_hs71, [1.0, 5.0, 5.0, 1.0], 1.0),
]


def solve_scipy(problem, x0):
    # SciPy ожидает неравенства в виде c(x) >= 0, поэтому берём -g
    m = problem.m
    cons = [{"type": "ineq", "fun": lambda x: -problem.g(x)[:m],
             "jac": lambda x: -problem.jac_g(x)[:m]}]
    if problem.g(np.asarray(x0, dtype=float)).size > m:
        cons.append({"type": "eq", "fun": lambda x: problem.g(x)[m:],
                     "jac": lambda x: problem.jac_g(x)[m:]})
    res = minimize(problem.f, x0, jac=problem.grad_f, constraints=cons,
                   method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000})
    return res.x


def compare_with_scipy():
    print("Сравнение с scipy.optimize.minimize (SLSQP)")
    results = []
    for idx, (make, x0, N0) in enumerate(EXPERIMENTS, start=1):
        problem = make()
        res = linearization(problem, x0, N=N0)
        x_ref = solve_scipy(problem, x0)
        dx = np.linalg.norm(res.x - x_ref)
        df = abs(problem.f(res.x) - problem.f(x_ref))
        ok = res.converged and dx < 1e-5
        print(f"\nПример {idx} ({problem.name}), итераций: {res.iterations}")
        print(f"  наш метод: x = {res.x}, f = {problem.f(res.x):.8f}")
        print(f"  SLSQP:     x = {x_ref}, f = {problem.f(x_ref):.8f}")
        print(f"  |x - x_SLSQP| = {dx:.1e}, |f - f_SLSQP| = {df:.1e}  ->",
              "совпадает" if ok else "НЕ СОВПАДАЕТ")
        print(f"  F(x) = {res.history[-1]['F']:.1e}, N = {res.N:.3g}, "
              f"lambda = {res.lam}, mu = {res.mu}")
        results.append((problem, res))
    return results


def study_delta():
    print("\nВлияние delta: итераций / среднее число неравенств в подзадаче")
    for idx in (4, 5):
        make, x0, N0 = EXPERIMENTS[idx - 1]
        row = []
        for delta in (0.1, 1.0, 10.0, np.inf):
            res = linearization(make(), x0, N=N0, delta=delta)
            row.append(f"{res.iterations} / {np.mean([h['active'] for h in res.history]):.2f}")
        print(f"  Пример {idx}:", ",  ".join(row))


def study_N():
    print("\nВлияние N: число итераций ('-' -- не сошёлся за 2000 итераций)")
    for idx in (2, 3, 5):
        make, x0, _ = EXPERIMENTS[idx - 1]
        row = []
        for adapt, values in ((True, (0.1, 1.0, 10.0)), (False, (0.1, 0.5, 1.0))):
            for N0 in values:
                res = linearization(make(), x0, N=N0, adapt_N=adapt, max_iter=2000)
                row.append(str(res.iterations) if res.converged else "-")
        print(f"  Пример {idx}: адаптивное N0 = 0.1, 1, 10: {row[:3]};"
              f"  фиксированное N = 0.1, 0.5, 1: {row[3:]}")


def plot_trajectory(problem, res, idx):
    # линии уровня f, допустимое множество и точки x_k
    xs = np.array([h["x"] for h in res.history])
    lo = np.minimum(xs.min(axis=0), res.x) - 0.7
    hi = np.maximum(xs.max(axis=0), res.x) + 0.7
    X1, X2 = np.meshgrid(np.linspace(lo[0], hi[0], 400), np.linspace(lo[1], hi[1], 400))
    pts = np.stack([X1.ravel(), X2.ravel()], axis=1)
    Z = np.array([problem.f(p) for p in pts]).reshape(X1.shape)
    G = np.array([problem.g(p) for p in pts]).reshape(*X1.shape, -1)
    m = problem.m

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.contour(X1, X2, Z, levels=np.unique(np.quantile(Z, np.linspace(0.01, 0.95, 25))),
               cmap="Greys", linewidths=0.6)
    if m > 0:
        feasible = np.all(G[..., :m] <= 0, axis=-1)
        ax.contourf(X1, X2, feasible, levels=[0.5, 1.5], colors=["#9ecae1"], alpha=0.5)
    for i in range(m, G.shape[-1]):
        ax.contour(X1, X2, G[..., i], levels=[0], colors="#08519c", linewidths=1.5)
    ax.plot(xs[:, 0], xs[:, 1], "o-", color="#d94801", ms=3, lw=1)
    ax.plot(*xs[0], "s", color="black", ms=6, label="$x_0$")
    ax.plot(*res.x, "*", color="#a50f15", ms=12, label="$x^*$")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"traj{idx}.pdf")
    fig.savefig(FIG_DIR / f"traj{idx}.png", dpi=120)
    plt.close(fig)


def plot_convergence(results):
    # |p_k| и F(x_k) по итерациям; нули рисуем на уровне 1e-16
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for idx, (_, res) in enumerate(results, start=1):
        k = [h["k"] for h in res.history]
        axes[0].semilogy(k, [max(h["p_norm"], 1e-16) for h in res.history], label=f"Пример {idx}")
        axes[1].semilogy(k, [max(h["F"], 1e-16) for h in res.history], label=f"Пример {idx}")
    axes[0].set_title(r"$\|p_k\|$")
    axes[1].set_title(r"$F(x_k)$")
    for ax in axes:
        ax.set_xlabel("$k$")
        ax.set_xscale("symlog", linthresh=10)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "convergence.pdf")
    fig.savefig(FIG_DIR / "convergence.png", dpi=120)
    plt.close(fig)


def main():
    np.set_printoptions(precision=6, suppress=True)
    FIG_DIR.mkdir(exist_ok=True)
    results = compare_with_scipy()
    study_delta()
    study_N()
    for idx, (problem, res) in enumerate(results, start=1):
        if len(res.x) == 2:
            plot_trajectory(problem, res, idx)
    plot_convergence(results)


if __name__ == "__main__":
    main()
