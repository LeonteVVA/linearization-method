# Метод линеаризации для задачи
#   f(x) -> min,  g_i(x) <= 0 (i < m),  g_i(x) = 0 (i >= m).
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls


@dataclass
class Problem:
    f: callable       # целевая функция
    grad_f: callable  # её градиент
    g: callable       # вектор ограничений (g_1, ..., g_s)
    jac_g: callable   # матрица Якоби ограничений, размер s x n
    m: int            # первые m ограничений -- неравенства, остальные -- равенства
    name: str = ""


@dataclass
class Result:
    x: np.ndarray
    lam: np.ndarray   # множители при неравенствах
    mu: np.ndarray    # множители при равенствах
    N: float
    iterations: int
    converged: bool
    history: list = field(default_factory=list)


def violation(gx, m):
    # F(x): насколько сильно нарушены ограничения (0, если x допустима)
    return max(0.0, np.max(gx[:m], initial=0.0), np.max(np.abs(gx[m:]), initial=0.0))


def penalty(problem, x, N):
    # функция штрафа Phi_N(x) = f(x) + N F(x)
    return problem.f(x) + N * violation(problem.g(x), problem.m)


def solve_ldp(G, h):
    # Ближайшая к нулю точка множества {z : G z >= h}.
    # Сводится к задаче min |E u - e|, u >= 0 (NNLS, Лоусон -- Хенсон).
    n = G.shape[1]
    if G.shape[0] == 0:
        return np.zeros(n), np.zeros(0)
    E = np.vstack([G.T, h])
    e = np.zeros(n + 1)
    e[-1] = 1.0
    u, _ = nnls(E, e, maxiter=50 * E.shape[1])
    r = E @ u - e
    if np.linalg.norm(r) < 1e-12:
        raise ValueError("линеаризованные ограничения несовместны")
    w = u / (-r[-1])       # множители Лагранжа
    return G.T @ w, w      # z = G^T w


def direction(c, A_in, b_in, A_eq, b_eq):
    # Вспомогательная задача:
    #   <c, p> + |p|^2 / 2 -> min,  b_in + A_in p <= 0,  b_eq + A_eq p = 0.
    # После замены z = p + c это поиск ближайшей к нулю точки,
    # равенство записываем как два неравенства.
    k, q = A_in.shape[0], A_eq.shape[0]
    G = np.vstack([-A_in, -A_eq, A_eq])
    h = np.concatenate([b_in - A_in @ c, b_eq - A_eq @ c, A_eq @ c - b_eq])
    z, w = solve_ldp(G, h)
    p = z - c
    lam = w[:k]
    mu = w[k:k + q] - w[k + q:]
    return p, lam, mu


def linearization(problem, x0, N=1.0, delta=1.0, eps=0.5, tau=1e-6,
                  max_iter=10000, min_step=1e-12, adapt_N=True):
    x = np.asarray(x0, dtype=float).copy()
    m = problem.m
    history = []

    for k in range(max_iter):
        fx, gx = problem.f(x), problem.g(x)
        df, J = problem.grad_f(x), problem.jac_g(x)
        F = violation(gx, m)

        # берём только неравенства, близкие к самому нарушенному (delta-активные)
        active = np.flatnonzero(gx[:m] >= F - delta)

        # направление p и множители из вспомогательной задачи
        p, lam_act, mu = direction(df, J[active], gx[active], J[m:], gx[m:])
        lam = np.zeros(m)
        lam[active] = lam_act
        Lam = lam.sum() + np.abs(mu).sum()
        p_norm = np.linalg.norm(p)

        rec = dict(k=k, x=x.copy(), f=fx, F=F, p_norm=p_norm,
                   active=len(active), Lam=Lam, N=N, alpha=None)
        history.append(rec)

        # остановка: p почти нулевое и точка почти допустима
        if p_norm <= tau and F <= tau:
            return Result(x, lam, mu, N, k, True, history)

        # если N мало, p может не уменьшать штраф -- увеличиваем N
        if adapt_N and Lam > N:
            N = max(2.0 * N, Lam)
            rec["N"] = N

        # дробим шаг, пока штраф не уменьшится достаточно
        phi = fx + N * F
        alpha = 1.0
        while penalty(problem, x + alpha * p, N) > phi - eps * alpha * p_norm ** 2:
            alpha /= 2.0
            if alpha < min_step:
                break
        rec["alpha"] = alpha

        x = x + alpha * p

    return Result(x, lam, mu, N, max_iter, False, history)
