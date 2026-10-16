# Тестовые задачи. Ограничения: сначала m неравенств g_i <= 0, затем равенства.
import numpy as np

from linearization import Problem


def example_quadratic():
    # x1^2 + x2^2 -> min,  1 - x1 - x2 <= 0
    return Problem(
        f=lambda x: x[0] ** 2 + x[1] ** 2,
        grad_f=lambda x: 2 * x,
        g=lambda x: np.array([1 - x[0] - x[1]]),
        jac_g=lambda x: np.array([[-1.0, -1.0]]),
        m=1, name="квадратичная, линейное неравенство")


def example_ellipse():
    # (x1 - 2)^2 + (x2 - 1)^2 -> min,  x1^2/4 + x2^2 - 1 <= 0,  x1 - 2 x2 + 1 = 0
    return Problem(
        f=lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
        grad_f=lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 1)]),
        g=lambda x: np.array([x[0] ** 2 / 4 + x[1] ** 2 - 1, x[0] - 2 * x[1] + 1]),
        jac_g=lambda x: np.array([[x[0] / 2, 2 * x[1]], [1.0, -2.0]]),
        m=1, name="эллипс и прямая")


def example_circle():
    # x1 + x2 -> min,  x1^2 + x2^2 = 2
    return Problem(
        f=lambda x: x[0] + x[1],
        grad_f=lambda x: np.array([1.0, 1.0]),
        g=lambda x: np.array([x[0] ** 2 + x[1] ** 2 - 2]),
        jac_g=lambda x: np.array([[2 * x[0], 2 * x[1]]]),
        m=0, name="окружность")


def example_rosenbrock():
    # функция Розенброка в круге x1^2 + x2^2 <= 1
    return Problem(
        f=lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
        grad_f=lambda x: np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                                   200 * (x[1] - x[0] ** 2)]),
        g=lambda x: np.array([x[0] ** 2 + x[1] ** 2 - 1]),
        jac_g=lambda x: np.array([[2 * x[0], 2 * x[1]]]),
        m=1, name="Розенброк в круге")


def example_hs71():
    # задача 71 Хока -- Шиттковского:
    # x1 x4 (x1 + x2 + x3) + x3 -> min,  x1 x2 x3 x4 >= 25,  1 <= xj <= 5,
    # x1^2 + x2^2 + x3^2 + x4^2 = 40
    def g(x):
        return np.concatenate([[25 - np.prod(x)], 1 - x, x - 5,
                               [np.sum(x ** 2) - 40]])

    def jac_g(x):
        prod_grad = np.array([np.prod(np.delete(x, j)) for j in range(4)])
        return np.vstack([-prod_grad, -np.eye(4), np.eye(4), 2 * x])

    return Problem(
        f=lambda x: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
        grad_f=lambda x: np.array([x[3] * (2 * x[0] + x[1] + x[2]),
                                   x[0] * x[3],
                                   x[0] * x[3] + 1,
                                   x[0] * (x[0] + x[1] + x[2])]),
        g=g, jac_g=jac_g, m=9, name="HS71")
