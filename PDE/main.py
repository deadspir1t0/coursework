from matplotlib.pyplot import style, figure, axes, show, subplots
from numpy import zeros, linspace, sin, pi
from tqdm import tqdm

from conditions import *
from utils import *
from PDE import *


def main():
    # ---------------------------
    # параметры задачи
    # ---------------------------

    # границы задачи
    a = 0
    b = 1
    t0 = 0.0
    T = 0.5

    # параметры сетки
    N = 20
    M = 30

    # параметры уравнения
    eps = 1e-2
    alpha_ROS = (1 + 1j) / 2

    # точность
    delta = 1e-3

    # параметры минимизации
    alpha = 1e-4
    beta = 1e-2

    # начальное приближение
    q0 = zeros(N+1)


    # ---------------------------
    # моделируем наблюдаемые значения функции
    # ---------------------------

    # Модельное q
    x = linspace(a, b, N + 1)
    q_model = sin(3 * pi * x)

    # решаем задачу для модельного q
    forward = PDEBurgForward(a, b, N, t0, T, M, initial, left, right, eps, q_model, alpha_ROS)
    u = forward.solve()
    f_obs = u[-1].copy()


    # ---------------------------
    # решаем задачу
    # ---------------------------

    s = 0
    q_inv = q0.copy()
    J = []

    # входим в цикл
    pbar = tqdm(desc=f"Calculating", unit="iter")
    while True:
        # прямая задача
        forward = PDEBurgForward(a, b, N, t0, T, M, initial, left, right, eps, q_inv, alpha_ROS)
        u = forward.solve()

        # ошибка
        err = integrate((u[-1] - f_obs) ** 2, a, b)
        J.append(err)

        pbar.set_postfix(iter=s, err=f"{err:.2e}")
        pbar.update(1)

        # если точность достигнута - выходим
        if err < delta:
            break

        # сопряжённая задача
        adjoint = PDEBurgAdjoint(a, b, N, t0, T, M, u, f_obs, eps, q_inv, alpha_ROS)
        psi = adjoint.solve()

        # считаем градиент функционала
        grad = zeros(u.shape[1])
        for x in range(len(grad)):
            grad[x] = integrate(u[:, x] * psi[:, x], t0, T) + 2 * alpha * q_inv[x]

        q_inv -= beta * grad    # q(s+1) = q(s) - beta * J'[x]
        s += 1
    pbar.close()


    # ---------------------------
    # строим графики
    # ---------------------------

    x = linspace(a, b, N + 1)

    fig1 = figure(figsize=(8, 5))
    ax1 = axes()

    ax1.plot(x, q_model, '-g', lw=2, label=r'$q_{model}(x)$')
    ax1.plot(x, q_inv, '-or', lw=2, label=r'$q_{inv}(x)$')

    ax1.set_xlabel(r'$x$', fontsize=16)
    ax1.set_ylabel(r'$q(x)$', fontsize=16)
    ax1.grid(True)
    ax1.legend(fontsize=12)

    fig2 = figure(figsize=(8, 5))
    ax2 = axes()

    ax2.plot(J, label=r"$J(q^{(s)})$" + rf",$\ \beta={beta}$")
    ax2.set_xlabel(r"$s$", fontsize=16)
    ax2.set_ylabel(r"$J[q]$", fontsize=16)
    ax2.grid(True)
    ax2.legend(fontsize=8)

    show()


if __name__ == "__main__":
    main()
