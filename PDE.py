# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')   # попробовать сначала так

from numpy import zeros, linspace, tanh, complex64, cosh
from matplotlib.pyplot import style, figure, axes, show, subplots
from celluloid import Camera
import numpy as np


class PDEBurgForward:
    def __init__(self, a, b, N, t_0, T, M, eps, q, alpha):
        # параметры сетки
        self.a = a
        self.b = b
        self.N = N
        self.t_0 = t_0
        self.T = T
        self.M = M

        # координатная сетка
        self.h = (self.b - self.a) / self.N
        self.x = linspace(self.a, self.b, self.N + 1)

        # временная сетка
        self.tau = (self.T - self.t_0) / self.M
        self.t = linspace(self.t_0, self.T, self.M + 1)

        # данные задачи
        self.eps = eps
        self.q = q
        self.alpha = alpha

        # массив решения прямой задачи
        self.u = zeros((self.M + 1, self.N + 1))

        # начальное условие
        self.u[0] = self.__u_init(self.x)

    # начальное условие
    def __u_init(self, x):
        return np.exp(-(x-1)**2/2) - np.exp(-(x+1)**2/2)

    # левое граничное условие
    def __u_left(self, t):
        return 0

    # правое граничное условие
    def __u_right(self, t):
        return 0

    # правая часть системы ОДУ
    def __f(self, y, t):
        f = zeros(self.N - 1)

        f[0] = (
            self.eps * (y[1] - 2 * y[0] + self.__u_left(t)) / self.h ** 2
            + y[0] * (y[1] - self.__u_left(t)) / (2 * self.h)
            - y[0] * self.q[0]
        )

        for n in range(1, self.N - 2):
            f[n] = (
                self.eps * (y[n + 1] - 2 * y[n] + y[n - 1]) / self.h ** 2
                + y[n] * (y[n + 1] - y[n - 1]) / (2 * self.h)
                - y[n] * self.q[n]
            )

        f[self.N - 2] = (
            self.eps * (self.__u_right(t) - 2 * y[self.N - 2] + y[self.N - 3]) / self.h ** 2
            + y[self.N - 2] * (self.__u_right(t) - y[self.N - 3]) / (2 * self.h)
            - y[self.N - 2] * self.q[self.N - 2]
        )

        return f

    # диагонали матрицы [E - alpha*tau*f_y]
    def __diagonals_preparation_forward(self, y, t):
        a = zeros(self.N - 1, dtype=complex64)
        b = zeros(self.N - 1, dtype=complex64)
        c = zeros(self.N - 1, dtype=complex64)

        a[0] = 1. - self.alpha * self.tau * (
            -2 * self.eps / self.h ** 2
            + (y[1] - self.__u_left(t)) / (2 * self.h)
            - self.q[0]
        )
        c[0] = - self.alpha * self.tau * (
            self.eps / self.h ** 2 + y[0] / (2 * self.h)
        )

        for n in range(1, self.N - 2):
            b[n] = - self.alpha * self.tau * (
                self.eps / self.h ** 2 - y[n] / (2 * self.h)
            )
            a[n] = 1. - self.alpha * self.tau * (
                -2 * self.eps / self.h ** 2
                + (y[n + 1] - y[n - 1]) / (2 * self.h)
                - self.q[n]
            )
            c[n] = - self.alpha * self.tau * (
                self.eps / self.h ** 2 + y[n] / (2 * self.h)
            )

        b[self.N - 2] = - self.alpha * self.tau * (
            self.eps / self.h ** 2 - y[self.N - 2] / (2 * self.h)
        )
        a[self.N - 2] = 1. - self.alpha * self.tau * (
            -2 * self.eps / self.h ** 2
            + (self.__u_right(t) - y[self.N - 3]) / (2 * self.h)
            - self.q[self.N - 2]
        )

        return a, b, c

    # метод прогонки
    def __tridiagonal_matrix_algorithm(self, a, b, c, f):
        n = len(f)
        v = zeros(n, dtype=complex64)
        w_1 = zeros(n, dtype=complex64)

        w = a[0]
        w_1[0] = f[0] / w

        for i in range(1, n):
            v[i - 1] = c[i - 1] / w
            w = a[i] - b[i] * v[i - 1]
            w_1[i] = (f[i] - b[i] * w_1[i - 1]) / w

        for j in range(n - 2, -1, -1):
            w_1[j] = w_1[j] - v[j] * w_1[j + 1]

        return w_1

    def solve(self):
        # внутренние значения начального условия
        y = self.__u_init(self.x[1:self.N])

        for m in range(self.M):
            diagonal, codiagonal_down, codiagonal_up = self.__diagonals_preparation_forward(
                y, self.t[m]
            )

            w_1 = self.__tridiagonal_matrix_algorithm(
                diagonal,
                codiagonal_down,
                codiagonal_up,
                self.__f(y, self.t[m] + self.tau / 2)
            )

            y = y + self.tau * w_1.real

            self.u[m + 1, 0] = self.__u_left(self.t[m + 1])
            self.u[m + 1, 1:self.N] = y
            self.u[m + 1, self.N] = self.__u_right(self.t[m + 1])

        return self.u


class PDEBurgAdjoint:
    def __init__(self, a, b, N, t_0, T, M, u, f_obs, eps, q, alpha):
        # параметры сетки
        self.a = a
        self.b = b
        self.N = N
        self.t_0 = t_0
        self.T = T
        self.M = M

        # координатная сетка
        self.h = (self.b - self.a) / self.N
        self.x = linspace(self.a, self.b, self.N + 1)

        # временная сетка
        self.tau = (self.T - self.t_0) / self.M
        self.t = linspace(self.t_0, self.T, self.M + 1)

        # данные задачи
        self.u = u
        self.f_obs = f_obs
        self.eps = eps
        self.q = q
        self.alpha = alpha

        # массив решения сопряжённой задачи
        self.v = zeros((self.M + 1, self.N + 1))

        # терминальное условие
        self.v[-1, 0] = self.__v_left(self.t[-1])
        self.v[-1, 1:self.N] = self.__v_init(self.x)
        self.v[-1, self.N] = self.__v_right(self.t[-1])

    # терминальное условие
    def __v_init(self, x):
        return -2 * (self.u[-1, 1:self.N] - self.f_obs[1:self.N])

    # левое граничное условие
    def __v_left(self, t):
        return 0

    # правое граничное условие
    def __v_right(self, t):
        return 0

    # правая часть системы ОДУ
    def __g(self, u_m, z, t):
        g = zeros(self.N - 1)

        g[0] = (
            u_m[0] * (z[1] - self.__v_left(t)) / (2 * self.h)
            - self.eps * (z[1] - 2 * z[0] + self.__v_left(t)) / (self.h ** 2)
            + self.q[0] * z[0]
        )

        for n in range(1, self.N - 2):
            g[n] = (
                u_m[n] * (z[n + 1] - z[n - 1]) / (2 * self.h)
                - self.eps * (z[n + 1] - 2 * z[n] + z[n - 1]) / (self.h ** 2)
                + self.q[n] * z[n]
            )

        g[self.N - 2] = (
            u_m[self.N - 2] * (self.__v_right(t) - z[self.N - 3]) / (2 * self.h)
            - self.eps * (self.__v_right(t) - 2 * z[self.N - 2] + z[self.N - 3]) / (self.h ** 2)
            + self.q[self.N - 2] * z[self.N - 2]
        )

        return g

    # диагонали матрицы [E - alpha*tau*g_z]
    def __diagonals_preparation_adjoint(self, u_m, t):
        a = zeros(self.N - 1, dtype=complex64)
        b = zeros(self.N - 1, dtype=complex64)
        c = zeros(self.N - 1, dtype=complex64)

        a[0] = 1. + self.alpha * self.tau * (2 * self.eps / self.h ** 2 + self.q[0])
        c[0] = + self.alpha * self.tau * (- self.eps / self.h ** 2 + u_m[0] / (2 * self.h))

        for n in range(1, self.N - 2):
            b[n] = + self.alpha * self.tau * (- self.eps / self.h ** 2 - u_m[n] / (2 * self.h))
            a[n] = 1. + self.alpha * self.tau * (2 * self.eps / self.h ** 2 + self.q[n])
            c[n] = + self.alpha * self.tau * (- self.eps / self.h ** 2 + u_m[n] / (2 * self.h))

        b[self.N - 2] = + self.alpha * self.tau * (- self.eps / self.h ** 2 - u_m[self.N - 2] / (2 * self.h))
        a[self.N - 2] = 1. + self.alpha * self.tau * (2 * self.eps / self.h ** 2 + self.q[self.N - 2])

        return a, b, c

    # метод прогонки
    def __tridiagonal_matrix_algorithm(self, a, b, c, f):
        n = len(f)
        v = zeros(n, dtype=complex64)
        w_1 = zeros(n, dtype=complex64)

        w = a[0]
        w_1[0] = f[0] / w

        for i in range(1, n):
            v[i - 1] = c[i - 1] / w
            w = a[i] - b[i] * v[i - 1]
            w_1[i] = (f[i] - b[i] * w_1[i - 1]) / w

        for j in range(n - 2, -1, -1):
            w_1[j] = w_1[j] - v[j] * w_1[j + 1]

        return w_1

    def solve(self):
        # внутренние значения терминального условия
        z = self.__v_init(self.x[1:self.N])

        for m in reversed(range(1, self.M + 1)):
            u_half = (self.u[m, 1:self.N] + self.u[m - 1, 1:self.N]) / 2

            diagonal, codiagonal_down, codiagonal_up = self.__diagonals_preparation_adjoint(
                u_half, self.t[m]
            )

            w_1 = self.__tridiagonal_matrix_algorithm(
                diagonal,
                codiagonal_down,
                codiagonal_up,
                self.__g(u_half, z, self.t[m] - self.tau / 2)
            )

            z = z - self.tau * w_1.real

            self.v[m - 1, 0] = self.__v_left(self.t[m - 1])
            self.v[m - 1, 1:self.N] = z
            self.v[m - 1, self.N] = self.__v_right(self.t[m - 1])

        return self.v


def __main__():
    # ---------------------------
    # параметры задачи
    # ---------------------------

    a = -6.0
    b = 6.0
    t_0 = 0.0
    T = 1.0
    eps = 1e-2
    alpha = (1 + 1j) / 2
    N = 300
    M = 500

    x = np.linspace(a, b, N + 1)
    # q = np.sin(3 * np.pi * x[1:N])
    q = zeros(N)

    # прямая задача
    forward = PDEBurgForward(a, b, N, t_0, T, M, eps, q, alpha)
    u = forward.solve()

    # если взять f_obs = u[-1], то psi будет тождественно нулевой
    # поэтому делаем нетривиальные модельные данные
    f_obs = u[-1].copy()
    f_obs += 0.05 * np.exp(-200 * (x - 0.45) ** 2)

    # сопряжённая задача
    adjoint = PDEBurgAdjoint(a, b, N, t_0, T, M, u, f_obs, eps, q, alpha)
    psi = adjoint.solve()

    # ---------------------------
    # Анимация
    # ---------------------------
    style.use('dark_background')
    fig, (ax1, ax2) = subplots(2, 1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.4)

    ax1.set_title(r'Прямая задача', fontsize=14)
    ax2.set_title(r'Сопряжённая задача', fontsize=14)

    camera = Camera(fig)

    ax1.set_xlim(a, b)
    ax1.set_ylim(1.5 * u.min(), 1.5 * u.max())
    ax1.set_xlabel(r'$x$', fontsize=20)
    ax1.set_ylabel(r'$u$', fontsize=20)

    ax2.set_xlim(a, b)
    ax2.set_ylim(1.5 * psi.min(), 1.5 * psi.max())
    ax2.set_xlabel(r'$x$', fontsize=20)
    ax2.set_ylabel(r'$\Psi$', fontsize=20)

    frame_step = 9
    for m in range(0, M + 1, frame_step):
        ax1.plot(x, u[m], color='y', ls='-', lw=2, label=r'$u(x,t)$')
        ax2.plot(x, psi[m], color='c', ls='-', lw=2, label=r'$\Psi(x,t)$')
        camera.snap()

    animation = camera.animate(interval=20, repeat=True, blit=True)
    show()


if __name__ == '__main__':
    __main__()
