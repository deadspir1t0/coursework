from numpy import zeros, complex64


def integrate(f, a, b):
    """
    Метод трапеций для уже вычисленных значений.

    y : массив значений функции на равномерной сетке
    """
    N = len(f) - 1
    h = (b - a) / N

    return h * (0.5 * f[0] + f[1:N].sum() + 0.5 * f[N])


# метод прогонки
def tridiagonal_matrix_algorithm(a, b, c, f):
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