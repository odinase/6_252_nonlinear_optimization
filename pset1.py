import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(f, jac, x0, alpha, max_iters: int = 1000, eps: float = 1e-8):
    x = x0.copy()
    xs = [x]

    err = np.inf
    n = 0
    while err > eps and n < max_iters and np.linalg.norm(jac(x)) > eps:
        x = x - alpha*jac(x)
        err = np.linalg.norm(xs[n] - x)
        xs.append(x)

        n += 1

        print(f"n: {n}\nerr: {err}\n||jac||: {np.linalg.norm(x)}")

    return xs

if __name__ == "__main__":
    Q = np.array([
        [50, 49],
        [49, 50]
    ])
    def f(x): return 0.5*x@Q@x
    def jac(x): return Q@x

    x0 = np.array([1.5, -1.0])
    xs = gradient_descent(f, jac, x0, 0.1)