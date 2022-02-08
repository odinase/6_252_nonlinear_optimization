import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(f, jac, x0, alpha, max_iters: int = 1000, eps: float = 1e-8):
    x = x0.copy()
    xs = [x0]

    err = np.inf
    n = 0
    while err > eps and n < max_iters and np.linalg.norm(jac(x)) > eps:
        x = x - alpha*jac(x)
        err = np.linalg.norm(xs[n] - x)
        xs.append(x)

        n += 1

    return np.array(xs)

if __name__ == "__main__":
    Q = np.array([
        [50, 49],
        [49, 50]
    ])
    def f(x):
        if len(x.shape) < 3:
            x = x.reshape((1, 1, 2))
        x = x[...,None] 
        out = 0.5*np.squeeze(x.transpose((0, 1, 3, 2))@Q.reshape((1, 1, 2, 2))@x)
        if len(out.shape) == 0:
            out = out.item()
        return out

    def jac(x): return Q@x

    x0 = np.array([1.5, -1.0])
    alphas = [0.1, 0.01, 0.001]
    fig, axes = plt.subplots(ncols=3)
    fig2, ax2 = plt.subplots()

    for alpha, ax in zip(alphas, axes.ravel()):
        xs = gradient_descent(f, jac, x0, alpha, max_iters=200)
        x_min, y_min = xs.min(axis=0)
        x_max, y_max = xs.max(axis=0)
        x = np.linspace(x_min-2, x_max+2, 1000)
        y = np.linspace(y_min-2, y_max+2, 1000)
        X, Y = np.meshgrid(x, y)

        P = np.stack((X, Y)).transpose((1, 2, 0))

        Z = f(P)

        c = ax.pcolormesh(X, Y, Z)
        fig.colorbar(c, ax=ax)
        ax.plot(*xs[1::200].T, 'rx')
        ax.plot(*xs[0], 'gx', ms=1.5, mew=10)
        ax.set_title(f"alpha: {alpha}")

        # fuck it
        fs = np.squeeze(0.5*xs.reshape((-1, 1, 2))@Q.reshape(1, 2, 2)@xs.reshape((-1, 2, 1)))
        ax2.plot(fs, label=f'alpha: {alpha}')

    ax2.legend()
    ax2.set_ylim([-10, 10])
    plt.show()