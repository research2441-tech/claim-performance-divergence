"""Prediction-powered inference: mean and moment-rectified OLS estimators with variances."""
import numpy as np
Z = 1.959963984540054
def mean_classical(y):
    return y.mean(), y.std(ddof=1) / np.sqrt(len(y))
def mean_naive(fu):
    return fu.mean(), fu.std(ddof=1) / np.sqrt(len(fu))
def mean_ppi(y, fl, fu, tuned=False):
    n, N = len(y), len(fu)
    lam = 1.0
    if tuned:
        c = np.cov(y, fl)[0, 1]; v = np.var(np.r_[fl, fu], ddof=1)
        lam = np.clip(c / ((1 + n / N) * v), 0, 1) if v > 0 else 0.0
    est = lam * fu.mean() + (y - lam * fl).mean()
    se = np.sqrt(lam ** 2 * fu.var(ddof=1) / N + (y - lam * fl).var(ddof=1) / n)
    return est, se
def ols(X, y):
    A = X.T @ X / len(y); b = X.T @ y / len(y); beta = np.linalg.solve(A, b)
    g = X * (y - X @ beta)[:, None]; Ai = np.linalg.inv(A)
    V = Ai @ np.cov(g.T) @ Ai.T / len(y)
    return beta, np.sqrt(np.diag(V))
def ols_ppi(Xl, yl, Xfl, yfl, Xfu, yfu):
    """Moment-rectified PPI for OLS: both regressors and outcome may be machine-predicted."""
    n, N = len(yl), len(yfu)
    A = Xfu.T @ Xfu / N + (Xl.T @ Xl - Xfl.T @ Xfl) / n
    b = Xfu.T @ yfu / N + (Xl.T @ yl - Xfl.T @ yfl) / n
    beta = np.linalg.solve(A, b); Ai = np.linalg.inv(A)
    gu = Xfu * (yfu - Xfu @ beta)[:, None]
    gd = Xl * (yl - Xl @ beta)[:, None] - Xfl * (yfl - Xfl @ beta)[:, None]
    V = Ai @ (np.cov(gu.T) / N + np.cov(gd.T) / n) @ Ai.T
    return beta, np.sqrt(np.diag(V))
