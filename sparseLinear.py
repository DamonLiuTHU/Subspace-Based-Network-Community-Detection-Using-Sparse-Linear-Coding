import numpy as np
from sklearn import linear_model

def decomposition(S):
    n = np.shape(S)[0]
    F = np.zeros([n, n])
    for i in range(n):
        Sh = np.column_stack((S[:, :i], np.zeros([n, 1]), S[:, i + 1:]))
        lasso = linear_model.Lasso(alpha = 0.05, fit_intercept = False)
        lasso.fit(Sh, S[:, i])
        w = lasso.coef_ / sum(lasso.coef_)
        #w = lasso.coef_
        #print w
        #print np.dot(Sh, np.reshape(w, [6, 1]))
        F[i, :] = F[i, :] + w
    max_dig = []
    for row in F:
        max_dig.append(np.max(row))
    # F = F / max_dig

    F = (F + np.transpose(F))/2

    return F


if __name__ == "__main__":
    S = np.array([[1.0, 0.8, 0.8, 0.7, 0.6, 0.6],
                [0.8, 1.0, 0.8, 0.7, 0.6, 0.6],
                [0.8, 0.8, 1.0, 0.8, 0.7, 0.7],
                [0.7, 0.7, 0.8, 1.0, 0.8, 0.8],
                [0.6, 0.6, 0.7, 0.8, 1.0, 0.8],
                [0.6, 0.6, 0.7, 0.8, 0.8, 1.0]])
    F = decomposition(S)
    print(F)
