import numpy as np

def R(x):
    return np.array([x[0]**2 + x[0] * x[1] - 3., x[0]*x[1] + x[1]**2 - 6.])

def J(x):
    return np.array([[2.*x[0] + x[1], x[0]], [x[1], x[0] + 2.*x[1]]])

def arnoldi_iteration(A,b,nimp):
     """
     Input
     A: (nxn matrix)
     b: (initial vector)
     nimp: number of iterations

     Returns Q, h

     """
     import numpy as np
     m = A.shape[0] # Shape of the input matrix

     h = np.zeros((nimp+1, nimp))    # Creats a zero matrix of shape (n+1)x n
     Q = np.zeros((m, nimp+1))       # Creats a zero matrix of shape m x n

     q  = b/np.linalg.norm(b)        # Normilize the intput vector
     Q[:, 0] = q                     # Adds q to the first column of Q

     for n in range(nimp):
         v = A.dot(q)                # A*q_0
         for j in range(n+1):
             h[j, n] = np.dot(Q[:,j], v)
             v = v - h[j,n]*Q[:,j]

         h[n+1, n] = np.linalg.norm(v)
         q = v / h[n+1, n]
         Q[:, n+1] = q
     return Q, h

def GMRes(A, b, x0, e, nmax_iter, restart=None):
    r = b - np.asarray(np.dot(A, x0)).reshape(-1)

    x = []
    q = [0] * (nmax_iter)

    x.append(r)

    q[0] = r / np.linalg.norm(r)

    h = np.zeros((nmax_iter + 1, nmax_iter))

    for k in range(nmax_iter):
        y = np.asarray(np.dot(A, q[k])).reshape(-1)

        for j in range(k):
            h[j, k] = np.dot(q[j], y)
            y = y - h[j, k] * q[j]
        h[k + 1, k] = np.linalg.norm(y)
        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            q[k + 1] = y / h[k + 1, k]

        b = np.zeros(nmax_iter + 1)
        b[0] = np.linalg.norm(r)

        result = np.linalg.lstsq(h, b)[0]

        x.append(np.dot(np.asarray(q).transpose(), result) + x0)

    return x

def myGMRes(A, b):
    r = b - np.asarray(np.dot(A, np.array([0, 0]))).reshape(-1)

    x = []

    ini_norm = np.linalg.norm(r)
    norm = ini_norm
    max_its = 30
    its = -1

    m = 1
    while norm / ini_norm > 1e-6 and its < max_its:
        h = np.zeros((m+1, m))
        q = []
        q.append(r / ini_norm)
        for j in range(m):
            wj = np.dot(A, q[j])
            for i in range(j):
                h[i][j] = np.dot(wj, q[i])
                wj = wj - h[i][j] * q[i]
            h[j+1][j] = np.linalg.norm(wj)
            if h[j+1][j] == 0:
                m = j
                break
            q.append(wj / h[j+1][j])

        e1 = np.zeros(m + 1)
        e1[0] = ini_norm
        ym = np.linalg.lstsq(h, e1)[0]
        xm = np.dot(np.asarray(q).transpose()[:,:m], ym)
        x.append(xm)
        m += 1
        norm = np.linalg.norm(b - np.asarray(np.dot(A, xm)).reshape(-1))
        its += 1

    return x

def preGMRes(A, b, x0):
    # preMat = np.linalg.inv(A)
    # preMat = blockPreMat(A)
    preMat = np.identity(2)

    true_ini_resid = b - np.asarray(np.dot(A, x0)).reshape(-1)
    precond_ini_resid = np.dot(preMat, b - np.dot(A, x0))
    beta = np.linalg.norm(precond_ini_resid)

    x = []
    norm = beta
    max_its = 30
    its = 0
    m = 1

    while norm / beta > 1e-6 and its < max_its:
        h = np.zeros((m+1, m))
        e1 = np.zeros(m + 1)
        q = []
        q.append(precond_ini_resid / beta)
        for j in range(1, m + 1):
            wj = np.dot(preMat, np.dot(A, q[j-1]))
            for i in range(1, j + 1):
                h[i-1][j-1] = np.dot(wj, q[i-1])
                wj = wj - h[i-1][j-1] * q[i-1]
            h[j][j-1] = np.linalg.norm(wj)
            if h[j][j-1] < 1e-10:
                m = j
                break
            q.append(wj / h[j][j-1])

        e1[0] = beta
        ym = np.linalg.lstsq(h[:m+1,:m], e1[:m+1])[0]
        xm = np.dot(np.asarray(q).transpose()[:,:m], ym) + x0
        x.append(xm)
        m += 1
        norm = np.linalg.norm(np.dot(preMat, b - np.dot(A, xm)))
        its += 1

    return x, its, q

def blockPreMat(A):
    return np.linalg.inv(np.diag(np.diag(A)))

def newton():
    x = [np.array([0.5, 0.5])]
    res = R(x[0])
    jac = J(x[0])
    while np.linalg.norm(res) > 1e-8:
        update = GMRes(jac, res, np.array([0, 0]), 0, 30)

        x.append(x[-1] - update[-1])
        res = R(x[-1])
        jac = J(x[-1])

    return x

ini = np.array([0.5, 0.5])
x, its, q = preGMRes(J(ini), R(ini), np.array([0, 0]))
xso, itsso, qso = preGMRes(np.array([[1, 1], [3, -4]]), np.array([3, 2]), np.array([1, 2]))
