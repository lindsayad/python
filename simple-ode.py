def f(u, dt, uold, coeff):
    return u * (1./dt - coeff) - uold / dt

def J(dt, coeff):
    return 1./dt - coeff

def newton_solve(f, J, uini, dt, coeff, rtol):
    u = uini
    fini = f(u, dt, uini, coeff)
    feval = f(u, dt, uini, coeff)
    Jeval = J(dt, coeff)
    print u
    print feval
    print Jeval
    i = 0
    while (abs(feval) / abs(fini) > rtol and i < 5):
        deltau = -feval / Jeval
        u = u + deltau
        feval = f(u, dt, uini, coeff)
        Jeval = J(dt, coeff)
        i = i + 1
        print u
        print feval
        print Jeval
    return u, feval

def stability_measure(dt, coeff):
    result = 1. / dt - coeff
    print result
    return result
