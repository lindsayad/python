def Ra(rho, T, dT, l, g, mu, k, cp):
    return rho * dT / T * l**3 * g / (mu * k / (rho * cp))

def rho(p, T, M):
    return p * M / (8.3145 * T)
