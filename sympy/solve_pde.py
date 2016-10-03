from sympy.solvers.pde import pdsolve
from sympy import Function, diff, Eq
from sympy.abc import x, y
f = Function('f')
u = f(x, y)
uxx = u.diff(x).diff(x)
uyy = u.diff(y).diff(y)
eq = Eq(1 - (2*(uxx)) - (3*(uyy)))
print(pdsolve(eq))
