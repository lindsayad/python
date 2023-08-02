 1/1:
def T(v, e):
    return e/cv
 1/2: from math import cos
 1/3:
def rho(x):
    cos(x)
 1/4:
def rho_u(x):
    cos(1.1*x)
 1/5: rho0 = 3.487882614709243
 1/6:
def rho(x):
    rho0*cos(x)
 1/7:
def rho_u(x):
    rho0*cos(1.1*x)
 1/8:
def u(x):
    rho_u(x) / rho(x)
 1/9: u(.1)
1/10:
def rho(x):
    return rho0*cos(x)
1/11:
def rho_u(x):
    return rho0*cos(1.1*x)
1/12:
def u(x):
    return rho_u(x) / rho(x)
1/13: u(.1)
1/14: u(1.1)
1/15: rho(1.1)
1/16: rho(.1)
1/17:
def rho_et(x):
    return 26.74394130735463*cos(1.2*x)
1/18:
def e(x):
    return rho_et(x) / rho - 0.5*u(x)*u(x)
1/19: e(.1)
1/20:
def e(x):
    return rho_et(x) / rho(x) - 0.5*u(x)*u(x)
1/21: e(.1)
1/22: e(1.1)
1/23:
gamma = 1.4
molar_mass = 29e-3
R = 8.3145
R_specific = R / molar_mass
cp = gamma * R_specific / (gamma - 1)
cv = cp / gamma
1/24:
def T(x):
    return e(x) / cv
1/25: T(.1)
1/26: T(1.1)
1/27:
def c(x):
    return sqrt(gamma * R_specific * T(x))
1/28: from math import sqrt
1/29: c(.1)
1/30: c(1.1)
1/31: u(.1)
1/32: u(1.1)
1/33: rho0
 2/1: 461.40 + 92.20 - 351.80
 3/1: height_brick = .4*2
 3/2: height_side_ref = .1*2
 3/3: height_hot_plenum = 1*.8
 3/4: height_bottom_reflector = .3 + .2 + .3 + .216 + .412
 3/5: cum = height_brick + height_side_ref + height_hot_plenum + height_bottom_reflector
 3/6: cum
 3/7: height_pebble_bed = 20 * .550
 3/8: cum = cum + height_pebble_bed
 3/9: cum
3/10: cavity = 1*.760
3/11: cum = cum + cavity
3/12: cum
 4/1: 496.54 / 9
 4/2: individ = 496.54 / 9
 4/3: individ
 4/4: individ * 2
 5/1: .4 * 12
 5/2: .2 * 12
 5/3: 6. / 52
 5/4: _ * 12
 6/1: 250 + 125 + 100 + 120
 7/1: 118000 * .02
 7/2: 163000 * .08
 8/1: dr_fgm = 500.
 8/2: dr_gfa = 1061.
 8/3: s_fgm = 410.
 8/4: s_gfa = 861.
 8/5: dr_fgx = dr_fgm / dr_gfa
 8/6: s_fgx = s_fgm / s_fga
 8/7: s_fgx = s_fgm / s_gfa
 8/8: dr_fgx
 8/9: (500 + 52.) / (1061. + 100.)
8/10: (410. + 42) / (861. + 100)
8/11: 19. / 31
8/12: 106. / 13.1
8/13: _ - 8.
8/14: _ * 60
 9/1: from math import *
 9/2: 1.1*sin(1.1*0.6)
 9/3: 1.1*sin(1.1*-0.6)
 9/4: 1.1*sin(1.1*-0.9)
 9/5: 1.1*sin(1.1*-1.2)
 9/6: 1.1*sin(0)
 9/7: 1.9999999999999998 + -3.348857074341554
 9/8: _ / 2.
 9/9:
def rho(x):
    return 1.1*sin(1.1*x)
9/10:
def vel(x):
    return 1.1*cos(1.1*x)
9/11:
def rhou(x):
    return rho(x) * vel(x)
9/12: rhou(-0.6)
9/13: vel(-0.6)
9/14: 2 * _ - 2.
9/15: rho(-0.6)
9/16: 2 * _ - 2.
9/17: rho_ghost = _
9/18: vel_ghost = 2 * vel(-0.6) - 2.
9/19: rho_ghost
9/20: vel_ghost
9/21: rho_elem = 2
9/22: vel_elem = 2
9/23: rhou_elem = rho_elem * vel_elem
9/24: rhou_ghost = rho_ghost * vel_ghost
9/25: rhou_elem
9/26: rhou_ghost
9/27: (rhou_elem + rhou_ghost) / 2.
10/1: (1.94444 + 2.5) / 2
11/1:
from math import sqrt

def error(numeric):
    square = (numeric - 1./3.)**2
    return sqrt(square)
11/2: error(1)
11/3:
def numerical_eval(r):
    return r**2
11/4:
def run_sim(nx):
    h = 1. / nx
    integral = 0
    for elem_id in range(nx):
        centroid = elem_id * h + h / 2
        integral = integral + numerical_eval(centroid) * h
    print(error(integral))
11/5: run_sim(1)
11/6: 1./3. - 1./4.
11/7: run_sim(2)
11/8:
r0 = run_sim(1)
r1 = run_sim(2)
r1 / r0
11/9:
def run_sim(nx):
    h = 1. / nx
    integral = 0
    for elem_id in range(nx):
        centroid = elem_id * h + h / 2
        integral = integral + numerical_eval(centroid) * h
    return error(integral))
11/10:
def run_sim(nx):
    h = 1. / nx
    integral = 0
    for elem_id in range(nx):
        centroid = elem_id * h + h / 2
        integral = integral + numerical_eval(centroid) * h
    return error(integral)
11/11: run_sim(1)
11/12: run_sim(2)
11/13:
r0 = run_sim(1)
r1 = run_sim(2)
r1 / r0
11/14:
r0 = run_sim(1)
r1 = run_sim(2)
r0 / r1
11/15: r2 = run_sim(4)
11/16: r1 / r2
12/1: from math import sin, cos
12/2: sin(1.4)
12/3: cos(1.3)
12/4: cos(0)
12/5: cos(-1.3)
12/6: sin(-1.4)
12/7: cos(1.4)
13/1:
def top_flux(x):
    return 1.1*cos(1.3*x)*cos(1.4)
13/2: face_coords =  [-0.984375 , -0.953125 , -0.921875 , -0.890625 , -0.859375 , -0.828125 , -0.796875 , -0.765625 , -0.734375 , -0.703125 , -0.671875 , -0.640625 , -0.609375 , -0.578125 , -0.546875 , -0.515625 , -0.484375 , -0.453125 , -0.421875 , -0.390625 , -0.359375 , -0.328125 , -0.296875 , -0.265625 , -0.234375 , -0.203125 , -0.171875 , -0.140625 , -0.109375 , -0.078125 , -0.046875 , -0.015625 , 0.015625 , 0.046875 , 0.078125 , 0.109375 , 0.140625 , 0.171875 , 0.203125 , 0.234375 , 0.265625 , 0.296875 , 0.328125 , 0.359375 , 0.390625 , 0.421875 , 0.453125 , 0.484375 , 0.515625 , 0.546875 , 0.578125 , 0.609375 , 0.640625 , 0.671875 , 0.703125 , 0.734375 , 0.765625 , 0.796875 , 0.828125 , 0.859375 , 0.890625 , 0.921875 , 0.953125 , 0.984375]
13/3:
for coord in face_coords:
    print(top_flux(coord))
13/4: from math import sin, cos
13/5:
for coord in face_coords:
    print(top_flux(coord))
13/6:
def v(x, y):
    return cos(1.3*x)*cos(1.4*y)
13/7: v(0.5, 1)
14/1: import numpy as np
14/2: import sympy as sp
14/3: def mmsf(x):
14/4:
def mmsf(x):
    return -np.sin(x)
14/5:
def run_test(nx):
    x = np.linspace(0, 1, nx)
14/6:
def run_test(nx):
    x = np.linspace(0, 1, nx)
    u = mmsf(x)
14/7: run_test(4)
14/8: x
14/9:
def run_test(nx):
    x = np.linspace(0, 1, nx)
    u = mmsf(x)
    return x
14/10: x = run_test(4)
14/11: x
14/12:
class Elem():
    def __init__(self, centroid):
        self.centroid = centroid
14/13:
class FaceInfo:
    def __init__(self, elem, neighbor=None):
        self.elem = elem
        self.neighbor = neighbor
14/14:
class Elem():
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node
        self.centroid = (left_node.x() + right_node.x()) / 2
14/15:
class Elem():
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node
        self.centroid = (left_node.x() + right_node.x()) / 2
        
    def centroid(self):
        return self.centroid
14/16:
class Node():
    def __init__(self, x):
        self.x = x
        
    def x(self):
        return self.x
15/1:
def ux(x, y):
    return -sin(x)*cos(y)
15/2: from math import sin, cos
15/3:
def uy(x, y):
    return cos(x)*sin(y)
15/4: ux(0.25)
15/5: ux(0.25, 0)
15/6: uy(0.25, 0)
15/7: ux(.25, .25)
15/8: uy(.25, .25)
15/9: ux(.125, .125)
15/10: uy(.125, .125)
15/11: ux(.375,.125)
15/12: uy(.375,.125)
15/13: ux(.625,.875)
15/14: uy(.625,.875)
15/15: .375 * 4
18/1:
from fipy.tools import numerix
from fipy.tools import inline
from fipy.variables.cellVariable import CellVariable
from fipy import *

class _FacesToCells(CellVariable):
    r"""surface integral of `self.faceVariable`, :math:`\phi_f`
    .. math:: \int_S \phi_f\,dS \approx \frac{\sum_f \phi_f A_f}{V_P}
    Returns
    -------
    integral : CellVariable
        volume-weighted sum
    """

    def __init__(self, faceVariable, mesh = None):
        if not mesh:
            mesh = faceVariable.mesh

        CellVariable.__init__(self, mesh, hasOld = 0, elementshape=faceVariable.shape[:-1])
        self.faceVariable = self._requires(faceVariable)

    def _calcValue(self):
        if inline.doInline and self.faceVariable.rank < 2:
            return self._calcValueInline()
        else:
            return self._calcValueNoInline()

    def _calcValueInline(self):

        NCells = self.mesh.numberOfCells
        ids = self.mesh.cellFaceIDs

        val = self._array.copy()

        inline._runInline("""
        int i;
        for(i = 0; i < numberOfCells; i++)
          {
          int j;
          value[i] = 0.;
          float counter = 0;
          for(j = 0; j < numberOfCellFaces; j++)
            {
              // cellFaceIDs can be masked, which caused subtle and
              // unreproducible problems on OS X (who knows why not elsewhere)
              long id = ids[i + j * numberOfCells];
              if (id >= 0) {
                  value[i] += orientations[i + j * numberOfCells] * faceVariable[id];
                  counter += 1.'
              }
            }
            value[i] = value[i] / counter;
          }
          """,
                          numberOfCellFaces = self.mesh._maxFacesPerCell,
                          numberOfCells = NCells,
                          faceVariable = self.faceVariable.numericValue,
                          ids = numerix.array(ids),
                          value = val,
                          orientations = numerix.array(self.mesh._cellToFaceOrientations),
                          cellVolume = numerix.array(self.mesh.cellVolumes))

        return self._makeValue(value = val)

    def _calcValueNoInline(self):
        ids = self.mesh.cellFaceIDs

        contributions = numerix.take(self.faceVariable, ids, axis=-1)

        # FIXME: numerix.MA.filled casts away dimensions
        s = (numerix.newaxis,) * (len(contributions.shape) - 2) + (slice(0, None, None),) + (slice(0, None, None),)

        faceContributions = contributions

        return numerix.tensordot(numerix.ones(faceContributions.shape[-2], 'd'),
                                 numerix.MA.filled(faceContributions, 0.), (0, -2)) / self.mesh.cellFaceIDs.shape[0]


def custom_interpolation(variable, query_points):

  f_int = interpolate.CloughTocher2DInterpolator(list(zip(variable.mesh.cellCenters.value[0], 
                                                          variable.mesh.cellCenters.value[1])), 
                                                variable.value)
  
  return f_int(query_points[0], query_points[1])
18/2:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_p = custom_interpolation(pressure, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
18/3:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable
import numpy as np

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1000000.
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.8
velocityRelaxation = 0.5
sweeps = 70

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=8)
fh_int = apply_int(fh, rec=8)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad, mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
pressureCorrection.constrain(velocity[0]*velocity[0]/2, mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

    # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
    velocity[0] = xVelocity.arithmeticFaceValue \
                  - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
                  + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
    velocity[1] = yVelocity.arithmeticFaceValue \
                  - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
                  + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0



    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
18/4:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
18/5:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_u = custom_interpolation(xVelocity, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
18/6:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable
import numpy as np
from scipy import interpolate

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1000000.
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.8
velocityRelaxation = 0.5
sweeps = 70

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=8)
fh_int = apply_int(fh, rec=8)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad, mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
pressureCorrection.constrain(velocity[0]*velocity[0]/2, mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

    # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
    velocity[0] = xVelocity.arithmeticFaceValue \
                  - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
                  + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
    velocity[1] = yVelocity.arithmeticFaceValue \
                  - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
                  + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0



    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
18/7:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
18/8:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_u = custom_interpolation(xVelocity, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
18/9:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_p = custom_interpolation(pressure, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
18/10:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_p = custom_interpolation(pressure, query_points)

plt.plot(query_x, new_p)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
19/1:
from fipy.tools import numerix
from fipy.tools import inline
from fipy.variables.cellVariable import CellVariable
from fipy import *

class _FacesToCells(CellVariable):
    r"""surface integral of `self.faceVariable`, :math:`\phi_f`
    .. math:: \int_S \phi_f\,dS \approx \frac{\sum_f \phi_f A_f}{V_P}
    Returns
    -------
    integral : CellVariable
        volume-weighted sum
    """

    def __init__(self, faceVariable, mesh = None):
        if not mesh:
            mesh = faceVariable.mesh

        CellVariable.__init__(self, mesh, hasOld = 0, elementshape=faceVariable.shape[:-1])
        self.faceVariable = self._requires(faceVariable)

    def _calcValue(self):
        if inline.doInline and self.faceVariable.rank < 2:
            return self._calcValueInline()
        else:
            return self._calcValueNoInline()

    def _calcValueInline(self):

        NCells = self.mesh.numberOfCells
        ids = self.mesh.cellFaceIDs

        val = self._array.copy()

        inline._runInline("""
        int i;
        for(i = 0; i < numberOfCells; i++)
          {
          int j;
          value[i] = 0.;
          float counter = 0;
          for(j = 0; j < numberOfCellFaces; j++)
            {
              // cellFaceIDs can be masked, which caused subtle and
              // unreproducible problems on OS X (who knows why not elsewhere)
              long id = ids[i + j * numberOfCells];
              if (id >= 0) {
                  value[i] += orientations[i + j * numberOfCells] * faceVariable[id];
                  counter += 1.'
              }
            }
            value[i] = value[i] / counter;
          }
          """,
                          numberOfCellFaces = self.mesh._maxFacesPerCell,
                          numberOfCells = NCells,
                          faceVariable = self.faceVariable.numericValue,
                          ids = numerix.array(ids),
                          value = val,
                          orientations = numerix.array(self.mesh._cellToFaceOrientations),
                          cellVolume = numerix.array(self.mesh.cellVolumes))

        return self._makeValue(value = val)

    def _calcValueNoInline(self):
        ids = self.mesh.cellFaceIDs

        contributions = numerix.take(self.faceVariable, ids, axis=-1)

        # FIXME: numerix.MA.filled casts away dimensions
        s = (numerix.newaxis,) * (len(contributions.shape) - 2) + (slice(0, None, None),) + (slice(0, None, None),)

        faceContributions = contributions

        return numerix.tensordot(numerix.ones(faceContributions.shape[-2], 'd'),
                                 numerix.MA.filled(faceContributions, 0.), (0, -2)) / self.mesh.cellFaceIDs.shape[0]


def custom_interpolation(variable, query_points):

  f_int = interpolate.CloughTocher2DInterpolator(list(zip(variable.mesh.cellCenters.value[0], 
                                                          variable.mesh.cellCenters.value[1])), 
                                                variable.value)
  
  return f_int(query_points[0], query_points[1])
19/2:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable
import numpy as np
from scipy import interpolate

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1000000.
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.8
velocityRelaxation = 0.5
sweeps = 70

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=8)
fh_int = apply_int(fh, rec=8)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad, mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
pressureCorrection.constrain(velocity[0]*velocity[0]/2, mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

    # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
    velocity[0] = xVelocity.arithmeticFaceValue \
                  - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
                  + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
    velocity[1] = yVelocity.arithmeticFaceValue \
                  - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
                  + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0



    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
19/3:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
19/4:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_u = custom_interpolation(xVelocity, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
19/5:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_p = custom_interpolation(pressure, query_points)

plt.plot(query_x, new_p)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
19/6:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable
import numpy as np
from scipy import interpolate

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1000000.
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.8
velocityRelaxation = 0.5
sweeps = 70

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=8)
fh_int = apply_int(fh, rec=8)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad, mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
pressureCorrection.constrain(velocity[0]*velocity[0]/2, mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

#     # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
#     velocity[0] = xVelocity.arithmeticFaceValue \
#                   - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
#                   + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
#     velocity[1] = yVelocity.arithmeticFaceValue \
#                   - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
#                   + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
#     velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
#     velocity[0, mesh.facesLeft.value] = U
#     velocity[1, mesh.facesLeft.value] = 0

    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
19/7:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
19/8:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable
import numpy as np
from scipy import interpolate

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1000000.
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.8
velocityRelaxation = 0.5
sweeps = 70

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=8)
fh_int = apply_int(fh, rec=8)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad, mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
pressureCorrection.constrain(velocity[0]*velocity[0]/2, mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

#     # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
#     velocity[0] = xVelocity.arithmeticFaceValue \
#                   - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
#                   + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
#     velocity[1] = yVelocity.arithmeticFaceValue \
#                   - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
#                   + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
#     velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
#     velocity[0, mesh.facesLeft.value] = U
#     velocity[1, mesh.facesLeft.value] = 0

    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
19/9:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
19/10:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_u = custom_interpolation(xVelocity, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
19/11:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_p = custom_interpolation(pressure, query_points)

plt.plot(query_x, new_p)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
19/12:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable
import numpy as np
from scipy import interpolate

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1000000.
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.8
velocityRelaxation = 0.5
sweeps = 70

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=8)
fh_int = apply_int(fh, rec=8)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
#                ImplicitSourceTerm(coeff=darcy_int) -\
#                ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
#                ImplicitSourceTerm(coeff=darcy_int) -\
#                ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad, mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
pressureCorrection.constrain(velocity[0]*velocity[0]/2, mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

#     # Compute new body forces
#     body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
#     body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
#     body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
#     body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
#     body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
#     body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal())# - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal())# - body_force_y_int.value
#     bpx[:] = 1/(1 + body_force_x.value/apx)
#     bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

#     # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
#     velocity[0] = xVelocity.arithmeticFaceValue \
#                   - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
#                   + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
#     velocity[1] = yVelocity.arithmeticFaceValue \
#                   - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
#                   + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
#     velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
#     velocity[0, mesh.facesLeft.value] = U
#     velocity[1, mesh.facesLeft.value] = 0

    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue# * \
#            (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue# * \
#            (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
19/13:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable
import numpy as np
from scipy import interpolate

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1000000.
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.8
velocityRelaxation = 0.5
sweeps = 70

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=8)
fh_int = apply_int(fh, rec=8)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
#                ImplicitSourceTerm(coeff=darcy_int) -\
#                ImplicitSourceTerm(coeff=fh_int) +\
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.])# -\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon)
#                ImplicitSourceTerm(coeff=darcy_int) -\
#                ImplicitSourceTerm(coeff=fh_int) +\
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad, mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
pressureCorrection.constrain(velocity[0]*velocity[0]/2, mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

#     # Compute new body forces
#     body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
#     body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
#     body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
#     body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
#     body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
#     body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal())# - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal())# - body_force_y_int.value
#     bpx[:] = 1/(1 + body_force_x.value/apx)
#     bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

#     # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
#     velocity[0] = xVelocity.arithmeticFaceValue \
#                   - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
#                   + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
#     velocity[1] = yVelocity.arithmeticFaceValue \
#                   - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
#                   + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
#     velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
#     velocity[0, mesh.facesLeft.value] = U
#     velocity[1, mesh.facesLeft.value] = 0

    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue# * \
#            (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue# * \
#            (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
19/14:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable
import numpy as np
from scipy import interpolate

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1000000.
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.8
velocityRelaxation = 0.5
sweeps = 70

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=8)
fh_int = apply_int(fh, rec=8)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
#                ImplicitSourceTerm(coeff=darcy_int) -\
#                ImplicitSourceTerm(coeff=fh_int) +\
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon)
#                ImplicitSourceTerm(coeff=darcy_int) -\
#                ImplicitSourceTerm(coeff=fh_int) +\
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad, mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
pressureCorrection.constrain(velocity[0]*velocity[0]/2, mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

#     # Compute new body forces
#     body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
#     body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
#     body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
#     body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
#     body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
#     body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal())# - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal())# - body_force_y_int.value
#     bpx[:] = 1/(1 + body_force_x.value/apx)
#     bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

#     # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
#     velocity[0] = xVelocity.arithmeticFaceValue \
#                   - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
#                   + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
#     velocity[1] = yVelocity.arithmeticFaceValue \
#                   - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
#                   + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
#     velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
#     velocity[0, mesh.facesLeft.value] = U
#     velocity[1, mesh.facesLeft.value] = 0

    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue# * \
#            (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue# * \
#            (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
19/15:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
19/16:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_u = custom_interpolation(xVelocity, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
19/17:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_p = custom_interpolation(pressure, query_points)

plt.plot(query_x, new_p)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
20/1:
from fipy.tools import numerix
from fipy.tools import inline
from fipy.variables.cellVariable import CellVariable
from fipy import *

class _FacesToCells(CellVariable):
    r"""surface integral of `self.faceVariable`, :math:`\phi_f`
    .. math:: \int_S \phi_f\,dS \approx \frac{\sum_f \phi_f A_f}{V_P}
    Returns
    -------
    integral : CellVariable
        volume-weighted sum
    """

    def __init__(self, faceVariable, mesh = None):
        if not mesh:
            mesh = faceVariable.mesh

        CellVariable.__init__(self, mesh, hasOld = 0, elementshape=faceVariable.shape[:-1])
        self.faceVariable = self._requires(faceVariable)

    def _calcValue(self):
        if inline.doInline and self.faceVariable.rank < 2:
            return self._calcValueInline()
        else:
            return self._calcValueNoInline()

    def _calcValueInline(self):

        NCells = self.mesh.numberOfCells
        ids = self.mesh.cellFaceIDs

        val = self._array.copy()

        inline._runInline("""
        int i;
        for(i = 0; i < numberOfCells; i++)
          {
          int j;
          value[i] = 0.;
          float counter = 0;
          for(j = 0; j < numberOfCellFaces; j++)
            {
              // cellFaceIDs can be masked, which caused subtle and
              // unreproducible problems on OS X (who knows why not elsewhere)
              long id = ids[i + j * numberOfCells];
              if (id >= 0) {
                  value[i] += orientations[i + j * numberOfCells] * faceVariable[id];
                  counter += 1.'
              }
            }
            value[i] = value[i] / counter;
          }
          """,
                          numberOfCellFaces = self.mesh._maxFacesPerCell,
                          numberOfCells = NCells,
                          faceVariable = self.faceVariable.numericValue,
                          ids = numerix.array(ids),
                          value = val,
                          orientations = numerix.array(self.mesh._cellToFaceOrientations),
                          cellVolume = numerix.array(self.mesh.cellVolumes))

        return self._makeValue(value = val)

    def _calcValueNoInline(self):
        ids = self.mesh.cellFaceIDs

        contributions = numerix.take(self.faceVariable, ids, axis=-1)

        # FIXME: numerix.MA.filled casts away dimensions
        s = (numerix.newaxis,) * (len(contributions.shape) - 2) + (slice(0, None, None),) + (slice(0, None, None),)

        faceContributions = contributions

        return numerix.tensordot(numerix.ones(faceContributions.shape[-2], 'd'),
                                 numerix.MA.filled(faceContributions, 0.), (0, -2)) / self.mesh.cellFaceIDs.shape[0]


def custom_interpolation(variable, query_points):

  f_int = interpolate.CloughTocher2DInterpolator(list(zip(variable.mesh.cellCenters.value[0], 
                                                          variable.mesh.cellCenters.value[1])), 
                                                variable.value)
  
  return f_int(query_points[0], query_points[1])
20/2:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable
import numpy as np
from scipy import interpolate

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1000000.
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.8
velocityRelaxation = 0.5
sweeps = 70

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=8)
fh_int = apply_int(fh, rec=8)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
#                ImplicitSourceTerm(coeff=darcy_int) -\
#                ImplicitSourceTerm(coeff=fh_int) +\
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon)
#                ImplicitSourceTerm(coeff=darcy_int) -\
#                ImplicitSourceTerm(coeff=fh_int) +\
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad, mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
pressureCorrection.constrain(velocity[0]*velocity[0]/2, mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

#     # Compute new body forces
#     body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
#     body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
#     body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
#     body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
#     body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
#     body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal())# - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal())# - body_force_y_int.value
#     bpx[:] = 1/(1 + body_force_x.value/apx)
#     bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

#     # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
#     velocity[0] = xVelocity.arithmeticFaceValue \
#                   - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
#                   + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
#     velocity[1] = yVelocity.arithmeticFaceValue \
#                   - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
#                   + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
#     velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
#     velocity[0, mesh.facesLeft.value] = U
#     velocity[1, mesh.facesLeft.value] = 0

    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue# * \
#            (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue# * \
#            (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
20/3:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
20/4:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable
import numpy as np
from scipy import interpolate

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1000000.
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.8
velocityRelaxation = 0.5
sweeps = 70

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=8)
fh_int = apply_int(fh, rec=8)
epsilon_int = apply_int(epsilon, rec=8)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon_int.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon_int) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon_int.grad.dot([1.,0.]))/epsilon_int)
#                ImplicitSourceTerm(coeff=darcy_int) -\
#                ImplicitSourceTerm(coeff=fh_int) +\
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon_int.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon_int) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon_int.grad.dot([0.,1.]))/epsilon_int)
#                ImplicitSourceTerm(coeff=darcy_int) -\
#                ImplicitSourceTerm(coeff=fh_int) +\
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad, mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
pressureCorrection.constrain(velocity[0]*velocity[0]/2, mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

#     # Compute new body forces
#     body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
#     body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
#     body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
#     body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
#     body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
#     body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal())# - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal())# - body_force_y_int.value
#     bpx[:] = 1/(1 + body_force_x.value/apx)
#     bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

#     # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
#     velocity[0] = xVelocity.arithmeticFaceValue \
#                   - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
#                   + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
#     velocity[1] = yVelocity.arithmeticFaceValue \
#                   - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
#                   + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
#     velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
#     velocity[0, mesh.facesLeft.value] = U
#     velocity[1, mesh.facesLeft.value] = 0

    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue# * \
#            (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue# * \
#            (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
20/5:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
20/6:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_u = custom_interpolation(xVelocity, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
20/7:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable
import numpy as np
from scipy import interpolate

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1000000.
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.8
velocityRelaxation = 0.5
sweeps = 70

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=8)
fh_int = apply_int(fh, rec=8)
epsilon_int = apply_int(epsilon, rec=0)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon_int.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon_int) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon_int.grad.dot([1.,0.]))/epsilon_int)
#                ImplicitSourceTerm(coeff=darcy_int) -\
#                ImplicitSourceTerm(coeff=fh_int) +\
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon_int.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon_int) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon_int.grad.dot([0.,1.]))/epsilon_int)
#                ImplicitSourceTerm(coeff=darcy_int) -\
#                ImplicitSourceTerm(coeff=fh_int) +\
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad, mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
pressureCorrection.constrain(velocity[0]*velocity[0]/2, mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

#     # Compute new body forces
#     body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
#     body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
#     body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
#     body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
#     body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
#     body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal())# - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal())# - body_force_y_int.value
#     bpx[:] = 1/(1 + body_force_x.value/apx)
#     bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

#     # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
#     velocity[0] = xVelocity.arithmeticFaceValue \
#                   - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
#                   + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
#     velocity[1] = yVelocity.arithmeticFaceValue \
#                   - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
#                   + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
#     velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
#     velocity[0, mesh.facesLeft.value] = U
#     velocity[1, mesh.facesLeft.value] = 0

    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue# * \
#            (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue# * \
#            (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
20/8:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
20/9:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_u = custom_interpolation(xVelocity, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
20/10:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable
import numpy as np
from scipy import interpolate

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1000000.
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.8
velocityRelaxation = 0.5
sweeps = 70

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=8)
fh_int = apply_int(fh, rec=8)
epsilon_int = apply_int(epsilon, rec=10)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon_int.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon_int) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon_int.grad.dot([1.,0.]))/epsilon_int)
#                ImplicitSourceTerm(coeff=darcy_int) -\
#                ImplicitSourceTerm(coeff=fh_int) +\
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon_int.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon_int) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon_int.grad.dot([0.,1.]))/epsilon_int)
#                ImplicitSourceTerm(coeff=darcy_int) -\
#                ImplicitSourceTerm(coeff=fh_int) +\
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad, mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
pressureCorrection.constrain(velocity[0]*velocity[0]/2, mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

#     # Compute new body forces
#     body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
#     body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
#     body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
#     body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
#     body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
#     body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal())# - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal())# - body_force_y_int.value
#     bpx[:] = 1/(1 + body_force_x.value/apx)
#     bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

#     # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
#     velocity[0] = xVelocity.arithmeticFaceValue \
#                   - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
#                   + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
#     velocity[1] = yVelocity.arithmeticFaceValue \
#                   - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
#                   + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
#     velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
#     velocity[0, mesh.facesLeft.value] = U
#     velocity[1, mesh.facesLeft.value] = 0

    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue# * \
#            (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue# * \
#            (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
20/11:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
20/12:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_u = custom_interpolation(xVelocity, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
21/1:
from fipy.tools import numerix
from fipy.tools import inline
from fipy.variables.cellVariable import CellVariable
from fipy import *

class _FacesToCells(CellVariable):
    r"""surface integral of `self.faceVariable`, :math:`\phi_f`
    .. math:: \int_S \phi_f\,dS \approx \frac{\sum_f \phi_f A_f}{V_P}
    Returns
    -------
    integral : CellVariable
        volume-weighted sum
    """

    def __init__(self, faceVariable, mesh = None):
        if not mesh:
            mesh = faceVariable.mesh

        CellVariable.__init__(self, mesh, hasOld = 0, elementshape=faceVariable.shape[:-1])
        self.faceVariable = self._requires(faceVariable)

    def _calcValue(self):
        if inline.doInline and self.faceVariable.rank < 2:
            return self._calcValueInline()
        else:
            return self._calcValueNoInline()

    def _calcValueInline(self):

        NCells = self.mesh.numberOfCells
        ids = self.mesh.cellFaceIDs

        val = self._array.copy()

        inline._runInline("""
        int i;
        for(i = 0; i < numberOfCells; i++)
          {
          int j;
          value[i] = 0.;
          float counter = 0;
          for(j = 0; j < numberOfCellFaces; j++)
            {
              // cellFaceIDs can be masked, which caused subtle and
              // unreproducible problems on OS X (who knows why not elsewhere)
              long id = ids[i + j * numberOfCells];
              if (id >= 0) {
                  value[i] += orientations[i + j * numberOfCells] * faceVariable[id];
                  counter += 1.'
              }
            }
            value[i] = value[i] / counter;
          }
          """,
                          numberOfCellFaces = self.mesh._maxFacesPerCell,
                          numberOfCells = NCells,
                          faceVariable = self.faceVariable.numericValue,
                          ids = numerix.array(ids),
                          value = val,
                          orientations = numerix.array(self.mesh._cellToFaceOrientations),
                          cellVolume = numerix.array(self.mesh.cellVolumes))

        return self._makeValue(value = val)

    def _calcValueNoInline(self):
        ids = self.mesh.cellFaceIDs

        contributions = numerix.take(self.faceVariable, ids, axis=-1)

        # FIXME: numerix.MA.filled casts away dimensions
        s = (numerix.newaxis,) * (len(contributions.shape) - 2) + (slice(0, None, None),) + (slice(0, None, None),)

        faceContributions = contributions

        return numerix.tensordot(numerix.ones(faceContributions.shape[-2], 'd'),
                                 numerix.MA.filled(faceContributions, 0.), (0, -2)) / self.mesh.cellFaceIDs.shape[0]


def custom_interpolation(variable, query_points):

  f_int = interpolate.CloughTocher2DInterpolator(list(zip(variable.mesh.cellCenters.value[0], 
                                                          variable.mesh.cellCenters.value[1])), 
                                                variable.value)
  
  return f_int(query_points[0], query_points[1])
21/2:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1e9
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.1
velocityRelaxation = 0.9
sweeps = 71

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
epsilon = apply_int(epsilon, rec=30)
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=1)
fh_int = apply_int(fh, rec=1)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#xVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
xVelocity.faceGrad[1].constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.faceGrad.constrain(0., mesh.facesRight)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#pressureCorrection.constrain(xVelocity.arithmeticFaceValue*xVelocity.arithmeticFaceValue/2, mesh.facesRight)
pressureCorrection.constrain(0., mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

    # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
    velocity[0] = xVelocity.arithmeticFaceValue \
                  - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
                  + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
    velocity[1] = yVelocity.arithmeticFaceValue \
                  - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
                  + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0



    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    #velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
21/3:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
21/4:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_u = custom_interpolation(xVelocity, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
21/5:
from fipy.tools import numerix
from fipy.tools import inline
from fipy.variables.cellVariable import CellVariable
from fipy import *
import numpy as np

class _FacesToCells(CellVariable):
    r"""surface integral of `self.faceVariable`, :math:`\phi_f`
    .. math:: \int_S \phi_f\,dS \approx \frac{\sum_f \phi_f A_f}{V_P}
    Returns
    -------
    integral : CellVariable
        volume-weighted sum
    """

    def __init__(self, faceVariable, mesh = None):
        if not mesh:
            mesh = faceVariable.mesh

        CellVariable.__init__(self, mesh, hasOld = 0, elementshape=faceVariable.shape[:-1])
        self.faceVariable = self._requires(faceVariable)

    def _calcValue(self):
        if inline.doInline and self.faceVariable.rank < 2:
            return self._calcValueInline()
        else:
            return self._calcValueNoInline()

    def _calcValueInline(self):

        NCells = self.mesh.numberOfCells
        ids = self.mesh.cellFaceIDs

        val = self._array.copy()

        inline._runInline("""
        int i;
        for(i = 0; i < numberOfCells; i++)
          {
          int j;
          value[i] = 0.;
          float counter = 0;
          for(j = 0; j < numberOfCellFaces; j++)
            {
              // cellFaceIDs can be masked, which caused subtle and
              // unreproducible problems on OS X (who knows why not elsewhere)
              long id = ids[i + j * numberOfCells];
              if (id >= 0) {
                  value[i] += orientations[i + j * numberOfCells] * faceVariable[id];
                  counter += 1.'
              }
            }
            value[i] = value[i] / counter;
          }
          """,
                          numberOfCellFaces = self.mesh._maxFacesPerCell,
                          numberOfCells = NCells,
                          faceVariable = self.faceVariable.numericValue,
                          ids = numerix.array(ids),
                          value = val,
                          orientations = numerix.array(self.mesh._cellToFaceOrientations),
                          cellVolume = numerix.array(self.mesh.cellVolumes))

        return self._makeValue(value = val)

    def _calcValueNoInline(self):
        ids = self.mesh.cellFaceIDs

        contributions = numerix.take(self.faceVariable, ids, axis=-1)

        # FIXME: numerix.MA.filled casts away dimensions
        s = (numerix.newaxis,) * (len(contributions.shape) - 2) + (slice(0, None, None),) + (slice(0, None, None),)

        faceContributions = contributions

        return numerix.tensordot(numerix.ones(faceContributions.shape[-2], 'd'),
                                 numerix.MA.filled(faceContributions, 0.), (0, -2)) / self.mesh.cellFaceIDs.shape[0]


def custom_interpolation(variable, query_points):

  f_int = interpolate.CloughTocher2DInterpolator(list(zip(variable.mesh.cellCenters.value[0], 
                                                          variable.mesh.cellCenters.value[1])), 
                                                variable.value)
  
  return f_int(query_points[0], query_points[1])
21/6:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_u = custom_interpolation(xVelocity, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
21/7:
from fipy.tools import numerix
from fipy.tools import inline
from fipy.variables.cellVariable import CellVariable
from fipy import *
import numpy as np
from scipy import interpolate

class _FacesToCells(CellVariable):
    r"""surface integral of `self.faceVariable`, :math:`\phi_f`
    .. math:: \int_S \phi_f\,dS \approx \frac{\sum_f \phi_f A_f}{V_P}
    Returns
    -------
    integral : CellVariable
        volume-weighted sum
    """

    def __init__(self, faceVariable, mesh = None):
        if not mesh:
            mesh = faceVariable.mesh

        CellVariable.__init__(self, mesh, hasOld = 0, elementshape=faceVariable.shape[:-1])
        self.faceVariable = self._requires(faceVariable)

    def _calcValue(self):
        if inline.doInline and self.faceVariable.rank < 2:
            return self._calcValueInline()
        else:
            return self._calcValueNoInline()

    def _calcValueInline(self):

        NCells = self.mesh.numberOfCells
        ids = self.mesh.cellFaceIDs

        val = self._array.copy()

        inline._runInline("""
        int i;
        for(i = 0; i < numberOfCells; i++)
          {
          int j;
          value[i] = 0.;
          float counter = 0;
          for(j = 0; j < numberOfCellFaces; j++)
            {
              // cellFaceIDs can be masked, which caused subtle and
              // unreproducible problems on OS X (who knows why not elsewhere)
              long id = ids[i + j * numberOfCells];
              if (id >= 0) {
                  value[i] += orientations[i + j * numberOfCells] * faceVariable[id];
                  counter += 1.'
              }
            }
            value[i] = value[i] / counter;
          }
          """,
                          numberOfCellFaces = self.mesh._maxFacesPerCell,
                          numberOfCells = NCells,
                          faceVariable = self.faceVariable.numericValue,
                          ids = numerix.array(ids),
                          value = val,
                          orientations = numerix.array(self.mesh._cellToFaceOrientations),
                          cellVolume = numerix.array(self.mesh.cellVolumes))

        return self._makeValue(value = val)

    def _calcValueNoInline(self):
        ids = self.mesh.cellFaceIDs

        contributions = numerix.take(self.faceVariable, ids, axis=-1)

        # FIXME: numerix.MA.filled casts away dimensions
        s = (numerix.newaxis,) * (len(contributions.shape) - 2) + (slice(0, None, None),) + (slice(0, None, None),)

        faceContributions = contributions

        return numerix.tensordot(numerix.ones(faceContributions.shape[-2], 'd'),
                                 numerix.MA.filled(faceContributions, 0.), (0, -2)) / self.mesh.cellFaceIDs.shape[0]


def custom_interpolation(variable, query_points):

  f_int = interpolate.CloughTocher2DInterpolator(list(zip(variable.mesh.cellCenters.value[0], 
                                                          variable.mesh.cellCenters.value[1])), 
                                                variable.value)
  
  return f_int(query_points[0], query_points[1])
21/8:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_u = custom_interpolation(xVelocity, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
21/9:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_p = custom_interpolation(pressure, query_points)

plt.plot(query_x, new_p)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
21/10:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_p = custom_interpolation(pressure, query_points)

plt.plot(query_x, new_p)
plt.xlabel('x $[m]$')
plt.xlabel('p $[Pa]$')
plt.grid()
21/11:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1e9
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.1
velocityRelaxation = 0.9
sweeps = 71

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
epsilon = apply_int(epsilon, rec=30)
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=1)
fh_int = apply_int(fh, rec=1)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#xVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
# xVelocity.faceGrad[1].constrain(0., mesh.facesBottom| mesh.facesTop)
# xVelocity.faceGrad.constrain(0., mesh.facesRight)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#pressureCorrection.constrain(xVelocity.arithmeticFaceValue*xVelocity.arithmeticFaceValue/2, mesh.facesRight)
pressureCorrection.constrain(0., mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

    # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
    velocity[0] = xVelocity.arithmeticFaceValue \
                  - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
                  + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
    velocity[1] = yVelocity.arithmeticFaceValue \
                  - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
                  + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0



    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    #velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
21/12:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
21/13:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_u = custom_interpolation(xVelocity, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
21/14:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1e9
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.1
velocityRelaxation = 0.9
sweeps = 71

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
epsilon = apply_int(epsilon, rec=30)
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=1)
fh_int = apply_int(fh, rec=1)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#xVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
# xVelocity.faceGrad[1].constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.faceGrad.constrain(0., mesh.facesRight)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#pressureCorrection.constrain(xVelocity.arithmeticFaceValue*xVelocity.arithmeticFaceValue/2, mesh.facesRight)
pressureCorrection.constrain(0., mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

    # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
    velocity[0] = xVelocity.arithmeticFaceValue \
                  - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
                  + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
    velocity[1] = yVelocity.arithmeticFaceValue \
                  - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
                  + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0



    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    #velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
21/15:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
21/16:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_u = custom_interpolation(xVelocity, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
21/17:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_p = custom_interpolation(pressure, query_points)

plt.plot(query_x, new_p)
plt.xlabel('x $[m]$')
plt.xlabel('p $[Pa]$')
plt.grid()
21/18:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1e9
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.1
velocityRelaxation = 0.9
sweeps = 71

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
epsilon = apply_int(epsilon, rec=0)
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=1)
fh_int = apply_int(fh, rec=1)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#xVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
# xVelocity.faceGrad[1].constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.faceGrad.constrain(0., mesh.facesRight)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#pressureCorrection.constrain(xVelocity.arithmeticFaceValue*xVelocity.arithmeticFaceValue/2, mesh.facesRight)
pressureCorrection.constrain(0., mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

    # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
    velocity[0] = xVelocity.arithmeticFaceValue \
                  - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
                  + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
    velocity[1] = yVelocity.arithmeticFaceValue \
                  - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
                  + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0



    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    #velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
21/19:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
21/20:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_u = custom_interpolation(xVelocity, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
21/21:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1e9
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.8
velocityRelaxation = 0.9
sweeps = 71

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
epsilon = apply_int(epsilon, rec=30)
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=1)
fh_int = apply_int(fh, rec=1)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#xVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
# xVelocity.faceGrad[1].constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.faceGrad.constrain(0., mesh.facesRight)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#pressureCorrection.constrain(xVelocity.arithmeticFaceValue*xVelocity.arithmeticFaceValue/2, mesh.facesRight)
pressureCorrection.constrain(0., mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

    # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
    velocity[0] = xVelocity.arithmeticFaceValue \
                  - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
                  + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
    velocity[1] = yVelocity.arithmeticFaceValue \
                  - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
                  + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0



    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    #velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
21/22:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1e9
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.6
velocityRelaxation = 0.9
sweeps = 71

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
epsilon = apply_int(epsilon, rec=30)
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=1)
fh_int = apply_int(fh, rec=1)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#xVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
# xVelocity.faceGrad[1].constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.faceGrad.constrain(0., mesh.facesRight)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#pressureCorrection.constrain(xVelocity.arithmeticFaceValue*xVelocity.arithmeticFaceValue/2, mesh.facesRight)
pressureCorrection.constrain(0., mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

    # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
    velocity[0] = xVelocity.arithmeticFaceValue \
                  - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
                  + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
    velocity[1] = yVelocity.arithmeticFaceValue \
                  - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
                  + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0



    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    #velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
21/23:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1e9
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.4
velocityRelaxation = 0.9
sweeps = 71

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
epsilon = apply_int(epsilon, rec=30)
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=1)
fh_int = apply_int(fh, rec=1)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#xVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
# xVelocity.faceGrad[1].constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.faceGrad.constrain(0., mesh.facesRight)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#pressureCorrection.constrain(xVelocity.arithmeticFaceValue*xVelocity.arithmeticFaceValue/2, mesh.facesRight)
pressureCorrection.constrain(0., mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

    # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
    velocity[0] = xVelocity.arithmeticFaceValue \
                  - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
                  + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
    velocity[1] = yVelocity.arithmeticFaceValue \
                  - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
                  + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0



    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    #velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
21/24:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1e9
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.2
velocityRelaxation = 0.9
sweeps = 71

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
epsilon = apply_int(epsilon, rec=30)
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=1)
fh_int = apply_int(fh, rec=1)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#xVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
# xVelocity.faceGrad[1].constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.faceGrad.constrain(0., mesh.facesRight)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#pressureCorrection.constrain(xVelocity.arithmeticFaceValue*xVelocity.arithmeticFaceValue/2, mesh.facesRight)
pressureCorrection.constrain(0., mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

    # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
    velocity[0] = xVelocity.arithmeticFaceValue \
                  - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
                  + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
    velocity[1] = yVelocity.arithmeticFaceValue \
                  - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
                  + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0



    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    #velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
21/25:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1e9
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.1
velocityRelaxation = 0.9
sweeps = 71

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
epsilon = apply_int(epsilon, rec=30)
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=1)
fh_int = apply_int(fh, rec=1)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#xVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
# xVelocity.faceGrad[1].constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.faceGrad.constrain(0., mesh.facesRight)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#pressureCorrection.constrain(xVelocity.arithmeticFaceValue*xVelocity.arithmeticFaceValue/2, mesh.facesRight)
pressureCorrection.constrain(0., mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

    # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
    velocity[0] = xVelocity.arithmeticFaceValue \
                  - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
                  + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
    velocity[1] = yVelocity.arithmeticFaceValue \
                  - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
                  + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0



    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    #velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
21/26:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1e9
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.1
velocityRelaxation = 0.9
sweeps = 71

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
epsilon = apply_int(epsilon, rec=30)
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=1)
fh_int = apply_int(fh, rec=1)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#xVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
# xVelocity.faceGrad[1].constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.faceGrad.constrain(0., mesh.facesRight)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#pressureCorrection.constrain(xVelocity.arithmeticFaceValue*xVelocity.arithmeticFaceValue/2, mesh.facesRight)
pressureCorrection.constrain(0., mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

    # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
    velocity[0] = xVelocity.arithmeticFaceValue \
                  - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
                  + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
    velocity[1] = yVelocity.arithmeticFaceValue \
                  - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
                  + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0



    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    #velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
21/27:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
21/28:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_u = custom_interpolation(xVelocity, query_points)

plt.plot(query_x, new_u)
plt.xlabel('x $[m]$')
plt.xlabel('u $[m/s]$')
plt.grid()
21/29:
import matplotlib.pyplot as plt

query_x = np.linspace(0., max(mesh.cellCenters[0]), 30)
query_y = query_x * 0 + 0.1
query_points = np.vstack([query_x, query_y])
new_p = custom_interpolation(pressure, query_points)

plt.plot(query_x, new_p)
plt.xlabel('x $[m]$')
plt.xlabel('p $[Pa]$')
plt.grid()
22/1:
!pip install fipy
#!pip install petsc4py
#!pip install pyamg
#!pip install pysparse
import numpy as np
from scipy import interpolate
22/2:
from fipy.tools import numerix
from fipy.tools import inline
from fipy.variables.cellVariable import CellVariable
from fipy import *
import numpy as np
from scipy import interpolate

class _FacesToCells(CellVariable):
    r"""surface integral of `self.faceVariable`, :math:`\phi_f`
    .. math:: \int_S \phi_f\,dS \approx \frac{\sum_f \phi_f A_f}{V_P}
    Returns
    -------
    integral : CellVariable
        volume-weighted sum
    """

    def __init__(self, faceVariable, mesh = None):
        if not mesh:
            mesh = faceVariable.mesh

        CellVariable.__init__(self, mesh, hasOld = 0, elementshape=faceVariable.shape[:-1])
        self.faceVariable = self._requires(faceVariable)

    def _calcValue(self):
        if inline.doInline and self.faceVariable.rank < 2:
            return self._calcValueInline()
        else:
            return self._calcValueNoInline()

    def _calcValueInline(self):

        NCells = self.mesh.numberOfCells
        ids = self.mesh.cellFaceIDs

        val = self._array.copy()

        inline._runInline("""
        int i;
        for(i = 0; i < numberOfCells; i++)
          {
          int j;
          value[i] = 0.;
          float counter = 0;
          for(j = 0; j < numberOfCellFaces; j++)
            {
              // cellFaceIDs can be masked, which caused subtle and
              // unreproducible problems on OS X (who knows why not elsewhere)
              long id = ids[i + j * numberOfCells];
              if (id >= 0) {
                  value[i] += orientations[i + j * numberOfCells] * faceVariable[id];
                  counter += 1.'
              }
            }
            value[i] = value[i] / counter;
          }
          """,
                          numberOfCellFaces = self.mesh._maxFacesPerCell,
                          numberOfCells = NCells,
                          faceVariable = self.faceVariable.numericValue,
                          ids = numerix.array(ids),
                          value = val,
                          orientations = numerix.array(self.mesh._cellToFaceOrientations),
                          cellVolume = numerix.array(self.mesh.cellVolumes))

        return self._makeValue(value = val)

    def _calcValueNoInline(self):
        ids = self.mesh.cellFaceIDs

        contributions = numerix.take(self.faceVariable, ids, axis=-1)

        # FIXME: numerix.MA.filled casts away dimensions
        s = (numerix.newaxis,) * (len(contributions.shape) - 2) + (slice(0, None, None),) + (slice(0, None, None),)

        faceContributions = contributions

        return numerix.tensordot(numerix.ones(faceContributions.shape[-2], 'd'),
                                 numerix.MA.filled(faceContributions, 0.), (0, -2)) / self.mesh.cellFaceIDs.shape[0]


def custom_interpolation(variable, query_points):

  f_int = interpolate.CloughTocher2DInterpolator(list(zip(variable.mesh.cellCenters.value[0], 
                                                          variable.mesh.cellCenters.value[1])), 
                                                variable.value)
  
  return f_int(query_points[0], query_points[1])
22/3:
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable

D = 0.2
N = 20
dL = D / N
viscosity = 0.1
U = 1.0
Dp = 1e9
#0.8 for pressure and 0.5 for velocity are typical relaxation values for SIMPLE
pressureRelaxation = 0.1
velocityRelaxation = 0.9
sweeps = 71

mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
velocity = FaceVariable(mesh=mesh, rank=1)
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.5

def apply_int(field, rec=1):
  for i in range(rec):
    field = _FacesToCells(field.arithmeticFaceValue)
  return field

epsilon = 1 - porosity
epsilon = apply_int(epsilon, rec=30)
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

# note: we really don't care about the number of interpolations here
# since we are solving the pressure equation in the non-interpolated version
darcy_int = apply_int(darcy, rec=1)
fh_int = apply_int(fh, rec=1)


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([1., 0.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=xVelocity*(epsilon.grad.dot([1.,0.]))/epsilon)
yVelocityEq = -PowerLawConvectionTerm(coeff=velocity/(epsilon.arithmeticFaceValue**2)) + \
               DiffusionTermCorrection(coeff=viscosity/epsilon) - \
               pressure.grad.dot([0., 1.]) -\
               ImplicitSourceTerm(coeff=darcy_int) -\
               ImplicitSourceTerm(coeff=fh_int) +\
               ImplicitSourceTerm(coeff=yVelocity*(epsilon.grad.dot([0.,1.]))/epsilon) # (u u) \cdot \nabla \epsilon
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
bpx = CellVariable(mesh=mesh, value=1.)
bpy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1)
ap_field[0] = bpx/apx * mesh.cellVolumes
ap_field[1] = bpy/apy * mesh.cellVolumes
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

#xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
#xVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
#yVelocity.faceGrad.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
yVelocity.constrain(0., mesh.facesLeft)
# xVelocity.faceGrad[1].constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.faceGrad.constrain(0., mesh.facesRight)
#pressureCorrection.faceGrad[0].constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#X, Y = mesh.faceCenters
#pressureCorrection.constrain(0., mesh.facesRight)
#bcsPC = (FixedGradient(faces=mesh.facesRight, value=viscosity * (xVelocity.faceGrad.divergence).arithmeticFaceValue),)
#pressureCorrection.constrain(viscosity*xVelocity.faceGrad[0], mesh.facesRight)
#pressureCorrection.constrain(xVelocity.arithmeticFaceValue*xVelocity.arithmeticFaceValue/2, mesh.facesRight)
pressureCorrection.constrain(0., mesh.facesRight)


for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    # Compute new body forces
    body_force_x_int = (darcy_int + fh_int) * xVelocity * mesh.cellVolumes
    body_force_y_int = (darcy_int + fh_int) * yVelocity * mesh.cellVolumes
    body_force_x = (darcy + fh)*xVelocity * mesh.cellVolumes
    body_force_y = (darcy + fh)*yVelocity * mesh.cellVolumes
    body_force_x_int_rc = (body_force_x.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes
    body_force_y_int_rc = (body_force_y.arithmeticFaceValue/mesh._faceAreas).divergence*mesh.cellVolumes

    ## update the ap coefficient from the matrix diagonal
    apx[:] = -numerix.asarray(xmat.takeDiagonal()) - body_force_x_int.value
    apy[:] = -numerix.asarray(ymat.takeDiagonal()) - body_force_y_int.value
    bpx[:] = 1/(1 + body_force_x.value/apx)
    bpy[:] = 1/(1 + body_force_y.value/apx)
    ap_field[0] = bpx/apx * mesh.cellVolumes
    ap_field[1] = bpy/apy * mesh.cellVolumes

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    # Computing modified RC terms
    coef_dx = mesh.cellVolumes / apx
    coef_dy = mesh.cellVolumes / apy

    # Todo: not sure if the volume force correction should be done with the interpolated volume forces ot the non-inteprolated ones
    velocity[0] = xVelocity.arithmeticFaceValue \
                  - bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * facepresgrad[0] - (coef_dx * presgrad[0]).arithmeticFaceValue) \
                  + bpx.arithmeticFaceValue*(coef_dx.arithmeticFaceValue * body_force_x.arithmeticFaceValue - (coef_dx * body_force_x_int_rc).arithmeticFaceValue)
    velocity[1] = yVelocity.arithmeticFaceValue \
                  - bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * facepresgrad[1] - (coef_dy * presgrad[1]).arithmeticFaceValue) \
                  + bpy.arithmeticFaceValue*(coef_dy.arithmeticFaceValue * body_force_y.arithmeticFaceValue - (coef_dy * body_force_y_int_rc).arithmeticFaceValue)
    velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0



    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / apx.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_x.arithmeticFaceValue-body_force_x_int.arithmeticFaceValue)

    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / apy.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])\
         + contrvolume / apx.arithmeticFaceValue * \
           (body_force_y.arithmeticFaceValue-body_force_y_int.arithmeticFaceValue)

    #velocity[1, mesh.facesBottom.value | mesh.facesTop.value] = 0.
    velocity[0, mesh.facesLeft.value] = U
    velocity[1, mesh.facesLeft.value] = 0

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               apx * bpx * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               apy * bpy * mesh.cellVolumes)

    if sweep%10 == 0:
        print('sweep:', sweep, ', x residual:', xres, \
                              ', y residual', yres, \
                              ', p residual:', pres, \
                              ', continuity:', max(abs(rhs)))
22/4:
import matplotlib.pyplot as plt

xc, yc = mesh.x.value, mesh.y.value

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value, 30)
plt.colorbar()
plt.title('X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value, 30)
plt.colorbar()
plt.title('V-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, xVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous X-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, yVelocity.value/epsilon, 30)
plt.colorbar()
plt.title('Porous Y-Velocity')

plt.figure(figsize=[8,2])
plt.tricontourf(xc, yc, pressure.value, 30)
plt.colorbar()
plt.title('Pressure')
23/1:
!pip install fipy
#!pip install petsc4py
#!pip install pyamg
#!pip install pysparse
import numpy as np
from scipy import interpolate
from fipy import *
23/2:
# Defining problem
from fipy import *
from fipy.variables.faceGradVariable import _FaceGradVariable

##############################################################
# Problem parameters
#############################################################
D = 1.0
N = 20
#dL = D / N
dL = 1.0
viscosity = 1.1 # 0.0001
U = 0.1
pressureRelaxation = 1.0
velocityRelaxation = 1.0
sweeps = 200
Dp = 0.1


##############################################################
# Problem domain
#############################################################

#mesh = Grid2D(Lx=D*4, Ly=D, dx=dL, dy=dL)
#mesh = Grid2D(Lx=4*D, Ly=D, dx=1./16., dy=1./16.)
#mesh = Grid2D(Lx=4, Ly=4, nx=4, ny=4)
mesh = Grid2D(Lx=50, Ly=10, nx=50, ny=10)
#mesh = Grid2D(Lx=4., Ly=.2, nx=80, ny=4)
#mesh = Grid2D(Lx=10, Ly=1, nx=100, ny=10)


##############################################################
# Problem variables
#############################################################

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity', value=0.0)
yVelocity = CellVariable(mesh=mesh, name='Y velocity')
adv_xVelocity = CellVariable(mesh=mesh, name='X velocity', value=0.0)
adv_yVelocity = CellVariable(mesh=mesh, name='Y velocity')
porosity = CellVariable(mesh=mesh, value=0.0)
velocity_mag = (xVelocity**2 + yVelocity**2)**0.5

porosity[(mesh.x >= 0.4) & (mesh.x <= 0.6)] = 0.0 #0.5

epsilon = 1 - porosity
darcy = 150*viscosity/Dp**2*(1-epsilon)**2/epsilon**3
fh = 1.75/Dp*(1-epsilon)/epsilon**3*velocity_mag

velocity = FaceVariable(mesh=mesh, rank=1)
velocity[0] = 0.0 #U
velocity[1] = 0.0
phi      = FaceVariable(mesh=mesh, rank=1)


##############################################################
# Symbolic equations
#############################################################

xVelocityEq = -CentralDifferenceConvectionTerm(coeff=velocity) + \
               DiffusionTermCorrection(coeff=viscosity) - \
               pressure.grad.dot([1., 0.]) #-\
               #ImplicitSourceTerm(coeff=darcy) -\
               #ImplicitSourceTerm(coeff=fh)
yVelocityEq = -CentralDifferenceConvectionTerm(coeff=velocity) + \
               DiffusionTermCorrection(coeff=viscosity) - \
               pressure.grad.dot([0., 1.]) #-\
               #ImplicitSourceTerm(coeff=darcy) -\
               #ImplicitSourceTerm(coeff=fh)
               
apx = CellVariable(mesh=mesh, value=1.)
apy = CellVariable(mesh=mesh, value=1.)
ap_field = CellVariable(mesh=mesh, rank=1) #FaceVariable(mesh=mesh, rank=1)
#ap_field[0] = 1./apx * mesh.cellVolumes #1./ apx.arithmeticFaceValue*mesh._faceAreas * mesh._cellDistances
#ap_field[1] = 1./apy * mesh.cellVolumes #1./ apx.arithmeticFaceValue*mesh._faceAreas * mesh._cellDistances
pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) + phi.divergence


##############################################################
# Boundary Conditions
#############################################################

xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
xVelocity.constrain(U, mesh.facesLeft)
xVelocity.faceGrad.constrain(0. * mesh.faceNormals, where = mesh.facesRight)
yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
yVelocity.constrain(0., mesh.facesLeft)
yVelocity.faceGrad.constrain(0. * mesh.faceNormals, where = mesh.facesRight)
adv_xVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
adv_xVelocity.constrain(U, mesh.facesLeft)
adv_xVelocity.faceGrad.constrain(0. * mesh.faceNormals, where = mesh.facesRight)
adv_yVelocity.constrain(0., mesh.facesBottom| mesh.facesTop)
adv_yVelocity.constrain(0., mesh.facesLeft)
adv_yVelocity.faceGrad.constrain(0. * mesh.faceNormals, where = mesh.facesRight)
#pressure.constrain(velocity[0]*velocity[0]/2, mesh.facesRight)
pressure.constrain(0., mesh.facesRight)
pressure.faceGrad.constrain(0. * mesh.faceNormals, where = mesh.facesLeft | mesh.facesBottom| mesh.facesTop)
#pressure.constrain(0., mesh.facesRight)
#xVelocity.faceGrad.constrain(0. * mesh.faceNormals, where = mesh.facesBottom| mesh.facesTop | mesh.facesRight)
# velocity.constrain(0, mesh.facesBottom| mesh.facesTop)
# velocity[0].constrain(U, mesh.facesLeft)
# velocity[1].constrain(0., mesh.facesLeft)

##############################################################
# Utils
#############################################################
volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume = volume.arithmeticFaceValue

advection_velocity_x = xVelocity.value.copy()
advection_velocity_y = yVelocity.value.copy()

verbose = False
bool_face_variables = False
for sweep in range(1):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix(); xVelocityEq.cacheRHSvector()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    xrhs = xVelocityEq.RHSvector

    yVelocityEq.cacheMatrix(); yVelocityEq.cacheRHSvector()
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ymat = yVelocityEq.matrix
    yrhs = yVelocityEq.RHSvector

    if verbose:
      print ("------ Iteration:", sweep)
      print ("Momentum predictor x-matrix")
      print (xmat)
      print ("Momentum predictor x-rhs")
      print (xrhs)
      print ("ustar in x direction")
      print (xVelocity)
      print ("Momentum predictor y-matrix")
      print (ymat)
      print ("Momentum predictor y-rhs")
      print (yrhs)
      print ("vstar in y direction")
      print (yVelocity)

    ## update the ap coefficient from the matrix diagonal
    apx[:] = numerix.asarray(xmat.takeDiagonal())
    apy[:] = numerix.asarray(ymat.takeDiagonal())
    ap_field[0] = 1./apx #*mesh.cellVolumes
    ap_field[1] = 1./apy #*mesh.cellVolumes

    if verbose:
      print ("Ainv x-diection")
      print (ap_field[0])
      print ("Ainv y-diection")
      print (ap_field[1])

    ## update the coefficients of the H action
    H_x = xmat.copy(); H_y = ymat.copy();
    H_x.addAtDiagonal(-xmat.takeDiagonal()); H_y.addAtDiagonal(-ymat.takeDiagonal())
    H_x_f = numerix.dot(-H_x.matrix, xVelocity.value) + (xrhs - pressure.grad.dot([1., 0.]).value) #*mesh.cellVolumes
    H_y_f = numerix.dot(-H_y.matrix, yVelocity.value) + (yrhs - pressure.grad.dot([0., 1.]).value) #*mesh.cellVolumes

    if verbose:
      print ("Hhat x-diection")
      print (H_x_f)
      print ("Hhat y-diection")
      print (H_y_f)

    ## updating phis
    u_asterix = H_x_f/apx
    v_asterix = H_y_f/apy
    phi[0] = u_asterix.arithmeticFaceValue
    phi[1] = v_asterix.arithmeticFaceValue

    if verbose:
      print ("Ainv*Hhat x-diection")
      print (u_asterix)
      print ("Ainv*Hhat y-diection")
      print (v_asterix)

    ## solve the pressure correction equation
    # pressureCorrectionEq = DiffusionTermCorrection(coeff=ap_field) + phi.divergence
    pressureCorrectionEq.cacheMatrix(); pressureCorrectionEq.cacheRHSvector()
    pres = pressureCorrectionEq.sweep(var=pressure)
    pmat = pressureCorrectionEq.matrix
    prhs = pressureCorrectionEq.RHSvector

    if verbose:
      print ("Pressure predictor matrix")
      print (pmat)
      print ("Pressure predictor rhs")
      print (prhs)
      print ("pressure")
      print (pressure)

    # Updating velocity field
    if bool_face_variables:
      velocity[0] = pressureRelaxation*(phi[0] - (ap_field[0] * pressure.grad.dot([1., 0.])).arithmeticFaceValue) + (1-pressureRelaxation) * velocity[0]
      velocity[1] = pressureRelaxation*(phi[1] - (ap_field[1] * pressure.grad.dot([0., 1.])).arithmeticFaceValue) + (1-pressureRelaxation) * velocity[1]
    else:
      adv_xVelocity = pressureRelaxation*(u_asterix[:] - (ap_field[0] * pressure.grad.dot([1., 0.]))[:]) + (1-pressureRelaxation)*adv_xVelocity
      adv_yVelocity = pressureRelaxation*(v_asterix[:] - (ap_field[1] * pressure.grad.dot([0., 1.]))[:]) + (1-pressureRelaxation)*adv_yVelocity

    if verbose:
      print ("Corrected velocity x-direction")
      print (pressureRelaxation*(u_asterix - (ap_field[0] * pressure.grad.dot([1., 0.]))) + (1-pressureRelaxation)*xVelocity)
      print ("Corrected velocity y-direction")
      print (pressureRelaxation*(v_asterix - (ap_field[1] * pressure.grad.dot([0., 1.]))) + (1-pressureRelaxation)*yVelocity)

    if bool_face_variables:
      velocity[..., mesh.facesBottom.value | mesh.facesTop.value] = 0.
      velocity[0, mesh.facesLeft.value] = U
      velocity[1, mesh.facesLeft.value] = 0
    else:
      velocity[0] = adv_xVelocity.arithmeticFaceValue
      velocity[1] = adv_yVelocity.arithmeticFaceValue

    if sweep%10 == 0:
      print('sweep:', sweep, ', x residual:', xres, \
                            ', y residual', yres, \
                            ', p residual:', pres, \
                            ', continuity:', max(abs(prhs)))
    
      print('u_x: ', np.mean(xVelocity.value))
      print('u_x_res: ', xres)
      print('u_y: ', np.mean(yVelocity.value))
      print('u_y_res: ', yres)
      print('phi_0: ', np.mean(phi[0].value))
      print('phi_1: ', np.mean(phi[1].value))
      print('p: ', np.mean(pressure.value))
      print('p_x: ', np.mean(pressure.grad[0].value))
      print('p_y: ', np.mean(pressure.grad[1].value))
      print('vel_x: ', np.mean(velocity[0]))
      print('vel_y: ', np.mean(velocity[1]))



plot = True
if plot:
  import matplotlib.pyplot as plt

  xc, yc = mesh.x.value, mesh.y.value

  plt.figure(figsize=[8,2])
  plt.tricontourf(xc, yc, xVelocity.value, 30)
  plt.colorbar()
  plt.title('X-Velocity')

  plt.figure(figsize=[8,2])
  plt.tricontourf(xc, yc, yVelocity.value, 30)
  plt.colorbar()
  plt.title('V-Velocity')

  plt.figure(figsize=[8,2])
  plt.tricontourf(xc, yc, pressure.value, 30)
  plt.colorbar()
  plt.title('Pressure')
26/1:
def Ra(rho, T, dT, l, g, mu, k, cp):
    return rho * dT / T * l**3 * g / (mu * k / (rho * cp))
26/2:
def rho(p, T):
    return p * .029 / (8.3145 * T)
26/3: Ra(rho(1e5, 305), 305, 10, 10, 1, 1e-2, 1e-1, 1)
26/4: Ra(rho(1e5, 305), 305, 10, 10, 1, 1e-2, 1e-2, 1)
26/5: %history
27/1: cd python
27/2: %import rayleigh.py
27/3: %run ./rayleigh.py
27/4: Ra(rho(1e5, 305, .029), 305, 10, 10, 1, 1e-2, 1e-1, 1)
27/5: Ra(rho(1e5, 305, .029), 305, 10, 10, 1, 1e-2, 1e-2, 1)
27/6: Ra(rho(1e5, 305, .029), 305, 10, 10, 1, 4e-2, 1e-2, 1)
29/1: (2. / 3)**2 * 2
29/2: from math import pi
31/1: 11 * 20
32/1: 38 * 20
33/1: 20 * 0.6
33/2: .1 * 10000. / 12.
33/3: 43. * 20.
33/4: 5.9 * 20.
33/5: 43 * 20.
33/6: 0.1 * 1. / 5.e-5
33/7: 0.1 / 8e-5
34/1: 1.523737e-11 / 3.195270e-12
34/2: 2.469457e-11 / 1.523737e-11
34/3: 9.498625e-11 / 2.469457e-11
35/1: (1.544845e+02 - 1e-3 * 1.544847e+05) / 6.218171e+05
36/1: from math import sin, cos
36/2:
def rotate(theta, x, y):
    return (x*cos(theta) - y*sin(theta), x*sin(theta)+y*cos(theta))
36/3: rt_tl = rotate(pi/2, 1, 1)
36/4: from math import pi
36/5: rt_tl = rotate(pi/2, 1, 1)
36/6: rt_tl
36/7: rt_tr = rotate(pi/2, 3, 1)
36/8: rt_tr
36/9: rt_bl = rotate(pi/2, 1, 0)
36/10: rt_br = rotate(pi/2, 3, 0)
36/11: rt_bl
36/12: rt_br
36/13:  box = (rt_tl, rt_tr, rt_bl, rt_br)
36/14: scaled_box = 5 * box
36/15: scaled_box
36/16: scaled_box = (5*i for i in box)
36/17: scaled_box
36/18: scaled_box = [5*i for i in box]
36/19: scaled_box
36/20: box
36/21: list(box)
36/22: scaled_box = [5*i for i in list(box)]
36/23: scaled_box
36/24: tr_bl[0]
36/25: rt_bl[0]
36/26: scaled_box = [(5*i[0], 5*i[1]) for i in list(box)]
36/27: scaled_box
37/1: 1 * 1 * 2 / (4e-3)
38/1: 1.5 / 0.05
38/2: from math import pi
38/3: 96 / (6.335 * pi * 1.5**2)
39/1: 101 * 51 * 31
40/1: %run Benefits
40/2: %run Benefits
41/1:
def tau(L, rho, c, k):
    return L**2 * rho * c / k
41/2: tau(.01, 1000, 4.2e3, 0.59)
41/3: 700 / 10
42/1: r7_src_lines = 3657
42/2: rc_hd_lines = 1879
42/3: r7_lines = r7_src_lines + rc_hd_lines
42/4: r7_lines
42/5: sam_lines = 33510
43/1: 7.52136e-7 - 4.36188e-7
43/2: _ * 0.1
43/3: (7.52136e-7 - 4.36188e-7) > 0.1 * 3.09047e-06
44/1:
def time(starting_pace, miles):
    if (miles == 0)
44/2:
if miles == 0:
    return
44/3:
def time(pace, miles):
    if miles == 0:
        return pace
    return pace + time(pace - .25, miles - 1)
44/4: time(10, 1)
44/5:
def time(pace, miles):
    if miles == 1:
        return pace
    return pace + time(pace - .25, miles - 1)
44/6: time(10, 1)
44/7: time(10, 2)
44/8: time(10, 3)
44/9: time(10, 12)
45/1: 100. / 83
45/2: 120. / 83
46/1: 0.8**2 + 0.2**2
46/2: from math import sqrt
46/3: sqrt(0.8**2 + 0.2**2)
47/1: sophie_sep = 1100. + .12 * 11000
47/2: sophie_sep
47/3: sophie_sep = 1100. + .12 * (25000 - 11000)
47/4: sophie_sep
47/5: alex_sep = 16290. + .24 * (170000 - 35375)
47/6: alex_sep
47/7: alex_sep = 16290. + .24 * (170000 - 95375)
47/8: alex_sep
47/9: together_sep = sophie_sep + alex_sep
47/10: together_sep
47/11: 25 + 170
47/12: together = 32580 + .24 * (195000 - 190750)
47/13: together
48/1: import matplotlib.pyplot as plt
48/2: x = [0, 1]
48/3: y = [0, 1]
48/4: plt.plot(x, y)
48/5: plt.show()
48/6: plt.plot(x, y, marker="o")
48/7: plt.show()
48/8: plt.scatter(x, y)
48/9: plt.show()
48/10: x = [.312684, .144432, .687939, .301469, .0327292, .496076, .589888, .300529]
48/11: y = [7.40195, 7.58277, 7.60056, 7.26087, 7.41742, 7.31349, 7.57396, 8.43873]
48/12: plt.scatter(x, y)
48/13: plt.show()
48/14:
x = [0.312684, 0.144432, 0.687939, 0.301469, 0.0327292, 0.496076, 0.589888, 0.300529]

y = [7.40195,  7.58277,  7.60056,  7.26087,   7.41742, 7.31349,  7.57396,  8.43873]
48/15: x
48/16: y
48/17: x = [0.312684, 0.144432, 0.687939, 0.301469, 0.0327292, 0.496076, 0.589888, 0.300529]
48/18: y = [7.40195,  7.58277,  7.60056,  7.26087,   7.41742, 7.31349,  7.57396,  8.43873]
48/19: plt.scatter(x, y)
48/20: plt.show()
49/1: from math import pi
49/2: pizzeta = pi * (5)**2
49/3: pizza = pi * (6.5)**2
49/4: pizzeta
49/5: pizza
49/6: pizzeta_cost = 17.
49/7: pizza_cost = 23.
49/8: pizzeta_rate = pizzeta / pizzeta_cost
49/9: pizzeta_rate
49/10: pizza_rate = pizza / pizz_cost
49/11: pizza_rate = pizza / pizza_cost
49/12: pizza_rate
49/13: 4 * pizzeta / (3 * pizzeta_cost)
50/1: R = 0.1
50/2: reff = 0.6
50/3: from math import sqrt
50/4: sqrt(2*R**2/reff)
50/5: newR = _
50/6: newR
50/7: P = math.pi * newR**2 * reff
50/8: from math import pi
50/9: P = pi * newR**2 * reff
50/10: P
50/11: R = 1e-4
50/12: newR = sqrt(2*R**2/reff)
50/13: newR
50/14: F0 = 2.546e9
50/15: P = pi * newR**2 * F0 * reff
50/16: P
51/1: 7.94 / 2.49
52/1: 18 + 18 + 44
52/2: 64 + 56 + 128
52/3: 80 / 3.
52/4: 248 / 3.
54/1: 0.0528308**2 + 0.**2 + 0.**2 + 0.**2 + 0.0576442**2 + 0.**2 + 0.**2 + 0.**2 + 0.0597575**2 + 0.**2 + 0.**2 + 0.**2 + -0.493087**2 + 0.**2 + -0.604854**2 + 0.**2 + -0.616124**2 + 0.**2 + 0.**2 + 0.**2 + -8.36277e-18**2 + -1.84726e-18**2 + 0.**2 + -3.20764e-18**2 + -4.76563e-17**2 + -0.0293504**2 + 0.0293504**2
54/2: 0.**2
54/3: (0.0528308)**2 + (0.)**2 + (0.)**2 + (0.)**2 + (0.0576442)**2 + (0.)**2 + (0.)**2 + (0.)**2 + (0.0597575)**2 + (0.)**2 + (0.)**2 + (0.)**2 + (-0.493087)**2 + (0.)**2 + (-0.604854)**2 + (0.)**2 + (-0.616124)**2 + (0.)**2 + (0.)**2 + (0.)**2 + (-8.36277e-18)**2 + (-1.84726e-18)**2 + (0.)**2 + (-3.20764e-18)**2 + (-4.76563e-17)**2 + (-0.0293504)**2 + (0.0293504)**2
55/1: Re = 0.2 * 10 * 3279 / .005926
55/2: Re
55/3: .15 * 2.2 * 3279 / .005926
55/4: 0.2 * 10 / (.15 * 2.2)
55/5: .15 * 2.2 * 3279 / .005926 * 6.06
55/6: Re = 0.2 * 3.3 * 3279 / .005926
55/7: Re
55/8: 0.2 * 1 * 3279 / .005926
56/1: 1. / 9.683324e-01
56/2: 1. / 1.0063926406
57/1: 160**2
57/2: 1. / 9.78696578e-01
57/3: 1. / 1.0063926423
57/4: 1. / 1.006703
57/5: 1. / 1.0067034549
   1:
def num_vertices(n):
    return n + 1
   2:
def num_nodes(n):
    return num_vertices(n) + n
   3:
def num_pressure_dofs(n):
    return (num_vertices(n))**2
   4:
def num_vel_dofs(n, dim):
    return dim*(num_nodes(n))**2
   5:
return total_dofs(n, dim):
    return num_pressure_dofs(n) + num_vel_dofs(n, dim)
   6:
def total_dofs(n, dim):
    return num_pressure_dofs(n) + num_vel_dofs(n, dim)
   7: total_dofs(100, 2)
   8: total_dofs(500, 2)
   9: total_dofs(1000, 2)
  10: total_dofs(2000, 2)
  11: total_dofs(3000, 2)
  12: total_dofs(2900, 2)
  13: total_dofs(2800, 2)
  14: total_dofs(2790, 2)
  15: total_dofs(2780, 2)
  16: total_dofs(2785, 2)
  17: total_dofs(2787, 2)
  18: total_dofs(2788, 2)
  19: total_dofs(2789, 2)
  20: % / 20000
  21: total_dofs(2789, 2) / 20000
  22: _ / 48
  23: 73 * 48
  24: %save generate-total-dofs.py
  25: %history -g -f generate-total-dofs.py
