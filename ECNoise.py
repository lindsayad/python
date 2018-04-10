from math import sqrt, fabs
import numpy as np

#
#  Determines the noise of a function from the function values
#
#     [fnoise,level,inform] = ECnoise(nf,fval)
#
#  The user must provide the function value at nf equally-spaced points.
#  For example, if nf = 7, the user could provide
#
#     f(x-3h), f(x-2h), f(x-h), f(x), f(x+h), f(x+2h), f(x+3h)
#
#  in the array fval. Although nf >= 4 is allowed, the use of at least
#  nf = 7 function evaluations is recommended.
#
#  Noise will not be detected by this code if the function values differ
#  in the first digit.
#
#  If noise is not detected, the user should increase or decrease the
#  spacing h according to the ouput value of inform.  In most cases,
#  the subroutine detects noise with the initial value of h.
#
#  On exit:
#    fnoise is set to an estimate of the function noise;
#       fnoise is set to zero if noise is not detected.
#
#    level is set to estimates for the noise. The k-th entry is an
#      estimate from the k-th difference.
#
#    inform is set as follows:
#      inform = 1  Noise has been detected.
#      inform = 2  Noise has not been detected; h is too small.
#                  Try 100*h for the next value of h.
#      inform = 3  Noise has not been detected; h is too large.
#                  Try h/100 for the next value of h.
#
#     Argonne National Laboratory
#     Jorge More' and Stefan Wild. November 2009.

def ECNoise(nf, fval):
    level = np.zeros((nf-1))
    dsgn  = np.zeros((nf-1))
    fnoise = 0.0
    gamma = 1.0 # = gamma(0)

    # Compute the range of function values.
    fmin = np.amin(fval)
    fmax = np.amax(fval)
    if (fmax-fmin)/max(fabs(fmax), fabs(fmin)) > .1:
        inform = 3
        return fnoise, level, inform

    # Construct the difference table.
    for j in range(nf-1):
        for i in range(nf-j):
            fval[i] = fval[i+1] - fval[i]

        # h is too small only when half the function values are equal.
        if (j==0 && sum([fval[k] == 0 for k in range(nf - 1)]) >= nf/2):
            inform = 2
            return fnoise, level, inform

        gamma = 0.5*((j+1)/(2*(j+1)-1))*gamma

        # Compute the estimates for the noise level.
        level[j] = sqrt(gamma*np.mean(np.square(fval[0:nf-j])))

        # Determine differences in sign.
        emin = np.amin(fval[0:nf-j])
        emax = np.amax(fval[0:nf-j])
        if (emin*emax < 0.0):
            dsgn[j] = 1

    # Determine the noise level.
    for k in range(nf-3):
        emin = np.amin(level[k:k+2))
        emax = np.amax(level(k:k+2))
        if (emax<=4*emin && dsgn(k))
            fnoise = level(k)
            inform = 1
            return
        end
    end

    # If noise not detected then h is too large.
    inform = 3

return
