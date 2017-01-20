from itertools import chain
from math import e, pi
from typing import Iterable

import numba
import numpy as np
from numpy import sqrt, cos, sin, log

π = pi



# todo add separate functions that accept additional arguments, like ddof.


# Note: '@numba.jit" is used instead of '@jit' when creating an array within the
# func.  nopython seems to not like np.empty.
jit = numba.jit(nopython=True)


@jit
def sum_(data: np.ndarray):
    """Similar to numpy.sum."""
    sum__ = 0.
    for i in range(data.size):
        sum__ += data[i]
    return sum__


@jit
def mean(data: np.ndarray):
    """Similar to numpy.mean."""
    sum__ = sum_(data)
    return sum__ / data.size


@jit
def var(data: np.ndarray):
    """Variance test, similar to numpy.var for a one-axis array."""
    M = data.size
    ddof = 0  # ddof is set with a kwarg in numpy.var

    # The closer K is to the mean, the more accurate the results, but
    # anything in the sample will do.
    K = data[0]

    sum__ = 0.
    sum_sq = 0.

    for i in range(M):
        sum__ += data[i] - K
        sum_sq += (data[i] - K) ** 2

    return (sum_sq - sum__**2 / (i+1 - ddof)) / (i+1 - ddof)


@jit
def cov(m: np.ndarray, y):
    """Covariance estimation, similar to numpy.cov. Returns only the covariance
    result as a float, instead of a 2x2 array."""
    M = m.size
    ddof = 1

    mean_m = mean(m)
    mean_y = mean(y)

    sum_sq = 0.
    for i in range(M):
        sum_sq += (m[i] - mean_m) * (y[i] - mean_y)

    return sum_sq / (M - ddof)


@jit
def std(data: np.ndarray):
    """Standard deviation, similar to numpy.std."""
    return var(data) ** .5


@jit
def corr(x: np.ndarray, y: np.ndarray):
    """Pearson correlation test, similar to scipy.stats.pearsonr."""
    M = x.size

    sum1 = 0.
    sum2 = 0.
    for i in range(M):
        sum1 += x[i]
        sum2 += y[i]
    mean1 = sum1 / M
    mean2 = sum2 / M

    var_sum1 = 0.
    var_sum2 = 0.
    cross_sum = 0.
    for i in range(M):
        var_sum1 += (x[i] - mean1) ** 2
        var_sum2 += (y[i] - mean2) ** 2
        cross_sum += (x[i] * y[i])

    std1 = (var_sum1 / M) ** .5
    std2 = (var_sum2 / M) ** .5
    cross_mean = cross_sum / M

    return (cross_mean - mean1 * mean2) / (std1 * std2)


@jit
def bisect(a: np.ndarray, x):
    """Similar to bisect.bisect() or bisect.bisect_right(), from the built-in library."""
    M = a.size
    for i in range(M):
        if a[i] > x:
            return i
    return M


@jit
def bisect_left(a: np.ndarray, x):
    """Similar to bisect.bisect_left(), from the built-in library."""
    M = a.size
    for i in range(M):
        if a[i] >= x:
            return i
    return M


# Currently slower than np.interp
# for small x sizes, but faster than the non-numba version. Haven't tested speed
# with large xp and fps.
@numba.jit
def interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray):
    """Similar to numpy.interp, if x is an array."""
    M = x.size

    result = np.empty(M, dtype=np.float)
    for i in range(M):
        i_r = bisect(xp, x[i])

        # These edge return values are set with left= and right= in np.interp.
        if i_r == 0:
            result[i] = fp[0]
            continue
        elif i_r == xp.size:
            result[i] = fp[-1]
            continue

        interp_port = (x[i] - xp[i_r-1]) / (xp[i_r] - xp[i_r-1])

        result[i] = fp[i_r-1] + (interp_port * (fp[i_r] - fp[i_r-1]))

    return result


@jit
def interp_one(x: np.ndarray, xp: np.ndarray, fp: np.ndarray):
    """Similar to numpy.interp, if x is a single value."""
    i = bisect(xp, x)

    # These edge return values are set with left= and right= in np.interp.
    if i == 0:
        return fp[0]
    elif i == xp.size:
        return fp[-1]

    interp_port = (x - xp[i-1]) / (xp[i] - xp[i-1])

    return fp[i-1] + (interp_port * (fp[i] - fp[i-1]))



@numba.jit
def detrend(data: np.ndarray, type_: str):
    """Similar to scipiy.signal.detrend. Currently for 1d arrays only."""
    M = data.size
    result = np.empty(M, dtype=np.float)

    if type_ == 'constant' or type_ == 'c':
        mean_ = mean(data)
        for i in range(M):
            result[i] = data[i] - mean_
        return result

    elif type_ == 'linear' or type_ == 'l':

        slope, intercept = ols_single(data)
        for i in range(M):
            result[i] = data[i] - (slope * i + intercept)
        return result

    else:
        raise AttributeError

# todo consider a least absolute deviations (lad) function.


@jit
def ols(x: np.ndarray, y: np.ndarray):
    """Simple OLS for two data sets."""
    M = x.size

    x_sum = 0.
    y_sum = 0.
    x_sq_sum = 0.
    x_y_sum = 0.

    for i in range(M):
        x_sum += x[i]
        y_sum += y[i]
        x_sq_sum += x[i] ** 2
        x_y_sum += x[i] * y[i]

    slope = (M * x_y_sum - x_sum * y_sum) / (M * x_sq_sum - x_sum**2)
    intercept = (y_sum - slope * x_sum) / M

    return slope, intercept


@numba.jit
def ols_single(y: np.ndarray):
    """Simple OLS for one data set."""
    x = np.arange(y.size)
    return ols(x, y)


@numba.jit
def lin_resids(x: np.ndarray, y: np.ndarray, slope: float, intercept: float):
    M = x.size
    result = np.empty(M, dtype=np.float)

    for i in range(M):
        result[i] = y[i] - (slope * x[i] + intercept)

    return result


@numba.jit
def lin_resids_single(y: np.ndarray, slope: float, intercept: float):
    M = y.size
    result = np.empty(M, dtype=np.float)

    for i in range(M):
        result[i] = y[i] - (slope * i + intercept)

    return result


@jit
def vdot(a: np.ndarray, b: np.ndarray):  # todo slower than numpy for large arrays
    """Take the dot product of two vectors.  Similar to np.vdot.
    a and b must have shape (n,)"""
    assert a.shape == b.shape, "both vectors must be the same size."
    M = a.size
    result = 0

    for i in range(M):
        result += a[i] * b[i]
    return result


@jit
def add_elwise(a: Iterable, b: Iterable):
    """Add two iterables element-wise."""
    result = []
    for i in zip(a, b):
        result.append(i[0] + i[1])
    return result


@jit
def sub_elwise(a: Iterable, b: Iterable):
    """Subtract the second iterable from the first, element-wise."""
    result = []
    for i in zip(a, b):
        result.append(i[0] - i[1])
    return result


@jit
def div_elwise(items: Iterable, value: float):
    """Divide all values in an iterable by a constant"""
    result = []
    for i in items:
        result.append(i / value)
    return result


@jit
def mult_elwise(items: Iterable, value: float):
    """Multiply all values in an iterable by a constant."""
    result = []
    for i in items:
        result.append(i * value)
    return result



##### WIP / undocumented functions below.


PP = np.array([
    7.96936729297347051624E-4,
    8.28352392107440799803E-2,
    1.23953371646414299388E0,
    5.44725003058768775090E0,
    8.74716500199817011941E0,
    5.30324038235394892183E0,
    9.99999999999999997821E-1], 'd')

PQ = np.array([
    9.24408810558863637013E-4,
    8.56288474354474431428E-2,
    1.25352743901058953537E0,
    5.47097740330417105182E0,
    8.76190883237069594232E0,
    5.30605288235394617618E0,
    1.00000000000000000218E0], 'd')

DR1 = 5.783185962946784521175995758455807035071
DR2 = 30.47126234366208639907816317502275584842

RP = np.array([
    -4.79443220978201773821E9,
    1.95617491946556577543E12,
    -2.49248344360967716204E14,
    9.70862251047306323952E15], 'd')

RQ = np.array([
    # 1.00000000000000000000E0,
    4.99563147152651017219E2,
    1.73785401676374683123E5,
    4.84409658339962045305E7,
    1.11855537045356834862E10,
    2.11277520115489217587E12,
    3.10518229857422583814E14,
    3.18121955943204943306E16,
    1.71086294081043136091E18], 'd')

QP = np.array([
    -1.13663838898469149931E-2,
    -1.28252718670509318512E0,
    -1.95539544257735972385E1,
    -9.32060152123768231369E1,
    -1.77681167980488050595E2,
    -1.47077505154951170175E2,
    -5.14105326766599330220E1,
    -6.05014350600728481186E0], 'd')

QQ = np.array([
    # 1.00000000000000000000E0,
    6.43178256118178023184E1,
    8.56430025976980587198E2,
    3.88240183605401609683E3,
    7.24046774195652478189E3,
    5.93072701187316984827E3,
    2.06209331660327847417E3,
    2.42005740240291393179E2], 'd')



@jit
def polevl(x, coef):
    """Taken from http://numba.pydata.org/numba-doc/0.12.2/examples.html"""
    N = len(coef)
    ans = coef[0]
    i = 1
    while i < N:
        ans = ans * x + coef[i]
        i += 1
    return ans


@jit
def p1evl(x, coef):
    """Taken from http://numba.pydata.org/numba-doc/0.12.2/examples.html"""
    N = len(coef)
    ans = x + coef[0]
    i = 1
    while i < N:
        ans = ans * x + coef[i]
        i += 1
    return ans


@jit
def j0(x):
    """Taken from http://numba.pydata.org/numba-doc/0.12.2/examples.html"""
    # Seems slower than scipy.special.j0 (Which it's a reimplentation of), but
    # this allows you to make functions that work with nopython=True.
    if x < 0:
        x = -x

    if x <= 5.0:
        z = x * x

        if x < 1.0e-5:
            return 1.0 - z / 4.0

        p = (z - DR1) * (z - DR2)
        p = p * polevl(z, RP) / p1evl(z, RQ)
        return p

    w = 5.0 / x
    q = 25.0 / x**2
    p = polevl(q, PP) / polevl(q, PQ)
    q = polevl(q, QP) / p1evl(q, QQ)
    xn = x - π/4
    p = p * cos(xn) - w * q * sin(xn)
    return p * sqrt(2/π) / sqrt(x)


# def y0_scipy(x):
#     """
#                                                          y0() 2  /
#     Bessel function of second kind, order zero  /
#
#     Rational approximation coefficients YP[], YQ[] are used here.
#     The function computed is  y0(x)  -  2  log(x)  j0(x) / NPY_PI,
#     whose value at x = 0 is  2  ( log(0.5) + EUL ) / NPY_PI
#     = 0.073804295108687225.
#
#     #define NPY_PI_4 .78539816339744830962
#     #define SQ2OPI .79788456080286535588
#     """
#
#     if x <= 5.0:
#         if x == 0.0:
#             mtherr("y0", SING)
#             return -np.inf
#         elif x < 0.0:
#             mtherr("y0", DOMAIN)
#             return np.nan
#
#         z = x**2
#         w = polevl(z, YP, 7) / p1evl(z, YQ, 7)
#         w += 2*π * log(x) * j0(x)
#         return w
#
#     w = 5.0 / x
#     z = 25.0 / (x * x)
#     p = polevl(z, PP) / polevl(z, PQ)
#     q = polevl(z, QP) / p1evl(z, QQ)
#     xn = x - pi/4
#     p = p * sin(xn) + w * q * cos(xn)
#     return p * sqrt(2/pi) / sqrt(x)


# Undocumented; reduced accuracy is probably not worth it.
@jit
def cov_fast(m, y):
    """Faster covariance function that doesn't need pre-calculate mean. May
    be less accurate."""
    M = m.size
    ddof = 1  # ddof is set with a kwarg in numpy.var

    # The closer K is to the mean, the more accurate the results, but
    # anything in the sample will do.
    K1 = m[0]
    K2 = y[0]

    sum1 = 0.
    sum2 = 0.
    sum_sq = 0.

    for i in range(M):
        sum1 += m[i] - K1
        sum2 += y[i] - K2
        sum_sq += (m[i] - K1) * (y[i] - K2)

    return (sum_sq - sum1 * sum2 / (M - ddof)) / (M - ddof)


# Undocumented.
# todo Slower than np atm.
@numba.jit
def dot(x, y):
    # todo implement for 1d inputs

    # for 2d arrays only atm.
    size1 = x.shape[0]
    size2 = y.shape[1]

    result = np.zeros((size1, size2), dtype=np.float)

    for i1 in range(size1):
        for i2 in range(size2):
            for j in range(size1):
                result[i1, i2] += x[i1, j] * y[j, i2]

    return result


# Undocumented.
# todo WIP
@numba.jit
def argmax(a):
    """Similar to numpy.argmax, with no axis argument provided."""


    # TODO CAN'T find a way to flatten properly. If using a 1d array, this is a bit fater
    # todo than numpy. If I use numpy's flatten, it's slower. Don't know how to make my own flatten.
    x = a


    max_ = x[0]
    max_i = 0
    for i in range(1, x.size):
        if x[i] > max_:
            max_ = x[i]
            max_i = i

    return max_i


# Undocumented.
# todo WIP
@numba.jit
def argmax_axis(a):
    """Similar to numpy.argmax, with an axis argument provided."""
    pass




# Undocumented.
# todo temp
def argmax_3d(data):
    shape = data.shape
    ndim = data.ndim

    max_ = data[0, 0, 0]
    max_i = 0

    count = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if data[i, j, k] > max_:
                    max_ = data[i, j, k]
                    max_i = count

            count += 1

    return max_i




# Undocumented.
# todo wip
# @numba.jit
def flatten(data):
    shape = data.shape
    ndim = data.ndim

    result = np.empty(data.size, dtype=np.float)
    count = 0



    for o2 in range(ndim):
        for o in range(ndim):
            indices =[]
            for i in range(shape[o]):
                indices.append(i)


        print(indices)
        # result[count] = data[indices]
        count += 1


    return result


# todo concept test
# @numba.jit
def remove_axis_r(data):
    shape = data.shape

    result = np.empty(tuple(chain(shape[:-2], (shape[-1] * shape[-2],))), dtype=np.float)

    for i in range(shape[0]):
        print(i, shape[1])
        result[i] = data[i, :shape[1]]

    return result


# todo concept test
# @numba.jit
def remove_axis_l(data):
    shape = data.shape

    result = np.empty(tuple(chain((shape[0] * shape[1],), shape[2:])), dtype=np.float)

    for i in range(shape[0]):
        start = 0
        end = shape[-1]
        for j in range(shape[i]):
            result[start: end] = data[i, j]
            start = end
            end = end + shape[-1]

    return result


# Undocumented.
# todo slower than np.flatten() atm.
@numba.jit
def flatten_3d(data):
    shape = data.shape

    result = np.empty(data.size, dtype=np.float)

    result_i = 0


    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                result[result_i] = data[i, j, k]
                result_i += 1

    return result

# todo compile some of the single/one funcs if you can without having a speed hit
# todo or awkward calls.

# # todo WIP
# @numba.jit
# def sum_batch(data, axis):
#     """Similar to numpy.sum."""
#     shape = data.shape
#
#     new_shape = []
#     for i in range(len(shape)):
#         if i != axis:
#             new_shape.append(shape[i])
#
#     result = np.empty(new_shape, dtype=np.float)
#
#     sum__ = 0.
#     for j in range(len(shape) - 1):
#         for i in range(shape[axis]):
#             sum__ += data[i, j, ]
#             0, 0, 0 + 1, 0, 0
#             0, 0, 1 + 1, 0, 1
#             0, 0, 2 + 1, 0, 2
#             0, 1, 2 + 1, 1, 2
#
#
#
#     for i in range(data.size):
#         sum__ += data[i]
#     return sum__


# Undocumented.
#todo WIP
@numba.jit
def dft(x, inverse):
    N = x.size
    inv = -1 if not inverse else 1
    X = np.zeros(N, dtype=np.complex128)

    for k in range(N):
        for n in range(N):
            X[k] += x[n] * e**(inv * 2j * π * k * n / N)
        if inverse:
            X[k] /= N
    return X


# Undocumented.
#todo WIP
@numba.jit
def fft(x, inverse):
    N = x.size
    inv = -1 if not inverse else 1

    if N % 2:
        raise ValueError

    if N <= 32:
        return dft(x, inverse)

    x_e = np.empty(N/2, dtype=np.complex128)
    x_o = np.empty(N/2, dtype=np.complex128)

    X_e = fft(x[::2], inverse)
    X_o = fft(x[1::2], inverse)

    X = np.empty(N, dtype=np.complex128)

    M = N // 2

    count = 0
    for k in range(M):
        X[count] = X_e[k] + X_o[k] * e ** (inv * 2j * π * k / N)
        count += 1
    for k in range(M, N):
        X[count] = X_e[k-M] - X_o[k-M] * e ** (inv * 2j * π * (k-M) / N)
        count += 1

    if inverse:
        inverse_result = np.empty(N, dtype=np.complex128)
        for i in range(N):
            inverse_result[i] = X[i] / 2
        return inverse_result

    return X
