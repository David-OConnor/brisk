import math

import numba
import numpy as np

e = math.e

jit = numba.jit(nopython=True)


@jit
def sum_(data):
    sum__ = 0.
    for i in range(data.size):
        sum__ += data[i]
    return sum__


@jit
def mean(data):
    """Similar to numpy.mean."""
    sum__ = sum_(data)
    return sum__ / data.size


@jit
def var(data):
    """Variance test, similar to numpy.var for a one-axis array."""
    M = data.size
    ddof = 0  # ddof is set with a kwarg in numpy.var

    # The closer K is to the mean, the more accurate the results, but
    # anything in the sample will do.
    K = data[0]

    sum__ = 0.
    sum_sqr = 0.
    for i in range(M):
        sum__ += data[i] - K
        sum_sqr += (data[i] - K) ** 2
    return (sum_sqr - sum__**2 / (i+1 - ddof)) / (i+1 - ddof)


@numba.jit
def cov(m, y):
    """Covariance estimation, similar to numpy.cov."""
    # reference /site-packages/numpy/lib/function_base.py/cov
    M = m.size
    ddof = 1

    mean_m = mean(m)
    mean_y = mean(y)

    X = np.empty((2, M), dtype=np.float)
    for i in range(M):
        X[0, i] = m[i] - mean_m
        X[1, i] = y[i] - mean_y

    result = np.zeros((2, 2), dtype=np.float)
    for i in range(M):
        result[0, 0] += X[0, i] * X[0, i]
        result[0, 1] += X[0, i] * X[1, i]
        result[1, 0] = result[0, 1]
        result[1, 1] += X[1, i] * X[1, i]

    result[0][0] /= M - ddof
    result[0][1] /= M - ddof
    result[1][0] /= M - ddof
    result[1][1] /= M - ddof

    return result



# todo WIP
def matrix_mult(data1, data2):
    # for 2d arrays only atm.
    shape1 = data1.shape
    shape2 = data2.shape

    min1 = min(shape1)
    min2 = min(shape2)
    final_shape = (min1, min2)


    result = np.zeros((2, 2), dtype=np.float)
    for i in range(M):
        result[0][0] += data[0, i] * data[0, i]
        result[0][1] += data[0, i] * data[1, i]
        result[1][0] += data[1, i] * data[0, i]
        result[1][1] += data[1, i] * data[1, i]

    return result




@jit
def std(data):
    """Standard deviation, similar to numpy.std."""
    return var(data) ** .5


@jit
def corr(data1, data2):
    """Pearson correlation test, similar to scipy.stats.pearsonr."""
    M = data1.size

    sum1 = 0.
    sum2 = 0.
    for i in range(M):
        sum1 += data1[i]
        sum2 += data2[i]
    mean1 = sum1 / M
    mean2 = sum2 / M

    var_sum1 = 0.
    var_sum2 = 0.
    cross_sum = 0.
    for i in range(M):
        var_sum1 += (data1[i] - mean1) ** 2
        var_sum2 += (data2[i] - mean2) ** 2
        cross_sum += (data1[i] * data2[i])

    std1 = (var_sum1 / M) ** .5
    std2 = (var_sum2 / M) ** .5
    cross_mean = cross_sum / M

    return (cross_mean - mean1 * mean2) / (std1 * std2)


@jit
def bisect(a, x):
    """Similar to bisect.bisect() or bisect.bisect_right(), from the built-in library."""
    M = a.size
    for i in range(M):
        if a[i] > x:
            return i
    return M


@jit
def bisect_left(a, x):
    """Similar to bisect.bisect_left(), from the built-in library."""
    M = a.size
    for i in range(M):
        if a[i] >= x:
            return i
    return M


# @jit  # nopython is failing with np.empty. Currently slower than np.interp
# for small x sizes, but faster than the non-numba version. Haven't tested speed
# with large xp and fps.
@numba.jit
def interp(x, xp, fp):
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
def interp_one(x, xp, fp):
    """Similar to numpy.interp, if x is a single value."""
    i = bisect(xp, x)

    # These edge return values are set with left= and right= in np.interp.
    if i == 0:
        return fp[0]
    elif i == xp.size:
       return fp[-1]

    interp_port = (x - xp[i-1]) / (xp[i] - xp[i-1])

    return fp[i-1] + (interp_port * (fp[i] - fp[i-1]))


# # @jit
# def log(data, base):
#     """Logarithm. Similar to math.log. For natural logarithm, use math.e,
#      or quick.e for base."""
#     M = data.size
#     result = np.empty(M, dtype=np.float)
#     for i in range(M):
#         n = 1000.0
#         result[i] = n * ((data[i] ** (1/n)) - 1)
#
#     return result
#
#
# # todo wip
# # @jit
# def log_one(x, base):
#     """Natural Logarithm. Similar to math.log, one argument."""
#     n = 100.0
#     return n * ((x ** (1/n)) - 1)


# todo currently slower than numpy implementation.
@numba.jit
def nonzero(data):
    M = data.size

    result_i = 0  # index, and also serves as size for new array.
    result = np.empty(M, dtype=np.int)

    for i in range(M):
        if data[i] != 0:
            result[result_i] = i
            result_i += 1

    return result

    # result_trimmed = np.empty(result_i, dtype=np.int)
    # for i in range(result_i):
    #     result_trimmed[i] = result[i]
    #
    # return result_trimmed