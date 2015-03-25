import math

import numba
import numpy as np



# todo add separate functions that accept additional arguments, like ddof.


e = math.e

#note: '@numba.jit" is used instead of '@jit' when creating an array within the
# func.  nopython seems to not like np.empty.
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
    sum_sq = 0.

    for i in range(M):
        sum__ += data[i] - K
        sum_sq += (data[i] - K) ** 2

    return (sum_sq - sum__**2 / (i+1 - ddof)) / (i+1 - ddof)


@jit
def cov(m, y):
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



# todo Slower than np atm.
@numba.jit
def dot(x, y):
    # todo implement for 1d inputs

    # for 2d arrays only atm.
    size1 = x.shape[0]
    size2 = x.shape[1]

    result = np.zeros((size1, size1), dtype=np.float)

    for i1 in range(size1):
        for i2 in range(size1):
            for j in range(size2):
                result[i1, i2] += x[i1, j] * y[j, i2]

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


# todo WIP
@jit
def argmax(a):
    """Similar to numpy.argmax, with no axis argument provided."""

    # todo flatten
    # x = a.flatten() # todo you probably need to hand-flatten it
    # todo try your flatten function if you can make it work,
    # but is prob better to just incorporate it here directly to avoid
    # a temp array.
    shape = a.shape
    shape_size = len(shape)

    x = a
    max_ = x[0]
    max_i = 0
    for i in range(1, x.size):
        if x[i] > max_:
            max_ = x[i]
            max_i = i

    return max_i


# todo wip
@numba.jit
def flatten(data):
    shape = data.shape
    ndim = data.ndim

    result = np.empty(data.size, dtype=np.float)

    count = 0
    for i in range(ndim):
        pass


    return result


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



def argmax_axis(a, axis):
    """Similar to numpy.argmax, with the axis argument provided."""


    # flatten - candidate for separate function?



#todo WIP
#@numba.jit
def detrend(data, axis, type_, bp):
    """Similar to scipiy.signal.detrend."""

    # bp stands for break points.

    if type_ == 'constant' or type_ == 'c':
        ret = data - np.expand_dims(mean(data, axis), axis)
        return ret

    else:
        dshape = data.shape
        N = dshape[axis]

        if bp == 0:
            bp = np.array([0, N], dtype=np.int)
        else:
            bp_size = bp.size
            # bp = np.array([0, bp, N], dtype=np.int)
            bp_new = np.empty(bp_size + 2, dtype=np.int)
            bp[0] = 0
            bp[-1] = N
            for i in range(bp_size):
                bp_new[i+1] = bp[i]

        # Restructure data so that axis is along first dimension and
        #  all other dimensions are collapsed into second dimension
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk


      # #todo fix thi smess
      #   axis = 1
      #   rnk = 11
      #
      #   newdims_size = 1 + axis + (rnk - axis - 1)
      #   newdims = np.empty(newdims_size, dtype=np.int)
      #   newdims[0] = axis
      #   for i in range(axis):
      #       newdims[i+1] = i
      #   for i in range(axis+1, (rnk-axis+3)):
      #       newdims[i] = i


        newdims = np.r_[axis, 0:axis, axis + 1:rnk]

        # print(axis, 'axis', rnk, 'rnk')
        # print(newdims, 'new')
        # print(newdims2, 'new2')
        #

        newdata = np.reshape(np.transpose(data, tuple(newdims)),
                          (N, np.prod(dshape, axis=0) // N))

        newdata = newdata.copy()  # make sure we have a copy
        if newdata.dtype.char not in 'dfDF':
            newdata = newdata.astype(np.float)


        # Find leastsq fit and remove it for each piece
        for m in range(bp.size - 1):
            Npts = bp[m + 1] - bp[m]
            A = np.ones((Npts, 2), np.float)
            A[:, 0] = np.cast[np.float](np.arange(1, Npts + 1) * 1.0 / Npts)
            sl = slice(bp[m], bp[m + 1])
            coef, resids, rank, s = np.linalg.lstsq(A, newdata[sl])
            newdata[sl] = newdata[sl] - np.dot(A, coef)


        # Put data back in original shape.
        tdshape = np.take(dshape, newdims, 0)
        ret = np.reshape(newdata, tuple(tdshape))
        vals = list(range(1, rnk))
        olddims = vals[:axis] + [0] + vals[axis:]
        ret = np.transpose(ret, tuple(olddims))
        return ret





