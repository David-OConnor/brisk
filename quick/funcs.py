import numba
import numpy as np

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


#todo implement array input, adn possibly single-pass version.
@jit
def var(data):
    """Variance test, similar to numpy.var for one-axis array."""
    M = data.size

    mean_ = mean(data)
    var_sum = 0.
    for i in range(M):
        var_sum += (data[i] - mean_) ** 2
    return var_sum / M


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

# todo wip
@jit
def log(data):
    M = data.size
    s_new = np.empty(M, dtype=np.float)
    for i in range(M):
        s_new[i] = math.log(data[i])

    return s_new