import numba

jit = numba.jit(nopython=True)


@jit
def sum_(data):
    sum__ = 0.
    for i in range(data.size):
        sum__ += data[i]
    return sum__


@jit
def mean(data):
    sum__ = sum_(data)
    return sum__ / data.size


@jit
def var(data):
    M = data.size

    mean_ = mean(data)
    var_sum = 0.
    for i in range(M):
        var_sum += (data[i] - mean_) ** 2
    return var_sum / M


@jit
def std(data):
    return var(data) ** .5


@jit
def corr(data1, data2):
    """Optimized pearson correlation test, similar to scipy.stats.pearsonr."""
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