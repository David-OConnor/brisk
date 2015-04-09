Brisk: Applied Numba
====================

Optimized numerical computation using Continuum's Numba. Intended as a drop-in replacement
for numerical functions in numpy, scipy, or builtins. Provides strong performance boosts.

`Numba website <http://numba.pydata.org/>`_

Inputs use numpy arrays. Using other formats like lists, or pandas Dataframes
will adversely affect speed.
Rough/early release - Open to suggestions and bug reports.

Included functions
------------------

- sum: Similar to builtin sum, or numpy.sum
- mean: Similar to numpy.mean
- var: Variance test, similar to numpy.var
- cov: Covariance estimation, similar to numpy.cov
- std: Standard deviation, similar to numpy.std
- corr: Pearson correlation test, similar to scipy.stats.pearsonr
- bisect: Similar to standard library bisect.bisect
- bisect_left: Similar to standard library builtin.bisect_left
- interp: Linear interpoliation, similar to numpy.interp. x is an array.
- interp_one: Linear interpolation, similar to numpy.interp. x is a single value.
- detrend: Similar to scipy.signal.detrend. Linear or constant trend.
- ols: Simple Ordinary Least Squares regression for two data sets.
- ols_single: Simple Ordinary Least Squares regression for one data set.
- lin_resids: Residuals calculation from a linear regression with two data sets
- lin_resids_single: Residuals calculation from a linear regression with one data set.


Basic documentation
-------------------

brisk.sum(data: numpy.array) -> float:
Inputs: data: Input data.
Ouputs: Sum of all values in data.


brisk.mean(data: numpy.array) -> float:
Inputs: data: Input data.
Ouputs: Mean of all values in data.


brisk.var(data: numpy.array) -> float:
Inputs: data: Input data.
Ouputs: Variance of data.


brisk.cov(m: numpy.array, y: numpy.array) -> float:
Inputs: m and y: two data sets to find the covariance of.
Must be the same size.
Ouputs: Covariance of m and y.


brisk.std(data: numpy.array) -> float:
Inputs: data: Input data.
Ouputs: Standard deviation of data.


brisk.corr(x: numpy.array, y: numpy.array) -> float:
Inputs: x and y: two numpy.arary data sets to find the pearson correlation of.
Must be the same size.
Ouputs: Pearson correlation of m and y.


brisk.std(data: numpy.array) -> float:
Inputs: data: a numpy.array.
Ouputs: Standard deviation of data.


brisk.bisect(a: float, x: np.array) -> int:
Inputs: a: Value to be inserted.
        x: numpy array to insert a into.
Ouputs: The insertion point for x in a to maintain sorted order.


brisk.bisect_left(a: float, x: np.array) -> int:
Inputs: a: Value to be inserted.
        x: numpy array to insert a into.
Ouputs: The insertion point for x in a to maintain sorted order.


brisk.interp(x: np.array, xp: np.array, fp: np.array) -> np.array:
Inputs: x: x coordinates of the interpolated values.
        xp: x coordinates of the data points.
        yp: y coordinates of the data points. Same size as xp.
Ouputs: The interpolated values.


brisk.interp_one(x: float, xp: np.array, fp: np.array) -> float:
Inputs: x: x coordinates of the interpolated value.
        xp: x coordinates of the data points.
        yp: y coordinates of the data points. Same size as xp.
Ouputs: The interpolated value.


brisk.detrend(data: np.array, type_: str) -> np.array:
Inputs: data: The data to detrend
        type_:Use 'c' or 'constant' for constant detrending.
        Use 'l' or 'linear' for linear detrending.
Ouputs: The detrended data.


brisk.ols(x: np.array, y: np.array) -> (float, float):
Inputs: x: x values to run regression on.
        y: y values to run regression on.
Ouputs: A tuple of the resulting slope and intercept.


brisk.ols_single(y: np.array) -> (float, float):
Inputs: y: y values to run regression on. x values are inferred to be a range
        from 0 to y.size.
Ouputs: A tuple of the resulting slope and intercept.


brisk.lin_resids(x: np.array, y: np.array, slope: float, intercept: float) -> np.array:
Inputs: x: x values regression was run on.
        y: y values regression was run on.
        slope: Regression slope.
        intercept: Regression intercept.
Ouputs: An array of the linear residuals.


brisk.lin_resids_single(x: np.array, slope: float, intercept: float) -> np.array:
Inputs: y: y values regression was run on. x values are inferred to be a range
        from 0 to y.size.
        slope: Regression slope.
        intercept: Regression intercept.
Ouputs: An array of the linear residuals.