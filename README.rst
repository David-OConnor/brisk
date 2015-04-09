Brisk: Applied Numba
====================

Optimized numerical computation using Continuum's Numba. Intended as a drop-in replacement
for numerical functions in numpy, scipy, or builtins. Provides strong performance boosts.

`Numba website <http://numba.pydata.org/>`_

Inputs use numpy arrays. Using other formats like lists, or pandas Dataframes
will adversely affect speed.
Rough/early release  - Open to suggestions and bug reports.

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

.. code-block:: python

    brisk.sum(data: numpy.array) -> float:

Inputs:
  - data: Input data.
Ouputs:
 -Sum of all values in data.


.. code-block:: python

    brisk.mean(data: numpy.array) -> float:

Inputs:
  - data: Input data.
Ouputs:
 -Mean of all values in data.


.. code-block:: python

    brisk.var(data: numpy.array) -> float:

Inputs:
  - data: Input data.
Ouputs:
 -Variance of data.


.. code-block:: python

    brisk.cov(m: numpy.array, y: numpy.array) -> float:

Inputs:
  - m and y: two data sets to find the covariance of. Must be the same size.

Ouputs:
 -Covariance of m and y.


.. code-block:: python

    brisk.std(data: numpy.array) -> float:

Inputs:
  - data: Input data.

Ouputs: Standard deviation of data.


.. code-block:: python

    brisk.corr(x: numpy.array, y: numpy.array) -> float:

Inputs:
 - x and y: two numpy.arary data sets to find the pearson correlation of. Must be the same size.

Ouputs:
 - Pearson correlation of m and y.


.. code-block:: python

    brisk.std(data: numpy.array) -> float:

Inputs:
 - data: a numpy.array.

Ouputs:
- Standard deviation of data.


.. code-block:: python

    brisk.bisect(a: float, x: numpy.array) -> int:

Inputs:
 - a: Value to be inserted.
 - x: numpy array to insert a into.

Ouputs:
 - The insertion point for x in a to maintain sorted order.


.. code-block:: python

    brisk.bisect_left(a: float, x: numpy.array) -> int:

Inputs:
 - a: Value to be inserted.
 - x: numpy array to insert a into.

Ouputs:
 - The insertion point for x in a to maintain sorted order.


.. code-block:: python

    brisk.interp(x: numpy.array, xp: numpy.array, fp: numpy.array) -> numpy.array:

Inputs:
 - x: x coordinates of the interpolated values.
 - xp: x coordinates of the data points.
 - yp: y coordinates of the data points. Same size as xp.

Ouputs:
 - The interpolated values.


.. code-block:: python

    brisk.interp_one(x: float, xp: numpy.array, fp: numpy.array) -> float:

Inputs:
 - x: x coordinates of the interpolated value.
 - xp: x coordinates of the data points.
 - yp: y coordinates of the data points. Same size as xp.

Ouputs:
 - The interpolated value.

.. code-block:: python

    brisk.detrend(data: numpy.array, type_: str) -> numpy.array:

Inputs:
- data: The data to detrend
- type: Use 'c' or 'constant' for constant detrending. Use 'l' or 'linear' for linear detrending.

Ouputs: The detrended data.


.. code-block:: python

    brisk.ols(x: numpy.array, y: numpy.array) -> (float, float):

Inputs:
 - x: x values to run regression on.
 - y: y values to run regression on.

Ouputs:
 - A tuple of the resulting slope and intercept.


.. code-block:: python

    brisk.ols_single(y: numpy.array) -> (float, float):

Inputs:
 - y: y values to run regression on. x values are inferred to be a range from 0 to y.size.

Ouputs:
 - A tuple of the resulting slope and intercept.


.. code-block:: python

    brisk.lin_resids(x: numpy.array, y: numpy.array, slope: float, intercept: float) -> numpy.array:

Inputs:
 - x: x values regression was run on.
 - y: y values regression was run on.
 - slope: Regression slope.
 - intercept: Regression intercept.

Ouputs:
 - An array of the linear residuals.


.. code-block:: python

    brisk.lin_resids_single(x: numpy.array, slope: float, intercept: float) -> numpy.array:

Inputs:
 - y: y values regression was run on. x values are inferred to be a range from 0 to y.size.
 - slope: Regression slope.
 - intercept: Regression intercept.

Ouputs:
 - An array of the linear residuals.