Quick: Applied Numba
==========================================

Optimized numerical computation using Continuum's Numba. Intended as a drop-in replacement
for numerical functions in numpy, scipy, or builtins. Provides strong performance boosts.

Numba website `<http://numba.pydata.org/>`_

Inputs use numpy arrays, not lists.
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
