Optimized numerical computations using Numba. Intended as a drop-in replacement
for numpy, scipy, or built-in functions. Provides strong performance boosts.

Inputs use numpy arrays, not lists.
Rough/early release - Open to suggestions and bug reports.

Included functions
------------------

- sum: Similar to builtin sum, or numpy.sum
- mean: Similar to numpy.mean
- var: Variance test, similar to numpy.var
- std: Standard deviation, similar to numpy.std
- corr: Pearson correlation test, ,imilar to scipy.stats.pearsonr
- bisect: Similar to builtin bisect
- bisect_left: Similar to builtin bisect_left
- interp: Similar to numpy.interp. x is an array.
- interp_one: Similar to numpy.interp. x is a single value.