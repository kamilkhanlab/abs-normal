# abs-normal
Tools in Julia for solving equations and optimization problems involving piecewise linear/affine functions expressed in *abs-normal form*, presented as a module `AbsNormal`. This makes use of new complementarity formulations that will be presented at the [AD2024 conference](https://www.autodiff.org/ad24/).

**Note**: Some parts of this module require the solver [BARON](https://github.com/jump-dev/BARON.jl), which in turn requires the user to have a valid BARON license. For users without a BARON license, the "non-BARON" parts of our source code and examples are presented as files ending with `WithoutBARON.jl`. The full source code and examples are presented as files ending with `WithBARON.jl`.

## Background

As shown by [Griewank et al. (2015)](https://doi.org/10.1016/j.laa.2014.12.017), any continuous piecewise-affine function $\mathbf{f}:R^n\to R^m$ may be evaluated as part of the solution $(\mathbf{z}, \mathbf{f}(\mathbf{x}))$ of an equation system with the following format:

$$
\mathbf{z} = \mathbf{c} + \mathbf{Z}\mathbf{x} + \mathbf{L}|\mathbf{z}| 
$$

$$
\mathbf{f}(\mathbf{x}) = \mathbf{b} + \mathbf{J}\mathbf{x} + \mathbf{Y}|\mathbf{z}|,
$$

where the various coefficient matrices and vectors have appropriate dimension, and where $\mathbf{L}$ is strictly lower triangular.
This is an *abs-normal form* of $\mathbf{f}$, and may be constructed automatically by a variant of automatic differentiation (AD).

The `AbsNormal` module in this repository provides tools for minimizing or solving equations involving functions that have already been expressed in abs-normal form, based on new complementarity formulations by Zhang and Khan.
