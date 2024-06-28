# abs-normal
Tools in Julia for solving equations and optimization problems involving piecewise linear/affine functions expressed in "absolute normal form", presented as a module `AbsNormal`.

**Note**: Some parts of this module require the solver [BARON](https://github.com/jump-dev/BARON.jl), which in turn requires the user to have a valid BARON license. For users without a BARON license, the "non-BARON" parts of our source code and examples are presented as files ending with `WithoutBARON.jl`. The full source code and examples are presented as files ending with `WithBARON.jl`.
