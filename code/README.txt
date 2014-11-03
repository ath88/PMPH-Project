To compile and run, go to one of the *Impl folder and type
"make run_{large,medium,small}".

In addition to OrigImpl as described below, these folders contain different versions of the implementation:
    -- TimedImpl: Same as the original implementation, but with profiling added.
    -- OpenMPImpl: Optimized with OpenMP.
    -- CUDAImpl: Optimized with CUDA.

Folder `OrigImpl' contains the original implementation:
    -- `ProjectMain.cpp'   contains the main function
    -- `ProjCoreOrig.cpp'  contains the core functions 
                                (to parallelize)
    -- `ProjHelperFun.cpp' contains the functions that compute
                                the input parameters, and 
                                (can be parallelize as well)

Folder `include' contains
    -- `ParserC.h'     implements a simple parser
    -- `ParseInput.h'  reads the input/output data
                        and provides validation.
    -- `OpenmpUtil.h'  some OpenMP-related helpers.        
    -- `Constants.h'   currently only sets up REAL
                        to either double or float
                        based on the compile-time
                        parameter WITH_FLOATS.

    -- `CudaUtilProj.cu.h' provides stubs for calling
                        transposition and inclusive 
                        (segmented) scan.
    -- `TestCudaUtil.cu'  a simple tester for 
                        transposition and scan.

