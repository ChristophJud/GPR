INSTALL(FILES 
	GaussianProcess.h 
	Kernel.h
	KernelFactory.h
	KernelUtils.h
	MatrixIO.h
	Likelihood.h
	LikelihoodUtils.h
	Prior.h
	PriorUtils.h
	LAPACKUtils.h
	SparseGaussianProcess.h
DESTINATION
	include
PERMISSIONS
	OWNER_READ GROUP_READ WORLD_READ
)
