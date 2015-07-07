SET(gp_lib_src GaussianProcessTest.cpp )
SET(io_lib_src IOTest.cpp )
SET(pk_lib_src PeriodicKernelTest.cpp )
SET(csk_lib_src SumKernelTest.cpp )
SET(cpk_lib_src ProductKernelTest.cpp )
SET(sc_lib_src ScopeTest.cpp )
SET(pp_lib_src PosteriorProcessTest.cpp )
SET(im_lib_src InversionMethodsTest.cpp )
SET(rq_lib_src RationalQuadraticKernelTest.cpp )
SET(hq_lib_src HighlyGeneralKernelTest.cpp )
SET(kd_lib_src KernelDerivativeTest.cpp )
SET(gl_lib_src GaussianLikelihoodTest.cpp)
SET(lp_lib_src LAPACKTest.cpp)
SET(si_lib_src SparseInferenceTest.cpp)


ADD_EXECUTABLE(gpTest ${gp_lib_src})
ADD_EXECUTABLE(ioTest ${io_lib_src})
ADD_EXECUTABLE(pkTest ${pk_lib_src})
ADD_EXECUTABLE(cskTest ${csk_lib_src})
ADD_EXECUTABLE(cpkTest ${cpk_lib_src})
ADD_EXECUTABLE(scTest ${sc_lib_src})
ADD_EXECUTABLE(ppTest ${pp_lib_src})
ADD_EXECUTABLE(imTest ${im_lib_src})
ADD_EXECUTABLE(rqTest ${rq_lib_src})
ADD_EXECUTABLE(hqTest ${hq_lib_src})
ADD_EXECUTABLE(kdTest ${kd_lib_src})
ADD_EXECUTABLE(glTest ${gl_lib_src})
ADD_EXECUTABLE(lpTest ${lp_lib_src})
ADD_EXECUTABLE(siTest ${si_lib_src})

TARGET_LINK_LIBRARIES( gpTest gplib)
TARGET_LINK_LIBRARIES( ioTest gplib)
TARGET_LINK_LIBRARIES( pkTest gplib)
TARGET_LINK_LIBRARIES( cskTest gplib)
TARGET_LINK_LIBRARIES( cpkTest gplib)
TARGET_LINK_LIBRARIES( scTest gplib)
TARGET_LINK_LIBRARIES( ppTest gplib)
TARGET_LINK_LIBRARIES( imTest gplib)
TARGET_LINK_LIBRARIES( rqTest gplib)
TARGET_LINK_LIBRARIES( hqTest gplib)
TARGET_LINK_LIBRARIES( kdTest gplib)
TARGET_LINK_LIBRARIES( glTest gplib)
TARGET_LINK_LIBRARIES( lpTest gplib)
TARGET_LINK_LIBRARIES( siTest gplib)

SET_TARGET_PROPERTIES(gpTest PROPERTIES OUTPUT_NAME gpTest)
SET_TARGET_PROPERTIES(ioTest PROPERTIES OUTPUT_NAME ioTest)
SET_TARGET_PROPERTIES(pkTest PROPERTIES OUTPUT_NAME pkTest)
SET_TARGET_PROPERTIES(cskTest PROPERTIES OUTPUT_NAME cskTest)
SET_TARGET_PROPERTIES(cpkTest PROPERTIES OUTPUT_NAME cpkTest)
SET_TARGET_PROPERTIES(scTest PROPERTIES OUTPUT_NAME scTest)
SET_TARGET_PROPERTIES(ppTest PROPERTIES OUTPUT_NAME ppTest)
SET_TARGET_PROPERTIES(imTest PROPERTIES OUTPUT_NAME imTest)
SET_TARGET_PROPERTIES(rqTest PROPERTIES OUTPUT_NAME rqTest)
SET_TARGET_PROPERTIES(hqTest PROPERTIES OUTPUT_NAME hqTest)
SET_TARGET_PROPERTIES(kdTest PROPERTIES OUTPUT_NAME kdTest)
SET_TARGET_PROPERTIES(glTest PROPERTIES OUTPUT_NAME glTest)
SET_TARGET_PROPERTIES(lpTest PROPERTIES OUTPUT_NAME lpTest)
SET_TARGET_PROPERTIES(siTest PROPERTIES OUTPUT_NAME siTest)

INSTALL(TARGETS
	gpTest
	ioTest
	pkTest
	cskTest
	cpkTest
	scTest
	ppTest
	imTest
	rqTest
	hqTest
	kdTest
	glTest
	lpTest
	siTest
RUNTIME DESTINATION bin)

INSTALL(FILES 
	data/breathing1D.mat 
	data/breathing3D.mat 
DESTINATION 
	data 
PERMISSIONS
	OWNER_READ GROUP_READ WORLD_READ)

