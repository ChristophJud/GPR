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
RUNTIME DESTINATION bin)

