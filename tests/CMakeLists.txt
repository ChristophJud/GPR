SET(gp_lib_src GaussianProcessTest.cpp )
SET(io_lib_src IOTest.cpp )
SET(pk_lib_src PeriodicKernelTest.cpp )
SET(sk_lib_src SumKernelTest.cpp )
SET(sc_lib_src ScopeTest.cpp )
SET(pp_lib_src PosteriorProcessTest.cpp )
SET(im_lib_src InversionMethodsTest.cpp )

ADD_EXECUTABLE(gpTest ${gp_lib_src})
ADD_EXECUTABLE(ioTest ${io_lib_src})
ADD_EXECUTABLE(pkTest ${pk_lib_src})
ADD_EXECUTABLE(skTest ${sk_lib_src})
ADD_EXECUTABLE(scTest ${sc_lib_src})
ADD_EXECUTABLE(ppTest ${pp_lib_src})
ADD_EXECUTABLE(imTest ${im_lib_src})

TARGET_LINK_LIBRARIES( gpTest gplib)
TARGET_LINK_LIBRARIES( ioTest gplib)
TARGET_LINK_LIBRARIES( pkTest gplib)
TARGET_LINK_LIBRARIES( skTest gplib)
TARGET_LINK_LIBRARIES( scTest gplib)
TARGET_LINK_LIBRARIES( ppTest gplib)
TARGET_LINK_LIBRARIES( imTest gplib)

SET_TARGET_PROPERTIES(gpTest PROPERTIES OUTPUT_NAME gpTest)
SET_TARGET_PROPERTIES(ioTest PROPERTIES OUTPUT_NAME ioTest)
SET_TARGET_PROPERTIES(pkTest PROPERTIES OUTPUT_NAME pkTest)
SET_TARGET_PROPERTIES(skTest PROPERTIES OUTPUT_NAME skTest)
SET_TARGET_PROPERTIES(scTest PROPERTIES OUTPUT_NAME scTest)
SET_TARGET_PROPERTIES(ppTest PROPERTIES OUTPUT_NAME ppTest)
SET_TARGET_PROPERTIES(imTest PROPERTIES OUTPUT_NAME imTest)

INSTALL(TARGETS
	gpTest
	ioTest
	pkTest
	skTest
	scTest
	ppTest
	imTest
RUNTIME DESTINATION bin)

