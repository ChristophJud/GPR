SET(gpl_app_src GaussianProcessLearn.cpp )
SET(gpp_app_src GaussianProcessPredict.cpp )

ADD_EXECUTABLE(gpLearn ${gpl_app_src})
ADD_EXECUTABLE(gpPredict ${gpp_app_src})

TARGET_LINK_LIBRARIES( gpLearn gplib)
TARGET_LINK_LIBRARIES( gpPredict gplib)

SET_TARGET_PROPERTIES(gpLearn PROPERTIES OUTPUT_NAME gpLearn)
SET_TARGET_PROPERTIES(gpPredict PROPERTIES OUTPUT_NAME gpPredict)

INSTALL(TARGETS
	gpLearn
	gpPredict
RUNTIME DESTINATION bin)

