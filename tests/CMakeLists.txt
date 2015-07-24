file(GLOB test_files "*.cpp")

enable_testing()

set(target_tests "")
foreach( _filename ${test_files})
	get_filename_component(_test ${_filename} NAME_WE)
	add_executable(${_test} ${_filename})
	target_link_libraries(${_test} gplib)
	set_target_properties(${_test} PROPERTIES OUTPUT_NAME ${_test})
	add_test(${_test} ${_test})
	set(target_tests ${target_tests} ${_test})
endforeach()

INSTALL(TARGETS ${target_tests} RUNTIME DESTINATION bin)

INSTALL(FILES 
	data/breathing1D.mat 
	data/breathing3D.mat 
DESTINATION 
	data 
PERMISSIONS
	OWNER_READ GROUP_READ WORLD_READ)


