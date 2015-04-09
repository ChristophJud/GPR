# - Find file for the GPR package
# It defines the following variables
#  GPR_INCLUDE_DIRS - include directories for GPR
#  GPR_LIBRARIES    - libraries to link against
#  GPR_EXECUTABLE   - the GPR executable
 
# Compute paths
get_filename_component(GPR_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

set(GPR_INCLUDE_DIRS "${GPR_CMAKE_DIR}/../install/include")
message(${GPR_INCLUDE_DIRS}) 
set(GPR_LIBRARY_DIRS "${GPR_CMAKE_DIR}/../install/lib")
set(GPR_LIBRARIES gplib)
