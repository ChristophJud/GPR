# GPR - Basic Gaussian Process Library

Basic Gaussian process regression library. (Eigen3 required)

## Features
* Multivariate Gaussian process regression
* Calculation of the derivative at a point
* Calculation of the uncertainty at a point
* Save and Load the Gaussian process to/from files
* Kernels: White, Gaussian, Periodic, RationalQuadratic, Sum and Product
* Derivative of the kernels
* Likelihood functions: Gaussian Log Likelihood (incl. derivative wrt. kernel parameter)
* Prior distributions: Gaussian, Inverse Gaussian, Gamma (incl. sampling, cdf and inverse cdf)
* * Prior distributions can be built by providing their mode and variance



## Getting Started
To setup the library clone the git repository first
```
git clone https://github.com/ChristophJud/GPR.git
```

The building of GPR is based on [cmake](http://www.cmake.org/). So navigate to the main directory GPR and create a build directory.
```
mkdir build	# create a build directory
cd build
ccmake ..	# ccmake is an easy tool to set config parameters
```
Set the build type and the installation directory and
```
CMAKE_CXX_FLAGS		-std=c++11
```

Since GPR depends on the matrix library [Eigen](http://eigen.tuxfamily.org) provide its include directory
```
EIGEN3_INCLUDE_DIR	/path/to/eigen/eigen-3.2.4/install/include/eigen3
```

If not all required [Boost](http://www.boost.org) libraries are found on the system provide a custom installation
```
Boost_INCLUDE_DIR 	/path/to/boost/boost_1_57_0/
Boost_LIBRARY_DIR	/path/to/boost/boost_1_57_0/stage/lib/

```
Make sure that boost has been built with C++11 by adding ```cxxflags="-std=c++11"``` to the ```b2``` command.

Finally, type
```
make install -j8
```
and the library including all test programs will be built.

### Include the library in your own cmake project
If you want to include the library into your own project the straight forward way is the following:
Add 
```
FIND_PACKAGE(GPR REQUIRED)
``` 
to your CMakeLists.txt file and provide
```
GPR_DIR			/path/to/main/gpr/project/dir/cmake 
```
In the CMakeLists.txt you can link your program with ```${GPR_LIBRARIES}```.

## Examples
The tests can be seen as good examples to how to use the library. 

## TODOs
* Matrix valued kernels
* Store/load into/from hdf5 files
 
## Issues
* Load/Save of product and sum kernel

## References
A thorough introduction can be found in the open book of C.E. Rasmussen: Rasmussen, Carl Edward. [Gaussian processes for machine learning.](http://www.gaussianprocess.org/gpml/) (2006).

## License
GPR itself is licensed under the Apache 2.0 license. It depends, however, on other open source projects, which are distributed under different licenses. These are [Eigen](http://eigen.tuxfamily.org) and [Boost](http://www.boost.org).
