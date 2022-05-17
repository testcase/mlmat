# mlmat
max package for machine learning using jitter matrices


## Building:
This package is using the [min-devkit](https://github.com/Cycling74/min-devkit) however there are a couple 
fixes in the local version related to issue #189 #168 #184 #183

Building the objects requires having both [mlpack 3.4.2](https://www.mlpack.org/getstarted.html) and
armadillo headers/libraries installed in some typical location. Eventually libraries will go in the package 
just need to look at how to do @loader_path stuff with cmake. The cmake files for each project should look for these
in normal system places as well as /usr/local/ - where the default installation is. I built mlpack from source
and believe grabbed aramadillo from [homebrew] (https://brew.sh)

## Structure

In source/shared is a new class, `mlpack_operator`, that inherits from `matrix_operator` and `object`. This has all objects inherit from. This has some 
standard setup stuff that seems common, a dummy calc_cell so don't need to add that each time, and also 
methods for converting to/from armadillo matrices and jitter matrices. 

Luckliy mlpack comes with some nice examples which are easy to adapt.

## Data conversion
Armadillo matrices are organized in memory using column major order. When used with mlpack classes
each column in the matrix is a data point or feature vector and each row is elements of vector.

mlpack really uses two data types for matrices - double or long. Objects are 
currently setup to convert to one of these types depending on what is needed.


To deal with converting matrices to/from armadillo, objects have 3 possible modes that are set 
using the @mode attribute. 



### mode 0: 
This is most interoperable with jitter matrix. This mode expects a 1 or 2d jitter matrix where each cell is a data point and each plane are the values for that data. This will obviously not work for feature vectors with more than 32 elements but it is very efficient otherwise. In this mode objects may output 1d, 2d or 3d, matrices depending on the function of the object. 

### mode 1:
In this mode jitter matrices are going to be 1 or 2d with a single plane. Each row is a data point and each column in the row is an element of that point. 
	
### mode 2: 
In this mode jitter matrices are going to be 1 or 2d with a single plane. Each column is a data point and each row in the column is an element of that point. This corresponds to the typical layout used by mlpack for armadillo matrices and would make it easier to replicate examples using mlpack.
	
## Building

Over view
0. Clone repo 
1. Build armadillo static lib
2. Install prereqs for buildng mlpack (I am using homebrew)
3. Build mlpack static lib
4. Generate projects and build objects


### Clone rep

remember this to get all min-api stuff :
* `git submodule init`
* `git submodule update --init --recursive`


## Building on macOS

### Build armadillo static lib

`cd source/armadillo` to go to armadillo directory

run `mkdir build` to create build directory

`cd build`

This is the cmake I use. I do not install superlu, ARPACK, OpenBLAS, or hdf5 at the moment. These may speed things up but at the moment want to keep things simple

`cmake  -DBUILD_SHARED_LIBS=OFF -DALLOW_BLAS_LAPACK_MACOS=ON -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" -DCMAKE_OSX_DEPLOYMENT_TARGET=10.11 -G Xcode ..`

You can now use 

`cmake --build . --config Release` to build release version of library

if wanting to debug also do

`cmake --build . --config Debug`

You should now have static library at build/Release/libarmadillo.a

### Build mlpack static lib 

mlpack has a number of prerequisites

The minimal I find to work need boost, ensmallen and cereal libraries. I get them from homebrew

`brew install boost`
`brew install ensmallen`
`brew install cereal`

From the source/mlpack directory run
`mkdir build`

`cd build`

These are the cmake options I use. mlpack has a number of potential bindings but these are not of use for this project

`cmake -DBUILD_SHARED_LIBS=OFF -DBUILD_CLI_EXECUTABLES=OFF -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_JULIA_BINDINGS=OFF -DBUILD_GO_BINDINGS=OFF -DBUILD_R_BINDINGS=OFF -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" -DCMAKE_OSX_DEPLOYMENT_TARGET=10.11 -DARMADILLO_LIBRARY="../armadillo/build/Release/libarmadillo.a" -DARMADILLO_INCLUDE_DIR="../../armadillo/include" -G Xcode ..`


You can now use 

`cmake --build . --config Release` to build release version of library

if wanting to debug also do

`cmake --build . --config Debug`


### Generate projects and build

cd into main mlmat directory

run 
`mkdir build`

`cd build`

`cmake -GXcode ..`


You should now be able to do the following but I have not figure out how to disable or bypass testing. Or maybe better make suitable tests. You can run builds from individual projects or modify the mlmat project to skip the testing. 

`cmake --build . --config Release` to build release version of library

if wanting to debug also do

`cmake --build . --config Debug`

## Building on Windows

### Build armadillo static lib

`cd source\armadillo` to go to armadillo directory

run `mkdir build` to create build directory

`cd build`

This is the cmake I use. I do not install superlu, ARPACK, OpenBLAS, or hdf5 at the moment. These may speed things up but at the moment want to keep things simple

`cmake  -DBUILD_SHARED_LIBS=OFF -A x64 -G "Visual Studio 16 2019" ..`

You can now use 

`cmake --build . --config Release` to build release version of library

if wanting to debug also do

`cmake --build . --config Debug`

You should now have static library at build\Release\libarmadillo.lib


### Build mlpack static lib 

mlpack has a number of prerequisites

The minimal I find to work need boost as is should download ensmallen and cereal libraries if needed.

vcpkg install ensmallen:x64-windows   -- this also install openblas and lapack. also installs armadillo but don't want to use this version. 
vcpkg install cereal:x64-windows
vcpkg install boost:x64-windows -- this takes FOREVER and mlpack uses very little of boost



From the source/mlpack directory run
`mkdir build`

`cd build`

These are the cmake options I use. mlpack has a number of potential bindings but these are not of use for this project

Replace `[path to vcpkg]` with the correct path for your system

`cmake  -DCMAKE_TOOLCHAIN_FILE="[path to vcpkg]\vcpkg\scripts\buildsystems\vcpkg.cmake"  -A x64 -DBUILD_SHARED_LIBS=OFF -DBUILD_CLI_EXECUTABLES=OFF -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_JULIA_BINDINGS=OFF -DBUILD_GO_BINDINGS=OFF -DBUILD_R_BINDINGS=OFF -DARMADILLO_LIBRARY="../armadillo/build/Release/libarmadillo.lib" -DARMADILLO_INCLUDE_DIR="../../armadillo/include" -G "Visual Studio 16 2019" ..`
 

You can now use 

`cmake --build . --config Release` to build release version of library

if wanting to debug also do

`cmake --build . --config Debug`


### Generate projects and build

cd into main mlmat directory

run 
`mkdir build`

`cd build`

`cmake -G "Visual Studio 16 2019" ..`


You should now be able to do the following but I have not figure out how to disable or bypass testing. Or maybe better make suitable tests. You can run builds from individual projects or modify the mlmat project to skip the testing. 

`cmake --build . --config Release` to build release version of library

if wanting to debug also do

`cmake --build . --config Debug`


	
## Objects
The following 7 objects have been started and have working code. All need to have the ability to use built in serialization capabilities to save models to a file. Can be text file, xml or binary. I guess xml would be best at this point.

*	mlpack.id3_tree - straightforward decision tree_
*	mlpack.kfn - highly configurable k-farthest neighbor search. 
*	mlpack.knn - highly configurable k-nearest neighbor search. 
*	mlpack.kmeans - kmeans clustering given labels and cluster centroids. centroids can be fed back into object to speed up clustering
*   mlpack.mean_shift - another clustering algorithm. 
*	mlpack.linear_svm - linear state vector machine. supervised learning for classification.
*	mlpack.som - self-organizing map - unsupervised clustering. not really part of mlpack but useful.
*	mlpack.gmm - gaussian mixture model for identification and generative uses. 
*	mlpack.data - reading and writing data files. csv, arff. handle bag or words encoding. also has ability to split into training testing. various preprocessing scaling. will likely be multiple objects.
*	mlpack.linear_regression - simple linear regression.
*	mlpack.sparse_autoencoder -  does do dimensionality reduction and generation
*	mlpack.pca - dimensionality reduction
*	mlpack.mlp - multi-layer perceptron. vanilla ann for learning/mapping. should have one-hot-encoding option. allow for various activiation functions - ReLU, tanh, etc.
*	mlpack.hmm - hidden markov model
*	mlpack.hoeffding_tree - is a decision tree that does incremental learning so useful for dealing with realtime developement of decision tree.
*   mlpack.sparse_autoencoder

potential future objects

*	mlpack.naive_bayes - probabilistic clustering. 
*	mlpack.random_forest - ensemble of trees.
*	mlpack.gan - generative adversarial network. not really suitable for doing typical image stuff but could be interesting particularly on 3d objects. can also do DCGAN, WGAN, WGANP variants. not sure those are useful.
*	mlpack.rbm - Restricted Boltzmann Machine
*	mlpack.rl - there are a number of reinforcement learning algorithms but need to look further. these certainly could be useful. 
*	mlpack.adaboost. 
*	... lots of others. kernel density stuff.
*	... so many options for ANN might be nice to have object that could build them from dictionary specification to allow full flexibility but might not have much use. will ait

I am also thinking about some objects that don't require any understandong of the underlying algorithms that could be used by a new user. For instance some
type of mapping object that one could give examples to and it could be told to learn discrete or continuous mappings from the examples. 

For many applications one wants multiples of a type of model. For instance 1 hmm for each example of a gesture. This can be done through patching but might want 
to have objects that create a bank of objects and can dynamically add to the bank if more models are needed.
	
## Examples 
Have been working on examples and helpfiles. Will be going back and adding more descriptions so auto-generated ref files are better. 
	

	
	






Armadillo C++ Linear Algebra Library
Copyright 2008-2022 Conrad Sanderson (http://conradsanderson.id.au)
Copyright 2008-2016 National ICT Australia (NICTA)
Copyright 2017-2022 Data61 / CSIRO

This product includes software developed by Conrad Sanderson (http://conradsanderson.id.au)
This product includes software developed at National ICT Australia (NICTA)
This product includes software developed at Data61 / CSIRO

---

Attribution Notice.
As per UN General Assembly Resolution A/RES/ES-11/1
adopted on 2 March 2022 with 141 votes in favour and 5 votes against,
we attribute the violation of the sovereignty and territorial integrity of Ukraine,
and subsequent destruction of many Ukrainian cities and civilian infrastructure,
to large-scale military aggression by the Russian Federation (aided by Belarus).
Further details:
https://undocs.org/A/RES/ES-11/1
https://digitallibrary.un.org/record/3965290/files/A_RES_ES-11_1-EN.pdf
https://digitallibrary.un.org/record/3965290/files/A_RES_ES-11_1-RU.pdf



mlpack is provided without any warranty of fitness for any purpose.  You
can redistribute the library and/or modify it under the terms of the 3-clause
BSD license.  The text of the 3-clause BSD license is contained below.

mlpack contains some reproductions of the source code of Armadillo, which is
licensed under the Mozilla Public License v2.0 (MPL2).  This code is found in
src/mlpack/core/arma_extend/ and more details on the licensing are available
there.

mlpack also contains some reproductions of the source code of Boost, which is
licensed under the Boost Software License, version 1.0.  This code is found in
src/mlpack/core/boost_backport/ and more details on the licensing are available
there.

mlpack contain some usage of the source code of MNMLSTC Core library, which is
a backport of C++17 features to C++11. MNMLSTC is licensed under the Apache 2.0
License. This code can be found in src/mlpack/core/std_backport/ and more
details about licensing can be found there.

mlpack may contain some usage of the source code of stb, which is licensed 
under the MIT License and the Public Domain (www.unlicense.org). This code
is used in src/mlpack/core/data/load_image.hpp.

----
Copyright (c) 2007-2020, mlpack contributors (see COPYRIGHT.txt)
All rights reserved.

Redistribution and use of mlpack in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
