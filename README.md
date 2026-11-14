# mlmat
max package for machine learning using jitter matrices


The package is most reliant on the [mlpack](https://www.mlpack.org) and [armadillo](http://arma.sourceforge.net). Copyright info for both are below. 

Many of the description of classes and their use comes from the main source code or the many wonderful examples included with mlpack. 

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

See [BUILD.md](BUILD.md) for complete instructions for macOS and Windows.
In short:

```
git clone --recursive <repo>
cmake --preset macos        # or: windows
cmake --build --preset macos-release   # or: windows-release
```

All third-party dependencies are git submodules (see BUILD.md for the pinned
versions). mlpack, ensmallen, cereal and armadillo are consumed header-only,
and armadillo is used in no-wrapper mode linked directly against BLAS/LAPACK
(Apple Accelerate on macOS, static OpenBLAS/clapack on Windows), so no
third-party libraries need to be built separately or distributed with the
package. The spectral objects (buffer_stft, buffer_istft, buffer_griffinlim)
use bqfft, which uses vDSP on macOS and its built-in FFT on Windows.

## Potential future objects

* 	mlpack.variational_autoencoder - have one working but unclear if useful given constraints. _
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
	

	

[x]mlmat.convert.maxhelp
[x]mlmat.gmm.maxhelp
[x]mlmat.hoeffding_tree.maxhelp
[x]mlmat.id3_tree.maxhelp
[x]mlmat.kfn.maxhelp
[x]mlmat.kmeans.maxhelp
[x]mlmat.knn.maxhelp
[ ]mlmat.linear_regression.maxhelp
[ ]mlmat.linear_svm.maxhelp
[x]mlmat.load.maxhelp
[x]mlmat.lookup.maxhelp
[x]mlmat.mean_shift.maxhelp
[x]mlmat.mlp_classifier.maxhelp
[x]mlmat.mlp_regressor.maxhelp
[x]mlmat.pca.maxhelp
[x]mlmat.scaling.maxhelp
[x]mlmat.som.maxhelp
[x]mlmat.sparse_autoencoder.maxhelp
[ ]mlmat.split.maxhelp
[ ]mlmat.starter.maxhelp


 

---

Armadillo C++ Linear Algebra Library
Copyright 2008-2022 Conrad Sanderson (http://conradsanderson.id.au)
Copyright 2008-2016 National ICT Australia (NICTA)
Copyright 2017-2022 Data61 / CSIRO

This product includes software developed by Conrad Sanderson (http://conradsanderson.id.au)
This product includes software developed at National ICT Australia (NICTA)
This product includes software developed at Data61 / CSIRO

---

Attribution Notice.

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
