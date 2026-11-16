# mlmat
max package for machine learning using jitter matrices


The package is most reliant on the [mlpack](https://www.mlpack.org) and [armadillo](http://arma.sourceforge.net). Cross platform FFT provided by [bqfft](https://hg.sr.ht/~breakfastquay/bqfft). See submodules for their licensing.

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

## Threading ##
There is a new threading use for some of the objects that have an explicit train message. (mlmat.gmm, mlmat.hmm, mlmat.hoeffding_tree, mlmat.id3_tree, mlmat.linear_svm, mlmat.mlp_classifier, mlmat.mlp_regressor, mlmat.som, mlmat.sparse_autoencoder, (but not mlmat.linear_regression, yet)). This is controlled by the @background attribute which defaults to 1. If set to 1, training runs on a background thread: the train message returns immediately, the object stays responsive, and queries keep using the previous model until the new one is ready. A done message is sent out the dumpout outlet when training finishes (error, if training failed). If 0, training blocks the main thread as before.
	
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
(Apple Accelerate on macOS, static multi-threaded OpenBLAS/clapack on Windows), so no
third-party libraries need to be built separately or distributed with the
package. The spectral objects (buffer_stft, buffer_istft, buffer_griffinlim)
use bqfft, which uses vDSP on macOS and its built-in FFT on Windows. 

## Examples
Have been working on examples and helpfiles. Will be going back and adding more descriptions so auto-generated ref files are better. 
	

	
- [] mlmat.buffer_griffinlim.maxhelp
- [] mlmat.buffer_istft.maxhelp
- [] mlmat.buffer_stft.maxhelp
- [x]mlmat.convert.maxhelp
- [x]mlmat.gmm.maxhelp
- [x]mlmat.hmm.maxhelp
- [x]mlmat.hoeffding_tree.maxhelp
- [x]mlmat.id3_tree.maxhelp
- [x]mlmat.kfn.maxhelp
- [x]mlmat.kmeans.maxhelp
- [x]mlmat.knn.maxhelp
- [x]mlmat.linear_regression.maxhelp
- [x]mlmat.linear_svm.maxhelp
- [x]mlmat.load.maxhelp
- [x]mlmat.lookup.maxhelp
- [x]mlmat.mean_shift.maxhelp
- [x]mlmat.mlp_classifier.maxhelp
- [x]mlmat.mlp_regressor.maxhelp
- [] mlmat.nmf.maxhelp
- [x]mlmat.pca.maxhelp
- []mlmat.radical.maxhelp
- [x]mlmat.scaling.maxhelp
- [x]mlmat.som.maxhelp
- [x]mlmat.sparse_autoencoder.maxhelp
- [] mlmat.spectral_convolve
- [] mlmat.spectral_peaks
- [x]mlmat.split.maxhelp

