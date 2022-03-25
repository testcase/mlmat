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
	
	
## Objects
The following 7 objects have been started and have working code. All need to have the ability to use built in serialization capabilities to save models to a file. Can be text file, xml or binary. I guess xml would be best at this point.

*	mlpack.id3_tree - straightforward decision tree_
*	mlpack.kfn - highly configurable k-farthest neighbor search. 
*	mlpack.knn - highly configurable k-nearest neighbor search. 
*	mlpack.kmeans - kmeans clustering given labels and cluster centroids. centroids can be fed back into object to speed up clustering
*	mlpack.linear_svm - linear state vector machine. supervised learning for classification.
*	mlpack.som - self-organizing map - unsupervised clustering. not really part of mlpack but useful.
*	mlpack.gmm - gaussian mixture model for identification and generative uses. 
*	mlpack.data - reading and writing data files. csv, arff. handle bag or words encoding. also has ability to split into training testing. various preprocessing scaling. will likely be multiple objects.
*	mlpack.linear_regression - simple linear regression.
*	mlpack.sparse_autoencoder -  does do dimensionality reduction and generation
*	mlpack.pca - dimensionality reduction
	
These objects have not been started but are planned. 
*	mlpack.mlp - multi-layer perceptron. vanilla ann for learning/mapping. should have one-hot-encoding option. allow for various activiation functions - ReLU, tanh, etc.
*	mlpack.hmm - hidden markov model
*	mlpack.hoeffding_tree - is a decision tree that does incremental learning so useful for dealing with realtime developement of decision tree.
*	mlpack.lstm - long short term memory. type of rnn
*  	********* will like to get to here before making public. **********
*	mlpack.meanshift - another clustering algorithm. 
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
	
mlmat.convert -> n/a 
mlmat.gmm
mlmat.hmm
mlmat.hoeffding_tree
mlmat.id3_tree
mlmat.kfn
mlmat.kmeans
mlmat.knn
mlmat.linear_regression -> how to do auto scaling with this?
mlmat.linear_svm
mlmat.load
mlmat.lookup
mlmat.mean_shift
mlmat.mlp_classifier
mlmat.mlp_regressor
mlmat.pca -> has special case of scaling. 
mlmat.scaling
mlmat.som
mlmat.sparse_autoencoder
mlmat.split
mlmat.variational_autoendoder


	
	






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
