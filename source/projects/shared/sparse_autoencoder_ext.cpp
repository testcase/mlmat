/**
 * @file methods/sparse_autoencoder/sparse_autoencoder.cpp
 * @author Siddharth Agrawal
 *
 * Implementation of sparse autoencoders.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "sparse_autoencoder_ext.hpp"


namespace mlpack {



SparseAutoencoderExt::SparseAutoencoderExt() :
    visibleSize(0),
    hiddenSize(0),
    lambda(.0001),
    beta(3),
    rho(.01) {
        parameters = arma::mat();
}


void SparseAutoencoderExt::GetNewFeatures(arma::mat& data,
                                          arma::mat& features)
{
    const size_t l1 = hiddenSize;
    const size_t l2 = visibleSize;
    
    Sigmoid(parameters.submat(0, 0, l1 - 1, l2 - 1) * data +
            arma::repmat(parameters.submat(0, l2, l1 - 1, l2), 1, data.n_cols),
            features);
}

void SparseAutoencoderExt::Predict(arma::mat& features,
                                   arma::mat& output)
{
    const size_t l1 = hiddenSize;
    const size_t l2 = visibleSize;
    const size_t l3 = 2 * hiddenSize;
    
    Sigmoid(parameters.submat(l1, 0, l3 - 1, l2 - 1).t() * features +
            arma::repmat(parameters.submat(l3, 0, l3, l2 - 1).t(), 1, features.n_cols), output);

}






} // namespace mlpack
