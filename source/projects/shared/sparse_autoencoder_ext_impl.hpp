/**
 * @file methods/sparse_autoencoder/sparse_autoencoder_impl.hpp
 * @author Siddharth Agrawal
 *
 * Implementation of sparse autoencoders.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_EXT_IMPL_HPP
#define MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_EXT_IMPL_HPP

// In case it hasn't been included yet.
#include "sparse_autoencoder_ext.hpp"

namespace mlpack {

template<typename OptimizerType>
SparseAutoencoderExt::SparseAutoencoderExt(const arma::mat& data,
                                     const size_t visibleSize,
                                     const size_t hiddenSize,
                                     double lambda,
                                     double beta,
                                     double rho,
                                     OptimizerType optimizer) :
    visibleSize(visibleSize),
    hiddenSize(hiddenSize),
    lambda(lambda),
    beta(beta),
    rho(rho)
{
  SparseAutoencoderFunction encoderFunction(data, visibleSize, hiddenSize,
                                            lambda, beta, rho);

  parameters = encoderFunction.GetInitialPoint();

  // Train the model.
  Timer::Start("sparse_autoencoder_optimization");
  const double out = optimizer.Optimize(encoderFunction, parameters);
  Timer::Stop("sparse_autoencoder_optimization");

  Log::Info << "SparseAutoencoderExt::SparseAutoencoderExt(): final objective of "
      << "trained model is " << out << "." << std::endl;
}

template<typename OptimizerType, typename... CallbackTypes>
SparseAutoencoderExt::SparseAutoencoderExt(const arma::mat& data,
                                     const size_t visibleSize,
                                     const size_t hiddenSize,
                                     double lambda,
                                     double beta,
                                     double rho,
                                     OptimizerType optimizer,
                                     CallbackTypes&&... callbacks) :
    visibleSize(visibleSize),
    hiddenSize(hiddenSize),
    lambda(lambda),
    beta(beta),
    rho(rho)
{
  SparseAutoencoderFunction encoderFunction(data, visibleSize, hiddenSize,
                                            lambda, beta, rho);

  parameters = encoderFunction.GetInitialPoint();

  // Train the model.
  Timer::Start("sparse_autoencoder_optimization");
  const double out = optimizer.Optimize(encoderFunction, parameters,
      callbacks...);
  Timer::Stop("sparse_autoencoder_optimization");

  Log::Info << "SparseAutoencoderExt::SparseAutoencoderExt(): final objective of "
      << "trained model is " << out << "." << std::endl;
}

template<typename Archive>
void SparseAutoencoderExt::serialize(Archive& ar, const uint32_t  /* version */)
{
    ar(CEREAL_NVP(visibleSize));
    ar(CEREAL_NVP(hiddenSize));
    ar(CEREAL_NVP(lambda));
    ar(CEREAL_NVP(beta));
    ar(CEREAL_NVP(rho));
    ar(CEREAL_NVP(parameters));
}



} // namespace mlpack

#endif
