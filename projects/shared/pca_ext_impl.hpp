/**
 * @file methods/pca/pca_impl.hpp
 * @author Ajinkya Kale
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Implementation of PCA class to perform Principal Components Analysis on the
 * specified data set.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_PCA_PCA_EXT_IMPL_HPP
#define MLPACK_METHODS_PCA_PCA_EXT_IMPL_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/lin_alg.hpp>
#include "pca_ext.hpp"

namespace mlpack {
namespace pca {

template<typename DecompositionPolicy>
PCA_EXT<DecompositionPolicy>::PCA_EXT(
    const bool scaleData, const DecompositionPolicy& decomposition) :
    scaleData(scaleData),
    decomposition(decomposition)
{ }

template<typename DecompositionPolicy>
double PCA_EXT<DecompositionPolicy>::Apply(arma::mat& data,
                                           arma::vec& eigVal,
                                           arma::mat& eigvec,
                                           const size_t newDimension)
{
  // Parameter validation.
  if (newDimension == 0)
    Log::Fatal << "PCA::Apply(): newDimension (" << newDimension << ") cannot "
        << "be zero!" << std::endl;
  if (newDimension > data.n_rows)
    Log::Fatal << "PCA::Apply(): newDimension (" << newDimension << ") cannot "
        << "be greater than the existing dimensionality of the data ("
        << data.n_rows << ")!" << std::endl;

  // Center the data into a temporary matrix.
  arma::mat centeredData;
  math::Center(data, centeredData);

  // Scale the data if the user ask for.
  ScaleData(centeredData);

  decomposition.Apply(data, centeredData, data, eigVal, eigvec, newDimension);

  if (newDimension < eigvec.n_rows)
    // Drop unnecessary rows.
    data.shed_rows(newDimension, data.n_rows - 1);

  // The svd method returns only non-zero eigenvalues so we have to calculate
  // the right dimension before calculating the amount of variance retained.
  double eigDim = std::min(newDimension - 1, (size_t) eigVal.n_elem - 1);

  // Calculate the total amount of variance retained.
  return (sum(eigVal.subvec(0, eigDim)) / sum(eigVal));
}



} // namespace pca
} // namespace mlpack

#endif
