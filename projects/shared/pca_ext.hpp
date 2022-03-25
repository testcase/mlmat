/**
 * @file methods/pca/pca.hpp
 * @author Ajinkya Kale
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Defines the PCA class to perform Principal Components Analysis on the
 * specified data set. There are many variations on how to do this, so
 * template parameters allow the selection of different techniques.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_PCA_PCA_EXT_HPP
#define MLPACK_METHODS_PCA_PCA_EXT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/pca/decomposition_policies/exact_svd_method.hpp>

namespace mlpack {
namespace pca {

/**
 * This class implements principal components analysis (PCA). This is a
 * common, widely-used technique that is often used for either dimensionality
 * reduction or transforming data into a better basis.  Further information on
 * PCA can be found in almost any statistics or machine learning textbook, and
 * all over the internet.
 */
template<typename DecompositionPolicy = ExactSVDPolicy>
class PCA_EXT
{
 public:
  /**
   * Create the PCA object, specifying if the data should be scaled in each
   * dimension by standard deviation when PCA is performed.
   *
   * @param scaleData Whether or not to scale the data.
   * @param decomposition Decomposition policy to use.
   */
    PCA_EXT(const bool scaleData = false,
      const DecompositionPolicy& decomposition = DecompositionPolicy());
    
   
    double Apply(arma::mat& data,
                 arma::vec& eigVal,
                 arma::mat& eigvec,
                 const size_t newDimension);
    

  

  //! Get whether or not this PCA object will scale (by standard deviation)
  //! the data when PCA is performed.
  bool ScaleData() const { return scaleData; }
  //! Modify whether or not this PCA object will scale (by standard deviation)
  //! the data when PCA is performed.
  bool& ScaleData() { return scaleData; }

 private:
  //! Scaling the data is when we reduce the variance of each dimension to 1.
  void ScaleData(arma::mat& centeredData)
  {
    if (scaleData)
    {
      // Scaling the data is when we reduce the variance of each dimension
      // to 1. We do this by dividing each dimension by its standard
      // deviation.
      arma::vec stdDev = arma::stddev(
          centeredData, 0, 1 /* for each dimension */);

      // If there are any zeroes, make them very small.
      for (size_t i = 0; i < stdDev.n_elem; ++i)
        if (stdDev[i] == 0)
          stdDev[i] = 1e-50;

      centeredData /= arma::repmat(stdDev, 1, centeredData.n_cols);
    }
  }

  //! Whether or not the data will be scaled by standard deviation when PCA is
  //! performed.
  bool scaleData;

  //! Decomposition method used to perform principal components analysis.
  DecompositionPolicy decomposition;
}; 

} // namespace pca
} // namespace mlpack

// Include implementation.
#include "pca_ext_impl.hpp"

#endif
