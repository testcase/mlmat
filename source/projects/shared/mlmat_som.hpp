/// @file mlmat_som.hpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
/// http://www.ai-junkie.com/ann/som/som4.html
///
/// Self-organizing map used by the mlmat.som object. The original
/// element-by-element implementation this was optimized against is kept as
/// a frozen reference in tests/som/ and compared there with fixed seeds.

#pragma once

#include <armadillo>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>
#include <cereal/cereal.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

class SOM {
public:
    enum initialization {uniform, gaussian, sample};
    SOM() = default;

    SOM(long cols, long rows, long weights, long epochs, long neighborhood, double rate = .01, initialization init = uniform) {
        m_num_iterations = epochs;
        m_learning_rate = rate;
        m_initialization = init;
        m_cols = cols;
        m_rows = rows;
        //will have to redo sample anyway
        if((m_initialization == uniform) || (m_initialization == sample)) {
            m_nodes = std::make_unique<arma::Mat<double>>(weights, cols*rows, arma::fill::randu);
        } else if(m_initialization == gaussian) {
            m_nodes = std::make_unique<arma::Mat<double>>(weights, cols*rows, arma::fill::randn);
        }
        m_map_radius = std::max(cols, rows) / 2;

        if(neighborhood != 0) {
            m_map_radius = std::min(double(neighborhood), m_map_radius);
        }

        // guard against degenerate maps (e.g. 1x1): keep the radius at 1 so
        // the time constant stays finite (the neighborhood then simply does
        // not decay, instead of producing NaNs as before)
        if(m_map_radius < 1) {
            m_map_radius = 1;
        }

        // note: for a radius of 1 this is +inf, i.e. the neighborhood
        // radius stays constant at 1
        m_time_constant = m_num_iterations / std::log(m_map_radius);

        // scratch buffers, so the hot loops do not allocate
        m_dots.set_size(cols*rows);
        m_dist_sq.set_size(cols*rows);
        m_node_norm_sq.set_size(cols*rows);
        m_exp_x.set_size(m_rows > 0 ? m_rows : 1);
        m_exp_y.set_size(m_cols > 0 ? m_cols : 1);
    }


    void run_epochs(arma::Mat<double>& data) {
        long n = m_num_iterations;
        long iter_count = 0;
        double rate = m_learning_rate;

        if(m_first_run && (m_initialization == sample)) {
            for(auto i=0;i<m_nodes->n_cols;i++) {
                int elem = mlpack::RandInt(data.n_cols);
                m_nodes->col(i) = data.col(elem);
            }
            m_first_run = false;
        }

        while(n--) {
            m_neighborhood_radius = m_map_radius * exp(-(double)iter_count/m_time_constant);
            double width_sq = m_neighborhood_radius * m_neighborhood_radius;
            iter_count++;

            // separable neighborhood lookup tables:
            //   exp(-(dx*dx + dy*dy) / (2*width_sq))
            //     == exp(-dx*dx / (2*width_sq)) * exp(-dy*dy / (2*width_sq))
            build_influence_luts(width_sq);

            // node norms, recomputed exactly once per epoch (they are
            // maintained incrementally during the online updates below)
            recompute_node_norms();

            for(auto k=0;k<data.n_cols;k++) {
                const arma::subview_col<double> data_elem = data.col(k);
                m_bmu = find_bmu(data_elem);

                // grid position of the best matching unit. (note: as in the
                // original, x runs over m_rows.)
                const long bmu_x = long(m_bmu) % m_rows;
                const long bmu_y = long(m_bmu) / m_rows;

                // only cells within the neighborhood radius rectangle around
                // the bmu can pass the (unchanged) disk test below
                const long reach = (width_sq > 0.0) ? long(std::ceil(std::sqrt(width_sq))) : 0;
                const long x0 = std::max(bmu_x - reach, 0L);
                const long x1 = std::min(bmu_x + reach, m_rows - 1);
                const long y0 = std::max(bmu_y - reach, 0L);
                const long y1 = std::min(bmu_y + reach, m_cols - 1);

                for(long jy = y0; jy <= y1; jy++) {
                    for(long jx = x0; jx <= x1; jx++) {
                        const long i = jy * m_rows + jx;
                        const long dx = bmu_x - jx;
                        const long dy = bmu_y - jy;
                        const double dist_sq = double((dx*dx)+(dy*dy));//squared distance

                        if(dist_sq < width_sq) {
                            const double influence = m_exp_x(size_t(dx < 0 ? -dx : dx)) *
                                                    m_exp_y(size_t(dy < 0 ? -dy : dy));
                            adjust_weights(i, data_elem, rate, influence);
                        }
                    }
                }
            }
            rate = m_learning_rate * exp(-(double)iter_count/m_num_iterations);
        }
    }


    void run_batch_knn(arma::Mat<double>& data) {
        long n = m_num_iterations;
        long iter_count = 0;

        if(m_first_run && (m_initialization == sample)) {
            for(auto i=0;i<m_nodes->n_cols;i++) {
                int elem = mlpack::RandInt(data.n_cols);
                m_nodes->col(i) = data.col(elem);
            }
            m_first_run = false;
        }

        while(n--) {
            arma::Mat<double> numerator(m_nodes->n_rows, m_nodes->n_cols, arma::fill::zeros);
            arma::Col<double> denominator(m_nodes->n_cols, arma::fill::zeros);
            m_neighborhood_radius = m_map_radius * exp(-(double)iter_count/m_time_constant);
            double width_sq = m_neighborhood_radius * m_neighborhood_radius;
            iter_count++;

            build_influence_luts(width_sq);
            recompute_node_norms();

            for(auto k=0;k<data.n_cols;k++) {
                const arma::subview_col<double> data_elem = data.col(k);
                m_bmu = find_bmu(data_elem);

                const long bmu_x = long(m_bmu) % m_rows;
                const long bmu_y = long(m_bmu) / m_rows;

                for(long jy = 0; jy < m_cols; jy++) {
                    const long dy = bmu_y - jy;
                    const double ey = m_exp_y(size_t(dy < 0 ? -dy : dy));
                    for(long jx = 0; jx < m_rows; jx++) {
                        const long i = jy * m_rows + jx;
                        const long dx = bmu_x - jx;
                        const double nh = m_exp_x(size_t(dx < 0 ? -dx : dx)) * ey;
                        numerator.col(i) += data_elem * nh;
                        denominator(i) += nh;
                    }
                }
            }

            for(auto j=0;j<m_nodes->n_cols;j++) {
                m_nodes->col(j) = numerator.col(j)/denominator(j);
            }
        }
    }

    void adjust_weights(long weight_index, const arma::subview_col<double>& target, const double learning_rate, const double influence) {
        const double s = learning_rate * influence;
        arma::subview_col<double> weights = m_nodes->col(weight_index);
        // element-wise in place; target and weights are only read at the
        // same index, so no temporary is needed
        weights += (target - weights) * s;

        // maintain ||w||^2 incrementally:
        //   ||w + a||^2 = ||w||^2 + 2*w'a + ||a||^2
        // with a = (target - w) * s: w'a = s * (w'target - ||w||^2) and
        // ||a||^2 = s^2 * ||target - w||^2, all already available
        // (m_dots(i) is target'w_i, m_dist_sq(i) the exact distance to the
        // winner). drift is corrected by the exact per-epoch recompute.
        m_node_norm_sq(weight_index) += 2.0 * s * (m_dots(weight_index) - m_node_norm_sq(weight_index))
                                        + s * s * m_dist_sq(weight_index);
    }

    template<typename Archive>
    void serialize(Archive& ar, const uint32_t /* version */)
    {
        ar(CEREAL_NVP(m_nodes));
    }

    void set_epochs(long i) {
        m_num_iterations = i;
    }

    std::unique_ptr<arma::Mat<double>> m_nodes { nullptr };

    // best matching unit via ||v - w||^2 = ||v||^2 - 2*v'w + ||w||^2.
    // all node dot products come from one matrix-vector product, and the
    // result is verified with an exact distance computation for all
    // near-tied candidates so the winner matches a full brute-force search.
    // (public so the regression test can exercise it directly)
    size_t find_bmu(const arma::subview_col<double>& vec) {
        const size_t node_count = m_nodes->n_cols;
        const size_t dim = m_nodes->n_rows;

        // v'w_i for every node (single BLAS call)
        m_dots = m_nodes->t() * vec;

        // ||v||^2
        const double* v = vec.colptr(0);
        double v_norm_sq = 0.0;
        for(size_t r = 0; r < dim; ++r) {
            v_norm_sq += v[r] * v[r];
        }

        size_t winner = 0;
        double lowest = std::numeric_limits<double>::max();
        for(size_t i = 0; i < node_count; ++i) {
            const double d = v_norm_sq - 2.0 * m_dots(i) + m_node_norm_sq(i);
            m_dist_sq(i) = d;
            if(d < lowest) {
                lowest = d;
                winner = i;
            }
        }

        // resolve near-ties exactly: recompute the exact distance for every
        // candidate within rounding distance of the minimum and pick the
        // first minimum (matching the reference's strict < scan)
        const double tol = 1e-9 * (std::fabs(lowest) + 1.0);
        size_t tied = 0;
        for(size_t i = 0; i < node_count; ++i) {
            if(m_dist_sq(i) <= lowest + tol) {
                ++tied;
            }
        }
        if(tied > 1) {
            size_t best_i = winner;
            double best_d = std::numeric_limits<double>::max();
            for(size_t i = 0; i < node_count; ++i) {
                if(m_dist_sq(i) <= lowest + tol) {
                    const double* w = m_nodes->colptr(i);
                    double d = 0.0;
                    for(size_t r = 0; r < dim; ++r) {
                        const double diff = v[r] - w[r];
                        d += diff * diff;
                    }
                    if(d < best_d) {
                        best_d = d;
                        best_i = i;
                    }
                }
            }
            winner = best_i;
            lowest = best_d;
        }
        m_dist_sq(winner) = lowest;

        return winner;
    }

    // exact per-node ||w||^2 (also public for the regression test)
    void recompute_node_norms() {
        const size_t node_count = m_nodes->n_cols;
        const size_t dim = m_nodes->n_rows;
        for(size_t i = 0; i < node_count; ++i) {
            const double* w = m_nodes->colptr(i);
            double norm_sq = 0.0;
            for(size_t r = 0; r < dim; ++r) {
                norm_sq += w[r] * w[r];
            }
            m_node_norm_sq(i) = norm_sq;
        }
    }

private:
    // per-epoch 1d influence tables so the inner loops need no exp() calls
    void build_influence_luts(double width_sq) {
        const double denom = 2.0 * width_sq;
        for(long j = 0; j < m_rows; ++j) {
            m_exp_x(size_t(j)) = (denom > 0.0) ? exp(-(double(j) * double(j)) / denom) : 0.0;
        }
        for(long j = 0; j < m_cols; ++j) {
            m_exp_y(size_t(j)) = (denom > 0.0) ? exp(-(double(j) * double(j)) / denom) : 0.0;
        }
    }

    size_t      m_bmu = 0;
    double      m_map_radius = 1;
    double      m_time_constant;
    long        m_num_iterations;
    long        m_rows;
    long        m_cols;
    double      m_neighborhood_radius;
    double      m_learning_rate;
    initialization m_initialization;
    bool m_first_run = true;

    // scratch buffers (never serialized; sized in the constructor)
    arma::Col<double>  m_dots;           // v'w for every node
    arma::Col<double>  m_dist_sq;        // ||v - w||^2 for every node
    arma::Col<double>  m_node_norm_sq;   // ||w||^2 per node
    arma::Col<double>  m_exp_x;          // exp(-dx^2 / 2 width^2)
    arma::Col<double>  m_exp_y;          // exp(-dy^2 / 2 width^2)
};
