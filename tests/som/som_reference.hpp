/// @file som_reference.hpp
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
///
/// Frozen copy of the original mlmat.som SOM implementation, kept as the
/// reference for the regression test (tests/som/som_test.cpp). The only
/// changes from the original: the MIN() macro is spelled std::min, and
/// find_bmu is public so the test can exercise it directly.

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
    enum initialization { uniform, gaussian, sample };
    SOM() {};

    SOM(long cols, long rows, long weights, long epochs, long neighborhood, double rate=.01, initialization init = uniform) {
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

        m_time_constant = m_num_iterations/log(m_map_radius);
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
            arma::Col<double> data_elem;
            m_neighborhood_radius = m_map_radius * exp(-(double)iter_count/m_time_constant);
            double width_sq = m_neighborhood_radius * m_neighborhood_radius;
            iter_count++;

            for(auto k=0;k<data.n_cols;k++) {
                m_bmu = find_bmu(data.col(k));


                for(auto i=0;i<m_nodes->n_cols;i++) {
                    long dx = (m_bmu % m_rows) - (i % m_rows);
                    long dy = (m_bmu / m_rows) - (i / m_rows);
                    double dist_sq = double((dx*dx)+(dy*dy));//squared distance

                    if(dist_sq < width_sq) {
                        data_elem = data.col(k);
                        m_influence = exp(-(dist_sq)/(2*width_sq));
                        adjust_weights(i, data_elem, rate, m_influence);
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
            std::vector<double> denominator(m_nodes->n_cols, 0.0);
            m_neighborhood_radius = m_map_radius * exp(-(double)iter_count/m_time_constant);
            double width_sq = m_neighborhood_radius * m_neighborhood_radius;
            iter_count++;

            for(auto k=0;k<data.n_cols;k++) {
                m_bmu = find_bmu(data.col(k));

                for(auto i=0;i<m_nodes->n_cols;i++) {
                    long dx = (m_bmu % m_rows) - (i % m_rows);
                    long dy = (m_bmu / m_rows) - (i / m_rows);
                    double dist_sq = double((dx*dx)+(dy*dy));//squared distance
                    double nh = exp(-(dist_sq)/(2*width_sq)); //neighborhood scaling
                    numerator.col(i) += data.col(k) * nh;
                    denominator[i] += nh;
                }
            }

            for(auto j=0;j<m_nodes->n_cols;j++) {
                m_nodes->col(j) = numerator.col(j)/denominator[j];
            }
        }
    }

    double get_euclidean_squared(const arma::Col<double>& target, const arma::Col<double>& weights) {
        return arma::accu(arma::square(target - weights));
    }

    void adjust_weights(long weight_index, arma::Col<double>& target, const double learning_rate, const double influence) {
        arma::Col<double> adjust = (target - m_nodes->col(weight_index)) * (learning_rate * influence );
        m_nodes->col(weight_index) += adjust;
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

    // (public in this frozen copy, for the regression test)
    size_t find_bmu(const arma::Col<double> &vec) {
        size_t winner = 0;
        double lowest_distance = std::numeric_limits<double>::max();
        double dist;
        for(auto i=0;i<m_nodes->n_cols;i++) {
            dist = get_euclidean_squared(vec, m_nodes->col(i));

            if(dist < lowest_distance) {
                lowest_distance = dist;
                winner = i;
            }
        }
        return winner;
    }


    long        m_bmu;
    double      m_map_radius;
    double      m_time_constant;
    long        m_num_iterations;
    long        m_rows;
    long        m_cols;
    double      m_neighborhood_radius;
    double      m_influence;
    double      m_learning_rate;
    initialization m_initialization;
    bool m_first_run = true;
};
