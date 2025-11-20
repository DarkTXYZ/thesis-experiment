
/// @file qbpp_nqueen.hpp
/// @brief Generates QUBO expression for the N-Queens problem using the QUBO++
/// library
/// @details This file provides a class to generate a Model object for the
/// N-Queens problem using the QUBO++ library
/// @author Koji Nakano
/// @version 2025.06.12

#pragma once
#include "qbpp.hpp"
namespace qbpp {
namespace nqueen {
class NQueenModel : public qbpp::Model {
 public:
  enum class Mode { EXPAND, FAST, PARALLEL };
 private:
  const int dim_;
  const Vector<Vector<qbpp::Var>> X_;
  static qbpp::Model expand_mode(int dim, const Vector<Vector<qbpp::Var>> &X) {
    auto a = qbpp::expr(2 * dim - 3);
    auto b = qbpp::expr(2 * dim - 3);
    for (int i = 0; i < 2 * dim - 3; i++) {
      int k = i + 1;
      for (int j = 0; j < dim; j++) {
        if (k - j >= 0 && k - j < dim) {
          a[static_cast<size_t>(i)] +=
              X[static_cast<size_t>(j)][static_cast<size_t>(k - j)];
        }
      }
      int l = dim - i - 1;
      for (int j = 0; j < dim; j++) {
        if (l + j >= 0 && l + j < dim) {
          b[static_cast<size_t>(i)] +=
              X[static_cast<size_t>(j)][static_cast<size_t>(l + j)];
        }
      }
    }
    auto expr = qbpp::sum(qbpp::vector_sum(X) == 1);
    expr += qbpp::sum(qbpp::vector_sum(qbpp::transpose(X)) == 1);
    expr += qbpp::sum(0 <= a <= 1);
    expr += qbpp::sum(0 <= b <= 1);
    expr.simplify_as_binary();
    return qbpp::Model(expr);
  }
  static qbpp::Model fast_mode(int dim, const Vector<Vector<qbpp::Var>> &X) {
    qbpp::Expr expr = dim;
    for (int x = 0; x < dim; ++x) {
      for (int y = 0; y < dim; ++y) {
        expr -= X[static_cast<size_t>(x)][static_cast<size_t>(y)];
        for (int i = x + 1; i < dim; ++i) {
          expr += X[static_cast<size_t>(x)][static_cast<size_t>(y)] *
                  X[static_cast<size_t>(i)][static_cast<size_t>(y)];
        }
        for (int j = y + 1; j < dim; ++j) {
          expr += X[static_cast<size_t>(x)][static_cast<size_t>(y)] *
                  X[static_cast<size_t>(x)][static_cast<size_t>(j)];
        }
        for (int d = 1; d < dim - x; ++d) {
          if (y - d >= 0) {
            expr += X[static_cast<size_t>(x)][static_cast<size_t>(y)] *
                    X[static_cast<size_t>(x + d)][static_cast<size_t>(y - d)];
          }
          if (y + d < dim) {
            expr += X[static_cast<size_t>(x)][static_cast<size_t>(y)] *
                    X[static_cast<size_t>(x + d)][static_cast<size_t>(y + d)];
          }
        }
      }
    }
    return qbpp::Model(expr.simplify_as_binary());
  }
  static qbpp::Model parallel_mode(int dim,
                                   const Vector<Vector<qbpp::Var>> &X) {
    qbpp::Vector<qbpp::Expr> exprs(static_cast<size_t>(dim));
    tbb::parallel_for(0, dim, [&](int x) {
      for (int y = 0; y < dim; ++y) {
        exprs[static_cast<size_t>(x)] -=
            X[static_cast<size_t>(x)][static_cast<size_t>(y)];
        for (int i = x + 1; i < dim; ++i) {
          exprs[static_cast<size_t>(x)] +=
              X[static_cast<size_t>(x)][static_cast<size_t>(y)] *
              X[static_cast<size_t>(i)][static_cast<size_t>(y)];
        }
        for (int j = y + 1; j < dim; ++j) {
          exprs[static_cast<size_t>(x)] +=
              X[static_cast<size_t>(x)][static_cast<size_t>(y)] *
              X[static_cast<size_t>(x)][static_cast<size_t>(j)];
        }
        for (int d = 1; d < dim - x; ++d) {
          if (y - d >= 0) {
            exprs[static_cast<size_t>(x)] +=
                X[static_cast<size_t>(x)][static_cast<size_t>(y)] *
                X[static_cast<size_t>(x + d)][static_cast<size_t>(y - d)];
          }
          if (y + d < dim) {
            exprs[static_cast<size_t>(x)] +=
                X[static_cast<size_t>(x)][static_cast<size_t>(y)] *
                X[static_cast<size_t>(x + d)][static_cast<size_t>(y + d)];
          }
        }
      }
    });
    return qbpp::Model(simplify_as_binary(qbpp::sum(exprs) + dim));
  }
  std::tuple<qbpp::Model, int, Vector<Vector<qbpp::Var>>> helper_func(
      int dim, Mode mode) {
    Vector<Vector<qbpp::Var>> X(qbpp::var("X", dim, dim));
    switch (mode) {
      case Mode::EXPAND:
        return {expand_mode(dim, X), dim, X};
      case Mode::FAST:
        return {fast_mode(dim, X), dim, X};
      case Mode::PARALLEL:
        return {parallel_mode(dim, X), dim, X};
      default:
        throw std::runtime_error("Invalid mode");
    }
  }
  NQueenModel(std::tuple<qbpp::Model, int, Vector<Vector<qbpp::Var>>> &&tuple)
      : qbpp::Model(std::get<0>(tuple)),
        dim_(std::get<1>(tuple)),
        X_(std::get<2>(tuple)) {}
 public:
  NQueenModel(int dim, Mode mode) : NQueenModel(helper_func(dim, mode)) {}
  qbpp::Var get_var(int i, int j) const {
    return X_[static_cast<size_t>(i)][static_cast<size_t>(j)];
  }
};
}  
}  
