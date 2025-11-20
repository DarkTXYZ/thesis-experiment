/// @file ilp_grb.cpp
/// @brief Solves an Integer Linear Programming (ILP) problem using Gurobi
/// Optimizer through QUBO++ library.
/// @details
/// This is a sample program to solve an Integer Linear Programming
/// (ILP) problem using EasySolver through QUBO++ library.
/// @author Koji Nakano
/// @copyright Copyright (c) 2025, Koji Nakano
/// @version 2025-02-04

#include "qbpp.hpp"
#include "qbpp_grb.hpp"
int main() {
  auto x = 0 <= qbpp::var_int("x") <= 10;
  auto y = 0 <= qbpp::var_int("y") <= 10;
  auto objective = x + y;
  auto c1 = 0 <= 3 * x + y <= 10;
  auto c2 = 0 <= 2 * x + 3 * y <= 15;
  auto f = -objective + 100 * (c1 + c2);
  f.simplify_as_binary();
  std::cout << "f = " << f << std::endl;
  auto model = qbpp_grb::HuboModel(f);
  auto sol = model.optimize();
  std::cout << sol << std::endl;
  std::cout << "x = " << sol.get(x) << std::endl;
  std::cout << "y = " << sol.get(y) << std::endl;
  std::cout << "obj = " << sol.get(objective) << std::endl;
  std::cout << "*c1 = " << sol.get(*c1) << std::endl;
  std::cout << "*c2 = " << sol.get(*c2) << std::endl;
}
