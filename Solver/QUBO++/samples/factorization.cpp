/// @file factorization.cpp
/// @brief Factorization using  QUBO++
/// @note cpp_int is used for large integers.
/// @author Koji Nakano
/// @version 2025.10.11

#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#define MAXDEG 4
#include "qbpp.hpp"
#include "qbpp_easy_solver.hpp"
#include "qbpp_exhaustive_solver.hpp"
namespace po = boost::program_options;
int main(int argc, char **argv) {
  po::options_description desc("Factorization using quartic polynomial model");
  desc.add_options()("help,h", "Show this help message")
    ("p,p", po::value<std::string>(), "Value of p")
    ("q,q", po::value<std::string>(), "Value of q")
    ("time,t", po::value<int>()->default_value(10), "Time limit in seconds")
    ("exhaustive,e", "Use exhaustive solver instead of easy solver");
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help") || !vm.count("p") || !vm.count("q")) {
      std::cout << desc << std::endl;
      return 0;
    }
    po::notify(vm);  
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\nUse -h for help.\n";
    return 1;
  }
  const auto p_val = qbpp::cpp_int(vm["p"].as<std::string>());
  const auto q_val = qbpp::cpp_int(vm["q"].as<std::string>());
  const auto product = p_val * q_val;
  auto p = 1 <= qbpp::var_int("p") <=
           qbpp::cpp_int(1) << boost::multiprecision::msb(p_val - 1) + 1;
  auto q = 1 <= qbpp::var_int("q") <=
           qbpp::cpp_int(1) << boost::multiprecision::msb(q_val - 1) + 1;
  std::cout << "p = " << p << "\n";
  std::cout << "q = " << q << "\n";
  auto f = p * q == product;
  f.simplify_as_binary();
  std::cout << "f = " << f << "\n";
  std::cout << "Var count = " << qbpp::all_var_count() << "\n";
  std::cout << "Linear term count = " << f.term_count(1) << "\n";
  std::cout << "Quadratic term count = " << f.term_count(2) << "\n";
  std::cout << "Cubic term count = " << f.term_count(3) << "\n";
  std::cout << "Quartic term count = " << f.term_count(4) << "\n";
  if (vm.count("exhaustive")) {
    auto solver = qbpp::exhaustive_solver::ExhaustiveSolver(f);
    solver.enable_default_callback();
    const auto sols = solver.search_optimal_solutions();
    std::cout << "Input : p * q = " << p_val << " * " << q_val << " = "
              << p_val * q_val << "\n";
    for (const auto &sol : sols) {
      std::cout << "Result: p * q = " << sol(p) << " * " << sol(q) << " = "
                << sol(p) * sol(q) << " Sol: " << sol << "\n";
    }
  } else {
    auto solver = qbpp::easy_solver::EasySolver(f);
    solver.time_limit(vm["time"].as<int>());
    solver.target_energy(0);
    solver.enable_default_callback();
    const auto sol = solver.search();
    std::cout << "Input   : p * q = " << p_val << " * " << q_val << " = "
              << p_val * q_val << "\n";
    std::cout << "Solution: p * q = " << sol(p) << " * " << sol(q) << " = "
              << sol(p) * sol(q) << std::endl;
    std::cout << "Sol: " << sol << "\n";
  }
  return 0;
}
