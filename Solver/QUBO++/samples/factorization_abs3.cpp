/// @file factorization_abs3.cpp
/// @brief Factorization using the quartic polynomial model by the ABS3 GPU
/// solver.
/// @author Koji Nakano
/// @version 2025.10.11

#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#define MAXDEG 4
#include "qbpp.hpp"
#include "qbpp_abs3_solver.hpp"
namespace po = boost::program_options;
const uint32_t MAXVAL = 0x8000;  
int main(int argc, char **argv) {
  po::options_description desc(
      "Factorization using the quartic polynomial model by the ABS3 GPU "
      "solver.");
  desc.add_options()("help,h", "Show this help message")
    ("p,p", po::value<uint32_t>(), "Value of p")
    ("q,q", po::value<uint32_t>(), "Value of q")
    ("time,t", po::value<double>()->default_value(10.0), "Time limit in seconds")
    ("verbose,v", po::value<int>()->default_value(1), "Verbosity level (0-2)");
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
  const auto p_val = vm["p"].as<uint32_t>();
  const auto q_val = vm["q"].as<uint32_t>();
  const auto product = p_val * q_val;
  if (p_val > MAXVAL || q_val > MAXVAL) {
    std::cerr << "Error: p and q must be less than or equal to " << MAXVAL
              << " to avoid overflow.\n";
    return 1;
  }
  auto p = 1 <= qbpp::var_int("p") <= MAXVAL;
  auto q = 1 <= qbpp::var_int("q") <= MAXVAL;
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
  auto solver = qbpp::abs3::ABS3Solver(f);
  auto params = qbpp::abs3::Params();
  params.add("target_energy", "0");
  params.add("time_limit", std::to_string(vm["time"].as<double>()));
  params.add("verbose", std::to_string(vm["verbose"].as<int>()));
  const auto sol = solver.search(params);
  std::cout << "Input   : p * q = " << p_val << " * " << q_val << " = "
            << p_val * q_val << "\n";
  std::cout << "Solution: p * q = " << sol(p) << " * " << sol(q) << " = "
            << sol(p) * sol(q) << std::endl;
  std::cout << "Sol: " << sol << "\n";
}
