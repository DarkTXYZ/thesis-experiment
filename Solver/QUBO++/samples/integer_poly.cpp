/// @file  integer_poly.cpp
/// @brief This program solves the integer polynomial problem.
/// @author Koji Nakano
/// @version 2025.06.14

#include <boost/program_options.hpp>
#include <cmath>
#include "qbpp.hpp"
#include "qbpp_easy_solver.hpp"
namespace po = boost::program_options;
po::variables_map vm;
std::string str_sol(const qbpp::Sol &sol) {
  std::ostringstream oss;
  oss << sol.energy() << " = ";
  for (qbpp::vindex_t i = 0; i < sol.var_count(); ++i) {
    oss << sol(i);
  }
  return oss.str();
}
int main(int argc, char **argv) {
  po::options_description desc("Solve the integer polynomial problem");
desc.add_options()
    ("help,h", "Show this help message.")
    ("n,n", po::value<uint32_t>(), "The value of n.")
    ("time,t", po::value<uint32_t>()->default_value(1), "Easy Solver Time limit in second.");
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const std::exception &e) {
    std::cerr << "Invalid arguments: " << e.what()
              << "\nUse -h or --help to see usage.\n";
    return 1;
  }
  if (vm.count("help") || !vm.count("n")) {
    std::cout << desc << std::endl;
    return 0;
  }
  auto n = vm["n"].as<uint32_t>();
  auto x = 0 <= qbpp::var_int("x") <= static_cast<uint32_t>(std::sqrt(n));
  auto y = 0 <= qbpp::var_int("y") <= static_cast<uint32_t>(std::sqrt(n));
  auto z = 0 <= qbpp::var_int("z") <= static_cast<uint32_t>(std::sqrt(n));
  auto objective = x * x * x - 2 * y * y * y + 5 * z;
  auto constraint = n <= x * x + y * y + z * z <= n + 1;
  auto f = objective + (n * n * n * n) * constraint;
  f.simplify_as_binary();
  std::cout << "f = " << f << std::endl;
  auto solver = qbpp::easy_solver::EasySolver(f);
  solver.time_limit(vm["time"].as<uint32_t>());
  solver.enable_default_callback();
  auto sol = solver.search();
  std::cout << "x = " << sol(x) << std::endl;
  std::cout << "y = " << sol(y) << std::endl;
  std::cout << "z = " << sol(z) << std::endl;
  std::cout << "Objective = " << sol(objective) << std::endl;
  std::cout << "Constraint = " << sol(*constraint) << std::endl;
}
