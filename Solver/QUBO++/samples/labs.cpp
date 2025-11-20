/// @file labs_hubo.cpp
/// @brief This program solves the LABS(Low Autocorrelation Binary Sequence).
/// @author Koji Nakano
/// @version 2025.10.14

#include <boost/program_options.hpp>
#include "qbpp.hpp"
#include "qbpp_easy_solver.hpp"
#include "qbpp_exhaustive_solver.hpp"
namespace po = boost::program_options;
po::variables_map vm;
std::string str_sol(const qbpp::Sol& sol) {
  std::ostringstream oss;
  oss << sol.energy() << " = ";
  for (uint32_t i = 0; i < sol.var_count(); ++i) {
    oss << sol(i);
  }
  return oss.str();
}
int main(int argc, char** argv) {
  po::options_description desc(
      "LABS(Low Autocorrelation Binary Sequence) Solver");
desc.add_options()
    ("help,h", "Show this help message.")
    ("size,s", po::value<uint32_t>()->default_value(20), "The size of the problem.")
    ("time,t", po::value<uint32_t>()->default_value(10), "Easy Solver Time limit in second.")
    ("target_energy,T", po::value<int32_t>()->default_value(-100), "Easy Solver Target Energy.")
    ("best_sols,b", po::value<size_t>()->default_value(10), "Easy Solver Best Solution Capacity.")
    ("exhaustive,e", "Use Exhaustive Solver instead of Easy Solver");
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << "Invalid arguments: " << e.what()
              << "\nUse -h or --help to see usage.\n";
    return 1;
  }
  if (vm.count("help") || !vm.count("size")) {
    std::cout << desc << std::endl;
    return 0;
  }
  auto size = vm["size"].as<uint32_t>();
  auto x = qbpp::var("x", size);
  auto f = qbpp::expr();
  for (size_t d = 1; d < size; ++d) {
    auto temp = qbpp::expr();
    for (size_t i = 0; i < size - d; ++i) {
      temp += (2 * x[i] - 1) * (2 * x[i + d] - 1);
    }
    f += qbpp::sqr(temp);
  }
  f.simplify_as_binary();
  std::cout << "f = " << f << std::endl;
  if (vm.count("exhaustive")) {
    auto solver = qbpp::exhaustive_solver::ExhaustiveSolver(f);
    solver.enable_default_callback();
    auto sols = solver.search_optimal_solutions();
    size_t i = 0;
    for (const auto& sol : sols) {
      std::cout << i++ << " : " << str_sol(sol) << std::endl;
    }
  } else {
    size_t best_sol_count = vm["best_sols"].as<size_t>();
    auto solver = qbpp::easy_solver::EasySolver(f);
    solver.time_limit(vm["time"].as<uint32_t>());
    solver.enable_best_sols(best_sol_count);
    solver.target_energy(vm["target_energy"].as<int32_t>());
    solver.enable_default_callback();
    auto sols = solver.search();
    std::cout << sols << std::endl;
    size_t i = 0;
    std::cout << "Best Solutions:" << sols.size() << std::endl;
    for (const auto& sol : sols.best_sols()) {
      std::cout << i++ << ":" << str_sol(sol) << std::endl;
    }
  }
}