/// @file labs_abs3.cpp
/// @brief This program solves the labs problem.
/// @author Koji Nakano
/// @version 2025.10.04

#include <boost/program_options.hpp>
#include "qbpp.hpp"
#include "qbpp_abs3_solver.hpp"
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
  po::options_description desc("LABS using ABS3 GPU Solver");
desc.add_options()
    ("help,h", "Show this help message.")
    ("size,s", po::value<uint32_t>()->default_value(20), "The size of the problem.")
    ("time,t", po::value<double>()->default_value(10.0), "Time limit in second.")
    ("target_energy,T", po::value<qbpp::energy_t>(), "Target energy.")
    ("verbose,v", po::value<int>()->default_value(0), "Verbose level (0: none, 1: new top sol)");
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
  auto solver = qbpp::abs3::ABS3Solver(f);
  auto params = qbpp::abs3::Params();
  if (vm.count("target_energy")) {
    params.add("target_energy",
               std::to_string(vm["target_energy"].as<qbpp::energy_t>()));
  }
  params.add("time_limit", std::to_string(vm["time"].as<double>()));
  params.add("verbose", std::to_string(vm["verbose"].as<int>()));
  auto sol = solver.search(params);
  std::cout << sol << std::endl;
}