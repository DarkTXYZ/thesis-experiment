/// @file nqueen_easy.cpp
/// @author Koji Nakano
/// @brief Solves the N-Queens problem using ABS2 QUBO solver from QUBO++
/// library.
/// @version 2025.06.19

#include <boost/program_options.hpp>
#define MAXDEG 2
#include "qbpp.hpp"
#include "qbpp_easy_solver.hpp"
#include "qbpp_nqueen.hpp"
namespace po = boost::program_options;
int main(int argc, char *argv[]) {
  po::options_description desc(
      "N-Queens Problem Solver using QUBO++ Easy Solver");
  desc.add_options()("help,h", "produce help message")
     ("dimension,d", po::value<int>()->default_value(8), "set dimension of the chessboard")
     ("time_limit,t", po::value<uint32_t>()->default_value(10), "set time limit in seconds")
     ("expand,e", "expand the one-hot formula for QUBO model generation")
     ("fast,f", "fast mode for QUBO model generation")
     ("parallel,p", "parallel mode for QUBO model generation (default)");
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
  } catch (const std::exception &e) {
    std::cout << "Wrong arguments. Please use -h/--help option to see the "
                 "usage.\n";
    return 1;
  }
  po::notify(vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  int dimension = vm["dimension"].as<int>();
  uint32_t time_limit = vm["time_limit"].as<uint32_t>();
  qbpp::nqueen::NQueenModel::Mode mode;
  if (vm.count("expand")) {
    mode = qbpp::nqueen::NQueenModel::Mode::EXPAND;
  } else if (vm.count("fast")) {
    mode = qbpp::nqueen::NQueenModel::Mode::FAST;
  } else if (vm.count("parallel")) {
    mode = qbpp::nqueen::NQueenModel::Mode::PARALLEL;
  } else {
    mode = qbpp::nqueen::NQueenModel::Mode::PARALLEL;
  }
  qbpp::nqueen::NQueenModel nqueen_model(dimension, mode);
  std::cout << "Generating the QUBO model." << std::endl;
  std::cout << "Variables = " << nqueen_model.var_count()
            << " Linear Terms = " << nqueen_model.term_count(1)
            << " Quadratic Terms = " << nqueen_model.term_count(2) << std::endl;
  std::cout << "Generating an EasySolver object for the QUBO model."
            << std::endl;
  auto solver = qbpp::easy_solver::EasySolver(nqueen_model);
  solver.time_limit(time_limit);
  solver.target_energy(0);
  solver.base_time(0);
  solver.enable_default_callback();
  std::cout << "Executing the EasySolver to solve the QUBO model." << std::endl;
  auto sol = solver.search();
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j)
      std::cout << static_cast<int>(sol.get(nqueen_model.get_var(i, j)));
    std::cout << std::endl;
  }
  std::cout << "Dimension = " << dimension << " TTS = " << std::fixed
            << std::setprecision(3) << std::setfill('0') << solver.tts()
            << "s Energy = " << sol.energy() << std::endl;
  std::cout << "flip_count = " << solver.flip_count() << std::endl;
}
