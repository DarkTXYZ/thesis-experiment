/// @file tsp_easy.cpp
/// @author Koji Nakano
/// @brief Solves randomly generated Traveling Salesman Problem (TSP) using the
/// QUBO++ Easy solver.
/// @version 2025.10.07

#include <boost/program_options.hpp>
#include "qbpp.hpp"
#include "qbpp_easy_solver.hpp"
#include "qbpp_misc.hpp"
#include "qbpp_tsp.hpp"
namespace po = boost::program_options;
class MyEasySolver : public qbpp::easy_solver::EasySolver {
  const qbpp::tsp::TSPQuadModel &tsp_quad_model_;
 public:
  MyEasySolver(const qbpp::tsp::TSPQuadModel &tsp_quad_model)
      : qbpp::easy_solver::EasySolver(tsp_quad_model),
        tsp_quad_model_(tsp_quad_model) {}
  void callback(const qbpp::Sol &sol, double tts,
                [[maybe_unused]] std::string info) const override {
    static std::mutex callback_mutex;
    static std::optional<qbpp::energy_t> prev_energy = std::nullopt;
    std::lock_guard<std::mutex> lock(callback_mutex);
    if (!prev_energy.has_value() || sol.energy() < prev_energy.value()) {
      qbpp::tsp::TSPSol tsp_sol(tsp_quad_model_, sol);
      std::cout << "TTS = " << std::fixed << std::setprecision(3)
                << std::setfill('0') << tts << "s ";
      tsp_sol.print();
      prev_energy = sol.energy();
    }
  }
};
int main(int argc, char **argv) {
  po::options_description desc(
      "Solve a randomly generated Traveling Salesman Problem (TSP) using the "
      "QUBO++ Easy Solver.");
  desc.add_options()
    ("help,h", "Display this help message.")
    ("nodes,n", po::value<uint32_t>()->default_value(10), "Specify the number of nodes in the TSP map.")
    ("time,t", po::value<uint32_t>()->default_value(10), "Set the time limit in seconds.")
    ("seed,s", po::value<uint32_t>(), "Set the random seed to reproduce the TSP map.")
    ("blank,b", po::value<std::string>(), "Specify the output file (e.g., png, svg) to save the blank Map.")
    ("output,o", po::value<std::string>(), "Specify the output file (e.g., png, svg) to save the TSP solution.")
    ("fix,f", "Fix node 0 as the starting node.");
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
  uint32_t nodes = vm["nodes"].as<uint32_t>();
  uint32_t time_limit = vm["time"].as<uint32_t>();
  bool fix_first = vm.count("fix");
  if (vm.count("seed")) {
    qbpp::random_set_seed(vm["seed"].as<uint32_t>());
  }
  std::cout << "Generating random TSP Map with " << nodes << " nodes"
            << std::endl;
  qbpp::tsp::TSPMap tsp_map;
  tsp_map.gen_random_map(nodes);
  std::cout << "Generating a TSP QUBO expression" << std::endl;
  qbpp::tsp::TSPQuadModel tsp_quad_model(tsp_map, fix_first);
  std::cout << "Variables = " << tsp_quad_model.var_count()
            << " Linear Terms = " << tsp_quad_model.term_count(1)
            << " Quadratic Terms = " << tsp_quad_model.term_count(2)
            << std::endl;
  std::cout << "Generating an EasySolver object" << std::endl;
  MyEasySolver solver(tsp_quad_model);
  solver.time_limit(time_limit);
  std::cout << "Solving the TSP" << std::endl;
  auto sol = solver.search();
  qbpp::tsp::TSPSol tsp_sol(tsp_quad_model, sol);
  tsp_sol.print();
  if (vm.count("output")) {
    qbpp::tsp::DrawSimpleGraph graph;
    for (uint32_t i = 0; i < nodes; ++i) graph.add_node(tsp_map[i]);
    for (uint32_t i = 0; i < nodes; ++i)
      graph.add_edge(tsp_sol[i], tsp_sol[(i + 1) % nodes]);
    graph.draw(vm["output"].as<std::string>());
  }
  if (vm.count("blank")) {
    qbpp::tsp::DrawSimpleGraph graph;
    for (uint32_t i = 0; i < nodes; ++i) graph.add_node(tsp_map[i]);
    graph.draw(vm["blank"].as<std::string>());
  }
}
