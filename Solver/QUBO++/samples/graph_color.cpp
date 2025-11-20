/// @file graph_color_easy.cpp
/// @author Koji Nakano
/// @brief Solves randomly generated Graph Coloring Problem using QUBO++ Easy
/// Solver.
/// @version 2025-01-05

#include <boost/program_options.hpp>
#include "qbpp.hpp"
#include "qbpp_easy_solver.hpp"
#include "qbpp_graph_color.hpp"
#include "qbpp_misc.hpp"
namespace po = boost::program_options;
int main(int argc, char **argv) {
  po::options_description desc(
      "Solve a randomly generated Graph Coloring Problem using the QUBO++ Easy "
      "Solver.");
    desc.add_options()
    ("help,h", "Display this help message.")
    ("nodes,n", po::value<uint32_t>()->default_value(100), "Specify the number of nodes.")
    ("proximity,p", po::value<uint32_t>()->default_value(20), "Specify the proximity between nodes.")
    ("Circle,C", "Arrange nodes in a circular layout.")
    ("Delaunay,D", "Connect nodes using Delaunay triangulation.")
    ("colors,c", po::value<uint32_t>()->default_value(4), "Specify the number of colors to use.")
    ("time,t", po::value<uint32_t>()->default_value(10), "Set the time limit in seconds.")
    ("seed,s", po::value<uint32_t>(), "Set the initial random seed to reproduce the generated map.")
    ("blank,b", po::value<std::string>(), "Specify the output file (e.g., png, svg) to save the blank Graph.")
    ("output,o", po::value<std::string>(), "Specify the output file (e.g., png, svg) to save the solution.");
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
  uint32_t node_count = vm["nodes"].as<uint32_t>();
  uint32_t time_limit = vm["time"].as<uint32_t>();
  uint32_t color_count = vm["colors"].as<uint32_t>();
  if (vm.count("seed")) {
    qbpp::random_set_seed(vm["seed"].as<uint32_t>());
  }
  std::cout << "Generating random graph with " << node_count << " nodes"
            << std::endl;
  qbpp::graph_color::GraphColorMap graph_color_map;
  graph_color_map.gen_random_map(node_count, vm.count("Circle"));
  if (vm.count("Delaunay")) {
    graph_color_map.gen_delaunay_edges();
  } else {
    graph_color_map.gen_proximity_edges(vm["proximity"].as<uint32_t>());
  }
  qbpp::graph_color::GraphColorQuadModel model(graph_color_map, color_count);
  std::cout << "Variables = " << model.var_count()
            << " Linear Terms = " << model.term_count(1)
            << " Quadratic Terms = " << model.term_count(2) << std::endl;
  qbpp::easy_solver::EasySolver solver(model);
  solver.time_limit(time_limit);
  solver.target_energy(0);
  solver.enable_default_callback();
  auto sol = solver.search();
  graph_color_map.set_color_histogram(model, sol);
  graph_color_map.print();
  if (vm.count("output") > 0) {
    std::cout << "Writing the solution to " << vm["output"].as<std::string>()
              << std::endl;
    graph_color_map.draw(vm["output"].as<std::string>());
  }
  if (vm.count("blank") > 0) {
    std::cout << "Writing the solution to " << vm["blank"].as<std::string>()
              << std::endl;
    graph_color_map.draw(vm["blank"].as<std::string>(), true);
  }
}
