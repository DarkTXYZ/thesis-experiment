/// @file qbpp_tsp.hpp
/// @author Koji Nakano
/// @brief Generates a QUBO Expression for the Traveling Salesman Problem (TSP)
/// using QUBO++ library.
/// @version 2025.10.07

#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "qbpp.hpp"
#include "qbpp_misc.hpp"
namespace qbpp {
namespace tsp {
constexpr uint32_t uint32_limit = std::numeric_limits<uint32_t>::max();
class TSPMap {
  const uint32_t grid_size_;
  std::vector<std::pair<int32_t, int32_t>> nodes_;
 public:
  TSPMap(uint32_t grid_size = 100) : grid_size_(grid_size) {};
  void gen_random_map(uint32_t n);
  void add_node(uint32_t x, uint32_t y) { nodes_.push_back({x, y}); }
  uint32_t dist(const std::pair<int32_t, int32_t> &p1,
                const std::pair<int32_t, int32_t> &p2) const {
    return static_cast<uint32_t>(std::round(
        std::sqrt((p1.first - p2.first) * (p1.first - p2.first) +
                  (p1.second - p2.second) * (p1.second - p2.second))));
  }
  uint32_t dist(uint32_t i, uint32_t j) const {
    return dist(nodes_[i], nodes_[j]);
  }
  uint32_t min_dist(uint32_t x, uint32_t y) const {
    uint32_t min_dist = grid_size_ * 2;
    for (const auto &[px, py] : nodes_) {
      if (dist({x, y}, {px, py}) < min_dist) min_dist = dist({x, y}, {px, py});
    }
    return min_dist;
  }
  uint32_t node_count() const { return static_cast<uint32_t>(nodes_.size()); }
  uint32_t get_grid_size() const { return grid_size_; }
  std::pair<int32_t, int32_t> &operator[](uint32_t index) {
    return nodes_[index];
  }
};
class TSPQuadModel : public qbpp::QuadModel {
  const bool fix_first_;
  const qbpp::Vector<qbpp::Vector<qbpp::Var>> x_;
  std::tuple<qbpp::Model, bool, qbpp::Vector<qbpp::Vector<qbpp::Var>>>
  helper_func(const TSPMap &map, bool fix_first);
  TSPQuadModel(
      std::tuple<qbpp::QuadModel, bool, qbpp::Vector<qbpp::Vector<qbpp::Var>>>
          tuple)
      : qbpp::QuadModel(std::get<0>(tuple)),
        fix_first_(std::get<1>(tuple)),
        x_(std::get<2>(tuple)) {}
 public:
  TSPQuadModel(const TSPMap &map, bool fix_first = false)
      : TSPQuadModel(helper_func(map, fix_first)) {}
  uint32_t node_count() const { return static_cast<uint32_t>(x_.size()); }
  qbpp::Var get_var(uint32_t i, uint32_t j) const { return x_[i][j]; }
  bool get_fix_first() const { return fix_first_; }
};
class TSPSol {
  const TSPQuadModel tsp_quad_model_;
  const Sol sol_;
  const std::vector<uint32_t> tour_;
  static std::vector<uint32_t> gen_tour(const TSPQuadModel &tsp_quad_model,
                                        const Sol &sol);
 public:
  TSPSol(const TSPQuadModel &tsp_quad_model, const qbpp::Sol &sol)
      : tsp_quad_model_(tsp_quad_model),
        sol_(sol),
        tour_(gen_tour(tsp_quad_model, sol)) {}
  uint32_t operator[](uint32_t index) const { return tour_[index]; }
  uint32_t node_count() const { return tsp_quad_model_.node_count(); }
  void print() const {
    std::cout << sol_.energy() << " :";
    for (const auto &i : tour_)
      if (i != uint32_limit)
        std::cout << " " << i;
      else
        std::cout << " -";
    std::cout << std::endl;
  }
  void print_matrix() const {
    for (uint32_t i = 0; i < tsp_quad_model_.node_count(); i++) {
      for (uint32_t j = 0; j < tsp_quad_model_.node_count(); j++) {
        if (i == 0 && j == 0)
          std::cout << "1";
        else if (i == 0 || j == 0)
          std::cout << "0";
        else {
          std::cout << sol_.get(tsp_quad_model_.get_var(i, j));
        }
      }
      std::cout << std::endl;
    }
  }
};
class TSPModel : public qbpp::QuadModel {
  const bool fix_first_;
  const qbpp::Vector<qbpp::Vector<qbpp::Var>> x_;
  std::tuple<qbpp::QuadModel, bool, qbpp::Vector<qbpp::Vector<qbpp::Var>>>
  helper_func(const TSPMap &map, bool fix_first);
  TSPModel(std::tuple<qbpp::QuadModel, bool,
                      qbpp::Vector<qbpp::Vector<qbpp::Var>>> &&tuple)
      : qbpp::QuadModel(std::move(std::get<0>(tuple))),
        fix_first_(std::get<1>(tuple)),
        x_(std::move(std::get<2>(tuple))) {}
 public:
  TSPModel(const TSPMap &map, bool fix_first = false)
      : TSPModel(helper_func(map, fix_first)) {}
  uint32_t node_count() const { return static_cast<uint32_t>(x_.size()); }
  qbpp::Var get_var(uint32_t i, uint32_t j) const { return x_[i][j]; }
  bool get_fix_first() const { return fix_first_; }
};
class DrawSimpleGraph {
  std::vector<std::tuple<int, int, std::string>> nodes_;
  std::vector<std::tuple<int, int>> edges_;
 public:
  DrawSimpleGraph() = default;
  void add_node(int x, int y, const std::string &label = "") {
    nodes_.push_back(std::make_tuple(x, y, label));
  }
  void add_node(std::pair<int, int> node, const std::string &label = "") {
    add_node(node.first, node.second, label);
  }
  void add_edge(unsigned int node1, unsigned int node2) {
    edges_.push_back(std::make_pair(node1, node2));
  }
  uint32_t node_count() const { return static_cast<uint32_t>(nodes_.size()); }
  uint32_t edge_count() const { return static_cast<uint32_t>(edges_.size()); }
  void draw(std::string filename) {
    std::ostringstream dot_stream;
    dot_stream << "graph G {\n"
               << "node [shape=circle, fixedsize=true, width=5, fontsize=200, "
                  "penwidth=10];\n"
               << "edge [penwidth=10];\n";
    int index = 0;
    for (auto [x, y, s] : nodes_) {
      dot_stream << index << " [label = \"" << index << "\", pos = \"" << x
                 << "," << y << "!\"";
      if (s != "") dot_stream << " " << s;
      dot_stream << "];\n";
      ++index;
    }
    for (auto [node1, node2] : edges_) {
      dot_stream << node1 << " -- " << node2 << "\n";
    }
    dot_stream << "}\n";
    std::string command = "neato -T" +
                          filename.substr(filename.rfind('.') + 1) + " -o " +
                          filename;
    std::unique_ptr<FILE, qbpp::misc::PcloseDeleter> pipe(
        popen(command.c_str(), "w"));
    if (!pipe) {
      throw std::runtime_error(THROW_MESSAGE("popen() failed!"));
    }
    fprintf(pipe.get(), "%s", dot_stream.str().c_str());
  }
};
inline void TSPMap::gen_random_map(uint32_t n) {
  nodes_.reserve(n);
  uint32_t x, y;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t counter = 0;
    uint32_t max_dist = 1;
    while (counter < 10) {
      x = qbpp::random_gen(grid_size_);
      y = qbpp::random_gen(grid_size_);
      uint32_t dist = min_dist(x, y);
      if (dist >= max_dist) {
        max_dist = dist;
        counter++;
      }
    }
    add_node(x, y);
  }
}
inline std::tuple<qbpp::Model, bool, qbpp::Vector<qbpp::Vector<qbpp::Var>>>
TSPQuadModel::helper_func(const TSPMap &tsp_map, bool fix_first) {
  uint32_t node_count = tsp_map.node_count();
  auto x = qbpp::var("x", node_count, node_count);
  std::cout << "Generating QUBO expression for permutation." << std::endl;
  auto permutation_expr = qbpp::sum(
      (qbpp::vector_sum(x) == 1) + (qbpp::vector_sum(qbpp::transpose(x)) == 1));
  std::cout << "Generating QUBO expressions for tour distances." << std::endl;
  qbpp::Vector<qbpp::Expr> exprs(node_count);
  tbb::parallel_for(decltype(node_count)(0), node_count, [&](uint32_t i) {
    uint32_t next_i = (i + 1) % node_count;
    auto &expr = exprs[i];
    for (uint32_t j = 0; j < node_count; j++) {
      for (uint32_t k = 0; k < node_count; k++) {
        if (j == k) continue;
        expr += tsp_map.dist(j, k) * x[i][j] * x[next_i][k];
      }
    }
  });
  auto tsp_expr = qbpp::sum(exprs) + permutation_expr * tsp_map.get_grid_size();
  if (fix_first) {
    std::cout << "Fixing the first visiting node." << std::endl;
    qbpp::MapList fix0 = {{x[0][0], 1}};
    for (uint32_t i = 1; i < node_count; i++) {
      fix0.push_back({x[0][i], 0});
      fix0.push_back({x[i][0], 0});
    }
    tsp_expr.replace(fix0);
  }
  std::cout << "Simplifying the QUBO expression." << std::endl;
  tsp_expr.simplify_as_binary();
  return {tsp_expr, fix_first, x};
}
inline std::vector<uint32_t> TSPSol::gen_tour(
    const TSPQuadModel &tsp_quad_model, const Sol &sol) {
  std::vector<uint32_t> tour;
  for (uint32_t i = 0; i < tsp_quad_model.node_count(); i++) {
    if (tsp_quad_model.get_fix_first() && i == 0) {
      tour.push_back(0);
      continue;
    }
    uint32_t count = 0;
    uint32_t node;
    for (uint32_t j = (tsp_quad_model.get_fix_first() ? 1 : 0);
         j < tsp_quad_model.node_count(); j++) {
      if (sol.get(tsp_quad_model.get_var(i, j)) == 1) {
        node = j;
        count++;
      }
    }
    if (count != 1) node = uint32_limit;
    tour.push_back(node);
  }
  return tour;
}
}  
}  
