/// @author Koji Nakano
/// @brief Generates a QUBO Expression for the Graph Coloring Problem
/// using QUBO++ library.
/// @version 2025-03-20

#include <boost/polygon/voronoi.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "qbpp.hpp"
#include "qbpp_misc.hpp"
namespace qbpp {
namespace graph_color {
class GraphColorMap;
class GraphColorQuadModel;
class GraphColorMap {
  const uint32_t grid_size_;
  std::vector<std::pair<int32_t, int32_t>> nodes_;
  std::vector<std::pair<uint32_t, uint32_t>> edges_;
  qbpp::Vector<int32_t> color_;
  std::vector<uint32_t> color_hist_;
  uint32_t failure_ = 0;
  void add_node(uint32_t x, uint32_t y) { nodes_.push_back({x, y}); }
  uint32_t dist(const std::pair<int32_t, int32_t> &p1,
                const std::pair<int32_t, int32_t> &p2) const {
    return static_cast<uint32_t>(std::round(
        std::sqrt((p1.first - p2.first) * (p1.first - p2.first) +
                  (p1.second - p2.second) * (p1.second - p2.second))));
  }
  uint32_t dist(size_t i, size_t j) const { return dist(nodes_[i], nodes_[j]); }
  uint32_t min_dist(uint32_t x, uint32_t y) const {
    uint32_t min_dist = grid_size_ * 2;
    for (const auto &[px, py] : nodes_) {
      if (dist({x, y}, {px, py}) < min_dist) min_dist = dist({x, y}, {px, py});
    }
    return min_dist;
  }
  std::pair<int32_t, int32_t> &operator[](uint32_t index) {
    return nodes_[index];
  }
 public:
  GraphColorMap(uint32_t grid_size = 100) : grid_size_(grid_size) {};
  void gen_random_map(uint32_t n, bool is_circle = false);
  void gen_proximity_edges(uint32_t proximity);
  void gen_delaunay_edges();
  void set_color_histogram(const GraphColorQuadModel &model,
                           const qbpp::Sol &sol);
  uint32_t node_count() const { return static_cast<uint32_t>(nodes_.size()); }
  uint32_t get_grid_size() const { return grid_size_; }
  const std::vector<std::pair<uint32_t, uint32_t>> get_edges() const {
    return edges_;
  }
  void draw(const std::string &filename, bool is_blank = false);
  void print() {
    for (size_t i = 0; i < color_hist_.size(); i++) {
      std::cout << "Color " << i << " : " << color_hist_[i] << std::endl;
    }
    std::cout << "Failure : " << failure_ << std::endl;
  }
};
class GraphColorQuadModel : public qbpp::QuadModel {
  const qbpp::Vector<qbpp::Vector<qbpp::Var>> x_;
  std::pair<qbpp::Model, qbpp::Vector<qbpp::Vector<qbpp::Var>>> helper_func(
      const GraphColorMap &map, uint32_t color_count);
  GraphColorQuadModel(
      std::pair<qbpp::QuadModel, qbpp::Vector<qbpp::Vector<qbpp::Var>>> pair)
      : qbpp::QuadModel(pair.first), x_(pair.second) {}
 public:
  GraphColorQuadModel(const GraphColorMap &map, uint32_t color_count)
      : GraphColorQuadModel(helper_func(map, color_count)) {}
  uint32_t node_count() const { return static_cast<uint32_t>(x_.size()); }
  const qbpp::Vector<qbpp::Vector<qbpp::Var>> &get_x() const { return x_; }
};
inline void GraphColorMap::gen_random_map(uint32_t n, bool is_circle) {
  nodes_.reserve(n);
  uint32_t x, y;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t counter = 0;
    uint32_t max_dist = 1;
    while (counter < 10) {
      if (is_circle) {
        do {
          x = qbpp::random_gen(grid_size_);
          y = qbpp::random_gen(grid_size_);
        } while ((x - grid_size_ / 2) * (x - grid_size_ / 2) +
                     (y - grid_size_ / 2) * (y - grid_size_ / 2) >
                 grid_size_ * grid_size_ / 4);
      } else {
        x = qbpp::random_gen(grid_size_);
        y = qbpp::random_gen(grid_size_);
      }
      uint32_t dist = min_dist(x, y);
      if (dist >= max_dist) {
        max_dist = dist;
        counter++;
      }
    }
    add_node(x, y);
  }
}
inline void GraphColorMap::gen_proximity_edges(uint32_t proximity) {
  for (size_t i = 0; i < nodes_.size(); i++) {
    for (size_t j = i + 1; j < nodes_.size(); j++) {
      if (dist(i, j) < proximity) {
        edges_.push_back({i, j});
      }
    }
  }
}
inline void GraphColorMap::gen_delaunay_edges() {
  typedef boost::polygon::point_data<int> Point;
  std::vector<Point> points;
  for (const auto &[x, y] : nodes_) {
    points.push_back(Point(x, y));
  }
  boost::polygon::voronoi_diagram<double> vd;
  boost::polygon::construct_voronoi(points.begin(), points.end(), &vd);
  for (const auto &cell : vd.cells()) {
    size_t source_index = cell.source_index();
    const auto *edge = cell.incident_edge();
    do {
      size_t twin_source_index = edge->twin()->cell()->source_index();
      if (source_index < twin_source_index)
        edges_.push_back({static_cast<uint32_t>(source_index),
                          static_cast<uint32_t>(twin_source_index)});
      edge = edge->next();
    } while (edge != cell.incident_edge());
  }
}
inline void GraphColorMap::set_color_histogram(const GraphColorQuadModel &model,
                                               const qbpp::Sol &sol) {
  color_ = qbpp::onehot_to_int(sol.get(model.get_x()));
  for (auto c : color_) {
    if (c < 0) {
      ++failure_;
    } else {
      if (c >= static_cast<decltype(c)>(color_hist_.size())) {
        color_hist_.resize(static_cast<size_t>(c) + 1, 0);
      }
      color_hist_[static_cast<size_t>(c)]++;
    }
  }
}
inline void GraphColorMap::draw(const std::string &filename, bool is_blank) {
  const std::vector<std::string> color_palette = {
      "#FFFFFF",  
      "#FF0000",  
      "#00FF00",  
      "#FFFF00",  
      "#00FFFF",  
      "#FF00FF",  
      "#FFA500",  
      "#800080",  
      "#A52A2A",  
      "#87CEEB",  
      "#FFD700",  
      "#808080",  
      "#FF1493",  
      "#00CED1",  
      "#ADFF2F",  
      "#ADD8E6",  
      "#008000",  
      "#F0E68C",  
      "#7FFF00",  
      "#40E0D0",  
      "#DDA0DD",  
      "#FF4500",  
      "#DA70D6",  
      "#F08080",  
      "#87CEFA",  
      "#FF6347",  
      "#FFE4B5",  
      "#BA55D3",  
      "#3CB371",  
      "#4682B4",  
      "#B0E0E6",  
      "#7B68EE"   
  };
  std::ostringstream dot_stream;
  dot_stream << "graph G {\n"
             << "node [shape=circle, fixedsize=true, width=3, fontsize=100, "
                "penwidth=8];\n"
             << "edge [penwidth=8];\n";
  for (size_t i = 0; i < nodes_.size(); ++i) {
    dot_stream << i << " [label=\"" << i << "\", pos=\"" << nodes_[i].first
               << "," << nodes_[i].second << "!\"" << ", fillcolor=\"";
    if (is_blank) {
      dot_stream << "#FFFFFF";
    } else if (static_cast<size_t>(color_[i] + 1) < color_palette.size()) {
      dot_stream << color_palette[static_cast<size_t>(color_[i]) + 1];
    } else {
      dot_stream << color_palette[0];
    }
    dot_stream << "\", style=filled];\n";
  }
  for (const auto &[node1, node2] : edges_) {
    if (!is_blank && (color_[node1] == color_[node2] || color_[node1] < 0 ||
                      color_[node2] < 0)) {
      dot_stream << node1 << " -- " << node2
                 << " [style=dashed,penwidth=16,color=\"red\"];\n";
    } else {
      dot_stream << node1 << " -- " << node2 << ";\n";
    }
  }
  dot_stream << "}\n";
  std::string command =
      "neato -T" + filename.substr(filename.rfind('.') + 1) + " -o " + filename;
  std::unique_ptr<FILE, qbpp::misc::PcloseDeleter> pipe(
      popen(command.c_str(), "w"));
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  std::string dot_content = dot_stream.str();
  fwrite(dot_content.c_str(), sizeof(char), dot_content.size(), pipe.get());
}
inline std::pair<qbpp::Model, qbpp::Vector<qbpp::Vector<qbpp::Var>>>
GraphColorQuadModel::helper_func(const GraphColorMap &graph_map,
                                 uint32_t color_count) {
  auto x = qbpp::var("x", graph_map.node_count(), color_count);
  auto f = qbpp::sum(qbpp::vector_sum(x) == 1);
  for (auto [i, j] : graph_map.get_edges()) {
    f += qbpp::sum(x[i] * x[j]);
  }
  return {simplify_as_binary(f), x};
}
}  
}  
