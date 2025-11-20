/// @file qbpp_misc.hpp
/// @author Koji Nakano
/// @brief A miscellaneous library used for sample programs of the QUBO++
/// library.
/// @details This library includes miscellaneous classes and functions used for
/// sample programs of the QUBO++ library. The classes include random number
/// generators and graph drawing functions.
/// @copyright 2025, Koji Nakano
/// @version 2025.06.19

#pragma once
#include <algorithm>
#include <atomic>
#include <boost/circular_buffer.hpp>
#include <boost/random/taus88.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <vector>
#include "qbpp.hpp"
extern "C" {
void *qbpp_tabu_create(qbpp::vindex_t var_count, qbpp::vindex_t tabu_capacity);
void qbpp_tabu_destroy(void *tabu);
qbpp::vindex_t qbpp_tabu_capacity(const void *tabu);
uint8_t qbpp_tabu_has(const void *tabu, qbpp::vindex_t index);
uint8_t qbpp_tabu_insert(void *tabu, qbpp::vindex_t index, bool must_be_new);
qbpp::vindex_t qbpp_non_tabu_random(const void *tabu);
}
namespace qbpp {
namespace misc {
struct PcloseDeleter {
  void operator()(FILE *file) const {
    if (file) pclose(file);
  }
};
class RandomSet {
  std::vector<vindex_t> position_;
  std::vector<vindex_t> variables_;
 public:
  explicit RandomSet(vindex_t size) : position_(size, vindex_limit) {
    variables_.reserve(size);
  }
  vindex_t var_count() const {
    return static_cast<vindex_t>(variables_.size());
  }
  void insert(vindex_t index) {
    if (position_[index] != vindex_limit) {
      throw std::runtime_error(THROW_MESSAGE("RandomSet: Insert variable (",
                                             index, ") already in set"));
    }
    variables_.push_back(index);
    position_[index] = static_cast<vindex_t>(variables_.size() - 1);
  }
  void swap(vindex_t pos1, vindex_t pos2) {
    if (pos1 == pos2) return;
    std::swap(variables_[pos1], variables_[pos2]);
    position_[variables_[pos1]] = pos1;
    position_[variables_[pos2]] = pos2;
  }
  void erase(vindex_t index) {
    if (position_[index] == vindex_limit) {
      throw std::runtime_error(
          THROW_MESSAGE("RandomSet: Erase variable (", index, ") not in set"));
    }
    swap(position_[index], var_count() - 1);
    variables_.pop_back();
    position_[index] = vindex_limit;
  }
  bool has(vindex_t index) const { return position_[index] != vindex_limit; }
  vindex_t select_at_random() const {
    return variables_[qbpp::random_gen(var_count())];
  }
  void print(const std::string &prefix = "") const {
    std::cout << prefix << " Size = " << var_count() << " : ";
    for (vindex_t i = 0; i < var_count(); ++i) {
      std::cout << " " << variables_[i];
    }
    std::cout << std::endl;
  }
};
template <typename T = energy_t>
class MinSet {
  using ValIndexMap = std::set<std::pair<T, vindex_t>>;
  ValIndexMap set_;
 public:
  MinSet() = default;
  vindex_t var_count() const { return static_cast<vindex_t>(set_.size()); }
  void insert(vindex_t index, T delta) {
    auto tuple_val = std::make_pair(delta, index);
    auto result = set_.insert(tuple_val);
    if (!result.second) {
      throw std::runtime_error(THROW_MESSAGE("MinSet: Insert variable (", index,
                                             ") already in set"));
    }
  }
  void erase(vindex_t index, T delta) {
    auto pair_val = std::make_pair(delta, index);
    auto result = set_.erase(pair_val);
    if (result == 0) {
      throw std::runtime_error(
          THROW_MESSAGE("MinSet: Erase variable (", index, ") not in set"));
    }
  }
  vindex_t first() const {
    if (set_.empty()) {
      throw std::runtime_error(THROW_MESSAGE("MinSet: Empty set"));
    }
    return (*set_.begin()).second;
  }
  std::pair<T, vindex_t> min() const {
    if (set_.empty()) {
      throw std::runtime_error(THROW_MESSAGE("MinSet: Empty set"));
    }
    return *set_.begin();
  }
  bool empty() const { return set_.empty(); }
  void print(const std::string &prefix) const {
    std::cout << prefix;
    for (auto &[val, i] : set_) {
      if constexpr (std::is_same_v<T, energy_t>) {
        std::cout << "(" << i << "," << val << ")";
      } else {
        std::cout << "(" << i << "," << val.constraint_delta() << ","
                  << val.objective_delta() << ")";
      }
    }
    std::cout << std::endl;
  }
};
template <typename T = energy_t>
class MinHeap {
  std::vector<std::pair<T, vindex_t>> heap_;
  std::vector<vindex_t> index_;
  void emplace_back(vindex_t index, energy_t delta) {
    heap_.emplace_back(std::make_pair(delta, index));
    index_[index] = heap_size() - 1;
  }
  void swap_heap(vindex_t i, vindex_t j) {
    std::swap(heap_[i], heap_[j]);
    index_[heap_[i].second] = i;
    index_[heap_[j].second] = j;
  }
  void bubble_up(vindex_t i) {
    while (i > 0) {
      vindex_t parent = (i - 1) / 2;
      if (heap_[i] < heap_[parent]) {
        swap_heap(i, parent);
        i = parent;
      } else {
        break;
      }
    }
  }
  void bubble_down(vindex_t i) {
    while (true) {
      vindex_t left = 2 * i + 1;
      vindex_t right = 2 * i + 2;
      vindex_t smallest = i;
      if (left < heap_size() && heap_[left] < heap_[smallest]) {
        smallest = left;
      }
      if (right < heap_size() && heap_[right] < heap_[smallest]) {
        smallest = right;
      }
      if (smallest != i) {
        swap_heap(i, smallest);
        i = smallest;
      } else {
        break;
      }
    }
  }
 public:
  MinHeap(size_t size) : index_(size, vindex_limit) {}
  vindex_t heap_size() const { return static_cast<vindex_t>(heap_.size()); }
  const std::vector<std::pair<T, vindex_t>> heap() const { return heap_; }
  void clear() {
    heap_.clear();
    std::fill(index_.begin(), index_.end(), vindex_limit);
  }
  std::pair<T, vindex_t> min() const {
    if (heap_.empty()) {
      throw std::runtime_error(THROW_MESSAGE("DualMinHeap: Empty heap"));
    }
    return heap_[0];
  }
  vindex_t first() const {
    if (heap_.empty()) {
      throw std::runtime_error(THROW_MESSAGE("DualMinHeap: Empty heap"));
    }
    return heap_[0].second;
  }
  vindex_t pop_first() {
    if (heap_.empty()) {
      throw std::runtime_error(THROW_MESSAGE("DualMinHeap: Empty heap"));
    }
    vindex_t index = heap_[0].second;
    erase(index);
    return index;
  }
  bool has(vindex_t index) const { return index_[index] != vindex_limit; }
  bool empty() const { return heap_.empty(); }
  vindex_t select_at_random() const {
    return heap_[qbpp::random_gen(heap_size())].second;
  }
  void insert(vindex_t index, energy_t delta) {
    if (index_[index] != vindex_limit) {
      throw std::runtime_error(THROW_MESSAGE("DualMinHeap: Insert variable (",
                                             index,
                                             ") already in "
                                             "heap"));
    }
    emplace_back(index, delta);
    bubble_up(heap_size() - 1);
  }
  void erase(vindex_t index) {
    vindex_t i = index_[index];
    if (i == vindex_limit) {
      throw std::runtime_error(THROW_MESSAGE("DualMinHeap: Erase variable (",
                                             index,
                                             ") not in "
                                             "heap"));
    }
    swap_heap(i, heap_size() - 1);
    heap_.pop_back();
    index_[index] = vindex_limit;
    if (i < heap_size()) {
      bubble_up(i);
      bubble_down(i);
    }
  }
  void print(const std::string &prefix) const {
    std::cout << prefix;
    for (auto &[delta, i] : heap_) {
      std::cout << "(" << i << "," << delta << ")";
    }
    std::cout << std::endl;
  }
};
template <typename T = energy_t>
class RandomMinSet {
  MinSet<T> min_set_;
  RandomSet random_set_;
 public:
  RandomMinSet(vindex_t size) : random_set_(size) {};
  vindex_t var_count() const {
    return static_cast<vindex_t>(min_set_.var_count());
  }
  void insert(vindex_t index, T delta) {
    min_set_.insert(index, delta);
    random_set_.insert(index);
  }
  void erase(vindex_t index, T delta) {
    min_set_.erase(index, delta);
    random_set_.erase(index);
  }
  bool has(vindex_t index) const { return random_set_.has(index); }
  vindex_t select_at_random() const { return random_set_.select_at_random(); }
  vindex_t first() const { return min_set_.first(); }
  std::pair<T, vindex_t> min() const { return min_set_.min(); }
  bool empty() const { return min_set_.empty(); }
  void print(const std::string &prefix) const {
    min_set_.print(prefix + " MIN:");
    random_set_.print(prefix + "RANDOM:");
  }
};
class Tabu {
  void *impl_;
  Tabu(const Tabu &) = delete;
  Tabu &operator=(const Tabu &) = delete;
  Tabu(Tabu &&) = delete;
  Tabu &operator=(Tabu &&) = delete;
 public:
  explicit Tabu(vindex_t var_count, vindex_t tabu_capacity) {
    impl_ = qbpp_tabu_create(var_count, tabu_capacity);
  }
  ~Tabu() { qbpp_tabu_destroy(impl_); }
  vindex_t capacity() const { return qbpp_tabu_capacity(impl_); }
  bool has(vindex_t index) const { return qbpp_tabu_has(impl_, index) == 1; }
  bool insert(vindex_t index, bool must_be_new) {
    return qbpp_tabu_insert(impl_, index, must_be_new) == 1;
  }
  bool insert(vindex_t index) { return qbpp_tabu_insert(impl_, index, false); }
  vindex_t non_tabu_random() const { return qbpp_non_tabu_random(impl_); }
};
}  
}  
