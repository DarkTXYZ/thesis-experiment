/// @file qbpp_exhaustive_solver.hpp
/// @author Koji Nakano
/// @brief Exhaustive HUBO Solver for solving HUBO problems.
/// @details The ExhaustiveSolver class provides a straightforward HUBO solver
/// that evaluates all possible solutions. This solver is primarily intended
/// for testing the correctness of HUBO expressions.
/// For more details on the algorithm, please refer to the following paper:
/// Masaki Tao et al., "A Work-Time Optimal Parallel Exhaustive Search
/// Algorithm for the QUBO and the Ising Model, with GPU Implementation," IPDPS
/// Workshops 2020: 557-566. https://doi.org/10.1109/IPDPSW50202.2020.00098
/// @version 2025.10.14

#pragma once
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <boost/circular_buffer.hpp>
#include <random>
#include <set>
#include "qbpp.hpp"
namespace qbpp {
namespace exhaustive_solver {
class SolDelta;
class Sol;
class SearchAlgorithm;
class ExhaustiveSolver;
class Sol : public qbpp::Sol {
  bool is_all_solutions_ = false;
  bool is_optimal_solutions_ = false;
  std::vector<qbpp::Sol> all_solutions_;
 public:
  explicit Sol(const HuboModel &hubo_model) : qbpp::Sol(hubo_model) {};
  Sol(const Sol &) = default;
  Sol(Sol &&) = default;
  Sol &operator=(const Sol &) = default;
  Sol &operator=(Sol &&) = default;
  std::vector<qbpp::Sol> &all_solutions() { return all_solutions_; }
  qbpp::Sol sol() const { return all_solutions_.front(); }
  void all_solution_mode() {
    is_optimal_solutions_ = false;
    is_all_solutions_ = true;
  }
  void optimal_solution_mode() {
    is_optimal_solutions_ = true;
    is_all_solutions_ = false;
  }
  void all_solutions(std::vector<qbpp::Sol> &&all_solutions) {
    all_solutions_ = all_solutions;
    qbpp::Sol::operator=(all_solutions_.front());
  }
  std::string str() const {
    if (is_all_solutions_ || is_optimal_solutions_) {
      std::ostringstream oss;
      uint32_t count = 0;
      for (const auto &sol : all_solutions_) {
        oss << "(" << count++ << ") " << sol;
        if (count < all_solutions_.size()) oss << std::endl;
      }
      return oss.str();
    }
    return qbpp::str(static_cast<qbpp::Sol>(*this));
  }
  std::vector<qbpp::Sol>::const_iterator begin() const {
    return all_solutions_.begin();
  }
  std::vector<qbpp::Sol>::const_iterator end() const {
    return all_solutions_.end();
  }
  size_t size() const { return all_solutions_.size(); }
  const qbpp::Sol &operator[](size_t i) const { return all_solutions_[i]; }
  qbpp::Sol &operator[](size_t i) { return all_solutions_[i]; }
};
inline std::ostream &operator<<(std::ostream &os, const Sol &sol) {
  os << sol.str();
  return os;
}
class ExhaustiveSolver {
 protected:
  const HuboModel hubo_model_;
  bool enable_default_callback_ = false;
 public:
  ExhaustiveSolver(const HuboModel &hubo_model) : hubo_model_(hubo_model) {}
  mutable std::mutex callback_mutex_;
  virtual ~ExhaustiveSolver() = default;
  virtual void callback(const SolHolder &sol_holder) const {
    static std::optional<energy_t> prev_energy = std::nullopt;
    std::lock_guard<std::mutex> lock(callback_mutex_);
    if (enable_default_callback_) {
      if (!prev_energy.has_value() ||
          sol_holder.energy() < prev_energy.value()) {
        prev_energy = sol_holder.energy();
        std::cout << "TTS = " << std::fixed << std::setprecision(3)
                  << std::setfill('0') << sol_holder.tts()
                  << "s Energy = " << sol_holder.energy() << std::endl;
      }
    }
  }
  vindex_t var_count() const { return hubo_model_.var_count(); }
  qbpp::Sol search();
  Sol search_optimal_solutions();
  Sol search_all_solutions();
  void enable_default_callback(bool enable = true) {
    enable_default_callback_ = enable;
  }
  const HuboModel &hubo_model() const { return hubo_model_; }
};
class SearchAlgorithm {
  const ExhaustiveSolver &exhaustive_solver_;
  const HuboModel &hubo_model_;
  const std::vector<vindex_t> var_order_;
  qbpp::SolHolder sol_holder_;
  bool is_all_solutions_ = false;
  bool is_optimal_solutions_ = false;
  std::vector<qbpp::Sol> all_solutions_;
  mutable std::mutex all_solutions_mutex_;
  std::vector<SolDelta> sol_deltas;
  std::vector<vindex_t> init_var_order(const HuboModel &hubo_model) {
    std::vector<std::pair<vindex_t, vindex_t>> degree_var;
    degree_var.resize(hubo_model.var_count());
    for (vindex_t i = 0; i < hubo_model.var_count(); ++i) {
      degree_var[i] = std::make_pair(hubo_model.degree(i), i);
    }
    std::sort(degree_var.begin(), degree_var.end(), std::greater<>());
    std::vector<vindex_t> var_order;
    var_order.resize(hubo_model.var_count());
    for (vindex_t i = 0; i < hubo_model.var_count(); ++i) {
      var_order[i] = degree_var[i].second;
    }
    return var_order;
  }
 public:
  explicit SearchAlgorithm(const ExhaustiveSolver &exhaustive_solver)
      : exhaustive_solver_(exhaustive_solver),
        hubo_model_(exhaustive_solver_.hubo_model()),
        var_order_(init_var_order(hubo_model_)),
        sol_holder_(hubo_model_) {}
  const HuboModel &hubo_model() const { return hubo_model_; }
  const std::vector<vindex_t> &var_order() const { return var_order_; }
  vindex_t var_count() const { return hubo_model_.var_count(); }
  std::vector<qbpp::Sol> &all_solutions() { return all_solutions_; }
  qbpp::SolHolder &sol_holder() { return sol_holder_; }
  void register_new_sol(const qbpp::Sol &sol) {
    if (!is_all_solutions_ && sol_holder_.energy() < sol.energy()) return;
    std::lock_guard<std::mutex> lock(all_solutions_mutex_);
    if (is_all_solutions_) {
      all_solutions_.push_back(sol);
    } else if (is_optimal_solutions_) {
      if (sol_holder_.energy() == sol.energy()) {
        all_solutions_.push_back(sol);
      } else if (sol_holder_.energy() > sol.energy()) {
        all_solutions_.clear();
        all_solutions_.push_back(sol);
      }
    }
    if (sol_holder_.set_if_better(sol)) {
      exhaustive_solver_.callback(sol_holder_);
    }
  }
  void search();
  void search_optimal_solutions();
  void search_all_solutions();
  void gen_sol_deltas(SolDelta &sol_delta, vindex_t index);
};
class SolDelta : public Sol {
 protected:
  const HuboModel &hubo_model_;
  SearchAlgorithm &search_algorithm_;
  const std::vector<vindex_t> &var_order_ = search_algorithm_.var_order();
  std::vector<energy_t> delta_;
 public:
  explicit SolDelta(SearchAlgorithm &search_algorithm)
      : Sol(search_algorithm.hubo_model()),
        hubo_model_(search_algorithm.hubo_model()),
        search_algorithm_(search_algorithm) {
    delta_.resize(var_count());
    for (vindex_t i = 0; i < var_count(); ++i) {
      delta_[i] = search_algorithm.hubo_model().term1(i);
    }
  }
  template <size_t N>
  inline void update_delta_delta(const TermVector<N - 1> &terms,
                                 std::vector<energy_t> &delta_delta,
                                 energy_t flip_bit_sign)  
  {
    static_assert(N >= 3, "N must be >= 3");
    for (size_t i = 0; i < terms.size(); ++i) {
      std::array<vindex_t, N - 1> idx{};
      std::array<uint8_t, N - 1> val{};
      uint8_t sum = 0;
      const auto inds = terms.indices(i);  
      for (size_t j = 0; j < N - 1; ++j) {
        idx[j] = inds[j];
        val[j] = static_cast<uint8_t>(get(idx[j]));
        sum = static_cast<uint8_t>(sum + val[j]);
      }
      const energy_t factor = terms.coeff(i) * flip_bit_sign;
      if (sum <= static_cast<uint8_t>(N - 3)) continue;
      if (sum == static_cast<uint8_t>(N - 1)) {
        for (size_t j = 0; j < N - 1; ++j) {
          delta_delta[idx[j]] -= factor;
        }
        continue;
      }
      uint8_t target = 0;
      for (; target < N - 1; ++target) {
        if (val[target] == 0) {
          delta_delta[idx[target]] += factor;
          break;
        }
      }
      if (target == N - 1) {
        delta_delta[idx[N - 2]] += factor;
      }
    }
  }
  void flip(vindex_t flip_index) override {
    vindex_t index = var_order_[flip_index];
    if (index >= var_count()) {
      throw std::out_of_range(
          THROW_MESSAGE("Sol: index (", index, ") out of range"));
    }
    energy_ = energy() + delta_[index];
    int32_t flip_bit_sign = 1 - 2 * get(index);
    std::vector<energy_t> delta_delta(var_count(), 0);
    const auto &t2 = hubo_model_.term2(index);
    for (size_t i = 0; i < t2.size(); ++i) {
      vindex_t j = t2.indices(i)[0];
      coeff_t coeff = t2.coeff(i);
      delta_delta[j] += coeff * flip_bit_sign * (1 - 2 * get(j));
    }
    if (hubo_model_.max_power() <= 2) goto post_process;
    update_delta_delta<3>(hubo_model_.term3(index), delta_delta, flip_bit_sign);
    if (hubo_model_.max_power() <= 3) goto post_process;
    update_delta_delta<4>(hubo_model_.term4(index), delta_delta, flip_bit_sign);
    if (hubo_model_.max_power() <= 4) goto post_process;
    update_delta_delta<5>(hubo_model_.term5(index), delta_delta, flip_bit_sign);
    if (hubo_model_.max_power() <= 5) goto post_process;
    update_delta_delta<6>(hubo_model_.term6(index), delta_delta, flip_bit_sign);
    if (hubo_model_.max_power() <= 6) goto post_process;
    update_delta_delta<7>(hubo_model_.term7(index), delta_delta, flip_bit_sign);
    if (hubo_model_.max_power() <= 7) goto post_process;
    update_delta_delta<8>(hubo_model_.term8(index), delta_delta, flip_bit_sign);
    if (hubo_model_.max_power() <= 8) goto post_process;
    update_delta_delta<9>(hubo_model_.term9(index), delta_delta, flip_bit_sign);
    if (hubo_model_.max_power() <= 9) goto post_process;
    update_delta_delta<10>(hubo_model_.term10(index), delta_delta,
                           flip_bit_sign);
    if (hubo_model_.max_power() <= 10) goto post_process;
    update_delta_delta<11>(hubo_model_.term11(index), delta_delta,
                           flip_bit_sign);
    if (hubo_model_.max_power() <= 11) goto post_process;
    update_delta_delta<12>(hubo_model_.term12(index), delta_delta,
                           flip_bit_sign);
    if (hubo_model_.max_power() <= 12) goto post_process;
    update_delta_delta<13>(hubo_model_.term13(index), delta_delta,
                           flip_bit_sign);
    if (hubo_model_.max_power() <= 13) goto post_process;
    update_delta_delta<14>(hubo_model_.term14(index), delta_delta,
                           flip_bit_sign);
    if (hubo_model_.max_power() <= 14) goto post_process;
    update_delta_delta<15>(hubo_model_.term15(index), delta_delta,
                           flip_bit_sign);
    if (hubo_model_.max_power() <= 15) goto post_process;
    update_delta_delta<16>(hubo_model_.term16(index), delta_delta,
                           flip_bit_sign);
  post_process:
    for (vindex_t i = 0; i < var_count(); ++i) {
      if (delta_delta[i] == 0) continue;
      if (i == index)
        throw std::runtime_error(
            THROW_MESSAGE("Unexpected error. delta_delta[i] != 0"));
      delta_[i] += delta_delta[i];
    }
    delta_[index] = -delta_[index];
    if (!energy_.has_value()) {
      throw std::out_of_range(THROW_MESSAGE("energy_ is not set."));
    }
    bit_vector_.flip(index);
  }
  void search(vindex_t index) {
    if (index >= 1) search(index - 1);
    flip(index);
    search_algorithm_.register_new_sol(*this);
    if (index >= 1) search(index - 1);
  }
  void search() {
    search_algorithm_.register_new_sol(*this);
    search(var_count() - 1);
  }
};
inline qbpp::Sol ExhaustiveSolver::search() {
  SearchAlgorithm search_algorithm(*this);
  search_algorithm.search();
  return search_algorithm.sol_holder().sol();
}
inline Sol ExhaustiveSolver::search_optimal_solutions() {
  SearchAlgorithm search_algorithm(*this);
  search_algorithm.search_optimal_solutions();
  Sol sol(hubo_model_);
  sol.optimal_solution_mode();
  sol.all_solutions(std::move(search_algorithm.all_solutions()));
  return sol;
}
inline Sol ExhaustiveSolver::search_all_solutions() {
  SearchAlgorithm search_algorithm(*this);
  search_algorithm.search_all_solutions();
  Sol sol(hubo_model_);
  sol.all_solution_mode();
  sol.all_solutions(std::move(search_algorithm.all_solutions()));
  return sol;
}
inline void SearchAlgorithm::gen_sol_deltas(SolDelta &sol_delta,
                                            vindex_t index) {
  if (var_count() == index) return;
  gen_sol_deltas(sol_delta, index + 1);
  sol_delta.flip(index);
  sol_deltas.push_back(sol_delta);
  gen_sol_deltas(sol_delta, index + 1);
}
inline void SearchAlgorithm::search() {
  const int parallel_param = 8;
  SolDelta sol_delta(*this);
  if (hubo_model_.term_count() == 0) {
    register_new_sol(sol_delta);
    return;
  }
  if (var_count() <= 16) {
    register_new_sol(sol_delta);
    sol_delta.search(var_count() - 1);
    return;
  }
  sol_deltas.push_back(sol_delta);
  gen_sol_deltas(sol_delta, var_count() - parallel_param);
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, sol_deltas.size()),
      [&](const tbb::blocked_range<size_t> &range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
          register_new_sol(sol_deltas[i]);
          sol_deltas[i].search(var_count() - (parallel_param + 1));
        }
      });
}
inline void SearchAlgorithm::search_optimal_solutions() {
  is_optimal_solutions_ = true;
  search();
  tbb::parallel_sort(all_solutions_.begin(), all_solutions_.end());
}
inline void SearchAlgorithm::search_all_solutions() {
  is_all_solutions_ = true;
  search();
  tbb::parallel_sort(all_solutions_.begin(), all_solutions_.end());
}
}  
}  
