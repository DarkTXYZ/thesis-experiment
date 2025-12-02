/// @file qbpp_easy_solver.hpp
/// @author Koji Nakano
/// @brief HUBO Solver for solving HUBO problems
/// @details High Order Unconstrained Binary Optimization (HUBO) solver up to
/// degree 8. This is a native solver that does not reduce the degree.
/// @version 2025.10.14

#pragma once
#include <queue>
#include <unordered_set>
#include "qbpp.hpp"
#include "qbpp_defs.hpp"
#include "qbpp_misc.hpp"
namespace qbpp {
namespace easy_solver {
class BestSols;
class EasySolver;
class SolDelta;
class QueueSet;
struct SolHash {
  uint64_t operator()(const qbpp::Sol& sol) const noexcept {
    return static_cast<uint64_t>(
        qbpp_hash(sol.bit_vector().size64(), sol.bit_vector().bits_ptr()));
  }
};
enum class SavePolicy {
  BestSolsOnly,        
  All,                 
  WeakLocalMinimum,    
  StrictLocalMinimum,  
};
class BestSols {
  const size_t capacity_;
  SavePolicy save_policy_;
  std::vector<qbpp::Sol> sol_vector_;
  std::unordered_set<qbpp::Sol, SolHash> sol_set_;
  std::optional<energy_t> worst_energy_;
  std::mutex mutex_;
 public:
  explicit BestSols(size_t capacity = 0,
                    SavePolicy save_policy = SavePolicy::BestSolsOnly)
      : capacity_(capacity), save_policy_(save_policy) {}
  void insert_if_better(const qbpp::Sol& sol,
                        const misc::MinHeap<energy_t>& neg_set,
                        const misc::MinHeap<energy_t>& pos_set) {
    if (capacity_ == 0) return;
    if (save_policy_ == SavePolicy::WeakLocalMinimum && !neg_set.empty())
      return;
    if (save_policy_ == SavePolicy::StrictLocalMinimum &&
        pos_set.heap_size() < sol.var_count())
      return;
    if (worst_energy_ && *worst_energy_ < sol.energy()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (sol_set_.count(sol)) return;
    if (save_policy_ == SavePolicy::BestSolsOnly && worst_energy_ &&
        sol.energy() < *worst_energy_) {
      sol_vector_.clear();
    }
    if (sol_vector_.size() < capacity_) {
      sol_vector_.push_back(sol);
      sol_set_.insert(sol);
    } else {
      if (sol.energy() >= sol_vector_.back().energy()) return;
      const auto& removed = sol_vector_.back();
      sol_set_.erase(removed);
      sol_vector_.back() = sol;
      sol_set_.insert(sol);
    }
    for (size_t i = sol_vector_.size() - 1; i > 0; --i) {
      if (sol_vector_[i].energy() < sol_vector_[i - 1].energy()) {
        std::swap(sol_vector_[i], sol_vector_[i - 1]);
      } else {
        break;
      }
    }
    worst_energy_ = sol_vector_.back().energy();
  }
  const std::vector<qbpp::Sol>& sol_vector() const { return sol_vector_; }
  std::vector<qbpp::Sol>& sol_vector() { return sol_vector_; }
};
class SolDelta : public Sol {
 protected:
  const EasySolver& easy_solver_;
  const HuboModel& hubo_model_;
  std::vector<energy_t> delta_;
  misc::MinHeap<energy_t> neg_set_;
  misc::MinHeap<energy_t> pos_set_;
  misc::Tabu tabu_;
  uint32_t fail_count_ = 0;
  const std::shared_ptr<qbpp::SolHolder> all_best_sol_ptr_;
  std::shared_ptr<qbpp::SolHolder> time_limited_best_sol_ptr_;
  uint64_t flip_count_ = 0;
  const size_t thread_id_;
  const size_t thread_count_;
  const size_t max_random_flip_count_;
  static size_t compute_max_random_flip_count(size_t thread_id,
                                              size_t var_count) {
    double base = static_cast<double>(var_count) / 2 + 10;
    double scale =
        1.0 / (1.0 + static_cast<double>(thread_id) * std::sqrt(thread_id));
    return static_cast<size_t>(base * scale + 10);
  }
 public:
  SolDelta(const EasySolver& easy_solver, size_t thread_id,
           size_t thread_count);
  SolDelta& set_all_one() {
    neg_set_.clear();
    pos_set_.clear();
    for (vindex_t i = 0; i < var_count(); ++i) {
      Sol::set(i, true);
      delta_[i] = -hubo_model_.coeff_sum(i);
      if (delta_[i] < 0) {
        neg_set_.insert(i, delta_[i]);
      } else if (delta_[i] > 0) {
        pos_set_.insert(i, delta_[i]);
      }
    }
    energy_ = hubo_model_.all_coeff_sum();
    return *this;
  }
  bool reach_time_limit() const;
  bool reach_target_energy() const;
  bool reach_end_time() const;
  bool should_stop() const {
    return reach_target_energy() || reach_time_limit() || reach_end_time();
  }
  double remaining_time() const;
  void set_if_better(const std::string& info);
  void set_if_better_neighbor(const std::string& info);
  void flip(vindex_t index) { flip_with_updated_vars(index, true); }
  uint64_t flip_count() const { return flip_count_; }
  template <size_t N>
  inline void update_delta_delta(
      const TermVector<N - 1>& terms,  
      std::vector<energy_t>& delta_delta,
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
  std::vector<vindex_t> flip_with_updated_vars(
      vindex_t index, bool disable_updated_vars = false) {
    std::vector<vindex_t> updated_vars;
    updated_vars.reserve(16);
    if (index >= var_count()) {
      throw std::out_of_range(
          THROW_MESSAGE("Flip index (", index, ") is out of range."));
    }
    int32_t flip_bit_sign = 1 - 2 * get(index);
    std::vector<energy_t> delta_delta(var_count(), 0);
    const auto& t2 = hubo_model_.term2(index);
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
    Sol::flip_bit_add_delta(index, delta_[index]);
    for (vindex_t i = 0; i < var_count(); ++i) {
      if (delta_delta[i] == 0) continue;
      if (i == index)
        throw std::runtime_error(
            THROW_MESSAGE("Unexpected error. delta_delta[i] != 0"));
      if (!disable_updated_vars) {
        updated_vars.emplace_back(i);
      }
      if (delta_[i] < 0) {
        neg_set_.erase(i);
      } else if (delta_[i] > 0) {
        pos_set_.erase(i);
      }
      delta_[i] += delta_delta[i];
      if (delta_[i] < 0) {
        neg_set_.insert(i, delta_[i]);
      } else if (delta_[i] > 0) {
        pos_set_.insert(i, delta_[i]);
      }
    }
    if (delta_[index] < 0) {
      neg_set_.erase(index);
    } else if (delta_[index] > 0) {
      pos_set_.erase(index);
    }
    delta_[index] = -delta_[index];
    if (delta_[index] < 0) {
      neg_set_.insert(index, delta_[index]);
    } else if (delta_[index] > 0) {
      pos_set_.insert(index, delta_[index]);
    }
    tabu_.insert(index);
    ++flip_count_;
    return updated_vars;
  }
  uint32_t greedy() {
    uint32_t count = 0;
    while (!neg_set_.empty()) {
      if (should_stop()) {
        break;
      }
      vindex_t min_index = neg_set_.first();
      flip(min_index);
      set_if_better("Greedy");
      tabu_.insert(min_index);
      ++count;
    }
    {
      static std::mutex cout_mutex;
      std::lock_guard<std::mutex> lock(cout_mutex);
      if (qbpp::queueset_insert(bit_vector())) {
        fail_count_ = 0;
      } else {
        ++fail_count_;
      }
    }
    return count + fail_count_;
  }
  qbpp::SolHolder& sol_holder() { return *all_best_sol_ptr_; }
  void pos_min(size_t iteration) {
    for (size_t i = 0; i < iteration; ++i) {
      if (should_stop()) {
        break;
      }
      std::optional<vindex_t> flip_index = std::nullopt;
      for (size_t j = 0; j < tabu_.capacity() * 2; ++j) {
        vindex_t candidate = vindex_limit;
        if (pos_set_.empty() || neg_set_.empty()) {
          candidate = qbpp::random_gen(var_count());
        } else if (qbpp::random_gen(neg_set_.heap_size() + 1) == 0) {
          candidate = pos_set_.first();
        } else {
          candidate = neg_set_.select_at_random();
        }
        if (candidate == vindex_limit) {
          throw std::runtime_error(THROW_MESSAGE("Unexpected error."));
        }
        if (!tabu_.has(candidate)) {
          flip_index = candidate;
          break;
        }
      }
      if (!flip_index.has_value()) {
        flip_index = tabu_.non_tabu_random();
      }
      flip(flip_index.value());
      set_if_better("PosMin");
      tabu_.insert(flip_index.value());
      if (neg_set_.empty()) {
        if (qbpp::queueset_insert(bit_vector())) {
          fail_count_ = 0;
        } else {
          ++fail_count_;
        }
      }
    }
  }
  void move_to(qbpp::Sol&& destination);
  void random_flip(size_t iteration);
  void random_flip();
  energy_t comp_delta(vindex_t index) {
    energy_t current_energy = get(hubo_model_.expr());
    Sol flipped_sol = *this;
    flipped_sol.flip(index);
    energy_t flipped_energy = flipped_sol.get(hubo_model_.expr());
    return flipped_energy - current_energy;
  }
  std::string str_delta() const {
    std::ostringstream oss;
    oss << "delta: ";
    for (vindex_t i = 0; i < delta_.size(); ++i) {
      oss << "(" << str(var(i)) << ", " << delta_[i] << ") ";
    }
    return oss.str();
  }
  std::string debug_str() {
    std::ostringstream oss;
    oss << "delta debug: ";
    for (vindex_t i = 0; i < delta_.size(); ++i) {
      oss << "" << str(var(i)) << ": Computed Delta = " << comp_delta(i)
          << ", Delta = " << delta_[i] << std::endl;
      ;
    }
    return oss.str();
  }
  void check_sets() {
    for (const auto& [val, index] : neg_set_.heap()) {
      if (delta_[index] >= 0) {
        throw std::runtime_error(THROW_MESSAGE("Unexpected error."));
      }
      if (delta_[index] != val) {
        std::cout << "neg_set: index = " << index
                  << " delta = " << delta_[index] << " val = " << val
                  << std::endl;
        throw std::runtime_error(THROW_MESSAGE("Unexpected error."));
      }
    }
    for (const auto& [val, index] : pos_set_.heap()) {
      if (delta_[index] <= 0) {
        throw std::runtime_error(THROW_MESSAGE("Unexpected error."));
      }
      if (delta_[index] != val) {
        std::cout << "pos_set: index = " << index
                  << " delta = " << delta_[index] << " val = " << val
                  << std::endl;
        throw std::runtime_error(THROW_MESSAGE("Unexpected error."));
      }
    }
  }
};
class Sols : public qbpp::Sol {
  const uint64_t flip_count_;
  const std::vector<qbpp::Sol> best_sols_;
 public:
  explicit Sols(const qbpp::SolHolder& sol_holder, uint64_t flip_count,
                std::vector<qbpp::Sol> best_sols)
      : qbpp::Sol(sol_holder.sol()),
        flip_count_(flip_count),
        best_sols_(best_sols) {}
  [[nodiscard]] size_t size() const noexcept { return best_sols_.size(); }
  const std::vector<qbpp::Sol>& best_sols() const noexcept {
    return best_sols_;
  }
};
class EasySolver {
  const qbpp::HuboModel hubo_model_;
  std::optional<double> time_limit_;
  std::optional<energy_t> target_energy_;
  std::shared_ptr<qbpp::SolHolder> all_best_sol_ptr_;
  size_t thread_count_{std::thread::hardware_concurrency() < 4
                           ? 4
                           : std::thread::hardware_concurrency()};
  bool enable_default_callback_{false};
  double start_time_;
  double base_time_{0.1};
  std::shared_ptr<BestSols> best_sols_ptr_ = std::make_shared<BestSols>(0);
  uint64_t flip_count_{0};
  mutable std::mutex flip_count_mutex_;
  double end_time_;
  void single_search(size_t thread_id);
  std::shared_ptr<qbpp::SolHolder> time_limited_best_sol_ptr_;
  void time_limited_search(double time_limit, bool use_best_sol);
  void add_flip_count(uint64_t flip_count) {
    std::lock_guard<std::mutex> lock(flip_count_mutex_);
    flip_count_ += flip_count;
  }
 public:
  EasySolver(const qbpp::Model& model)
      : hubo_model_(model),
        all_best_sol_ptr_(std::make_shared<qbpp::SolHolder>(hubo_model_)),
        best_sols_ptr_(std::make_shared<BestSols>(0)) {
    if (hubo_model_.term_count() == 0) {
      throw std::runtime_error(
          "The HUBO model has no terms. Please check the model.");
    }
  }
  const qbpp::HuboModel& hubo_model() const { return hubo_model_; }
  const std::shared_ptr<qbpp::SolHolder>& sol_holder_ptr() const {
    return all_best_sol_ptr_;
  }
  const qbpp::SolHolder& sol_holder() const { return *all_best_sol_ptr_; }
  const std::shared_ptr<qbpp::SolHolder>& time_limited_best_sol_ptr() const {
    return time_limited_best_sol_ptr_;
  }
  virtual ~EasySolver() {}
  vindex_t var_count() const { return hubo_model_.var_count(); }
  void time_limit(double limit) { time_limit_ = limit; }
  void target_energy(energy_t energy) { target_energy_ = energy; }
  void base_time(double time = 0.0) { base_time_ = time; }
  bool reach_target_energy() const {
    if (!target_energy_.has_value()) {
      return false;
    }
    return all_best_sol_ptr_->energy() <= target_energy_.value();
  }
  bool reach_time_limit() const {
    if (!time_limit_.has_value()) {
      return false;
    }
    return qbpp::time() > start_time_ + time_limit_.value();
  }
  bool reach_end_time() const {
    if (base_time_ == 0.0) {
      return false;
    }
    return qbpp::time() >= end_time_;
  }
  bool should_stop() const {
    return reach_target_energy() || reach_time_limit() || reach_end_time();
  }
  double remaining_time() const {
    if (!time_limit_.has_value()) {
      return std::numeric_limits<double>::max();
    }
    return start_time_ + time_limit_.value() - qbpp::time();
  }
  const std::optional<energy_t>& target_energy() const {
    return target_energy_;
  }
  const std::optional<double>& time_limit() const { return time_limit_; }
  void thread_count(size_t count) { thread_count_ = count; }
  void enable_default_callback() { enable_default_callback_ = true; }
  uint64_t flip_count() const {
    std::lock_guard<std::mutex> lock(flip_count_mutex_);
    return flip_count_;
  }
  double tts() const { return all_best_sol_ptr_->tts(); }
  virtual void callback(const qbpp::Sol& sol, double tts,
                        std::string info) const {
    static std::mutex callback_mutex_;
    static std::optional<energy_t> prev_energy_ = std::nullopt;
    std::lock_guard<std::mutex> lock(callback_mutex_);
    if (enable_default_callback_) {
      if (!prev_energy_.has_value() || sol.energy() < prev_energy_.value()) {
        std::cout << "TTS = " << std::fixed << std::setprecision(3)
                  << std::setfill('0') << tts << "s Energy = " << sol.energy()
                  << " thread = "
                  << tbb::this_task_arena::current_thread_index() << " " << info
                  << std::endl;
        prev_energy_ = sol.energy();
      }
    }
  }
  void enable_best_sols(size_t capacity,
                        SavePolicy save_policy = SavePolicy::BestSolsOnly) {
    best_sols_ptr_ = std::make_shared<BestSols>(capacity, save_policy);
  }
  std::shared_ptr<BestSols>& best_sols_ptr() { return best_sols_ptr_; }
  const std::shared_ptr<BestSols>& best_sols_ptr() const {
    return best_sols_ptr_;
  }
  const std::vector<qbpp::Sol>& best_sols() const {
    return best_sols_ptr_->sol_vector();
  }
  Sols search(bool has_initial_sol = false);
  qbpp::Sol search(const qbpp::Sol& initial_sol) {
    all_best_sol_ptr_->sol(initial_sol);
    return search(true);
  }
};
inline void EasySolver::single_search(size_t thread_id) {
  const size_t min_flip_count = 100;
  size_t max_flip_count = (var_count() / 2);
  SolDelta sol_delta(*this, thread_id, thread_count_);
  if (thread_count_ > 1) {
    double k = std::pow(static_cast<double>(max_flip_count) / min_flip_count,
                        1.0 / static_cast<double>(thread_count_ - 1));
    max_flip_count =
        static_cast<size_t>(min_flip_count * std::pow(k, thread_id));
    if (max_flip_count < min_flip_count) max_flip_count = min_flip_count;
  }
  energy_t prev_energy = time_limited_best_sol_ptr_->energy();
  size_t random_flip_count = 1;
  if (thread_id % 2) {
    sol_delta.set_all_one();
  }
  while (!should_stop()) {
    sol_delta.random_flip(random_flip_count);
    sol_delta.greedy();
    sol_delta.move_to(*time_limited_best_sol_ptr_);
    sol_delta.greedy();
    sol_delta.random_flip(random_flip_count);
    sol_delta.greedy();
    sol_delta.pos_min(max_flip_count);
    sol_delta.greedy();
    sol_delta.random_flip(random_flip_count);
    sol_delta.greedy();
    if (prev_energy == time_limited_best_sol_ptr_->energy()) {
      random_flip_count = (random_flip_count + 1) % (max_flip_count / 2);
      if (random_flip_count == 0) random_flip_count = 1;
    } else {
      random_flip_count = 1;
    }
    prev_energy = time_limited_best_sol_ptr_->energy();
  }
  add_flip_count(sol_delta.flip_count());
}
inline void EasySolver::time_limited_search(double duration,
                                            bool use_best_sol) {
  end_time_ = qbpp::time() + duration;
  if (should_stop()) {
    return;
  }
  if (use_best_sol) {
    time_limited_best_sol_ptr_ =
        std::make_shared<qbpp::SolHolder>(all_best_sol_ptr_->sol());
  } else {
    time_limited_best_sol_ptr_ = std::make_shared<qbpp::SolHolder>(hubo_model_);
  }
  tbb::parallel_for(size_t(0), static_cast<size_t>(thread_count_),
                    [&](size_t thread_id) { single_search(thread_id); });
}
double compute_duration(size_t index, double base) {
  uint32_t k = static_cast<uint32_t>(
      std::ceil((std::sqrt(1 + 8.0 * static_cast<double>(index + 1)) - 1) / 2));
  uint32_t position = static_cast<uint32_t>(index) - k * (k - 1) / 2;
  return base * (1u << position);
}
inline Sols EasySolver::search(bool has_initial_sol) {
  if (!has_initial_sol) {
    all_best_sol_ptr_ = std::make_shared<qbpp::SolHolder>(hubo_model_);
  }
  start_time_ = qbpp::time();
  queueset_capacity(1000000);
  for (size_t i = 0;; ++i) {
    if (reach_time_limit() || reach_target_energy()) break;
    double duration = compute_duration(i, base_time_);
    time_limited_search(duration, false);
    time_limited_search(duration, true);
  }
  return Sols(*all_best_sol_ptr_, flip_count_, best_sols_ptr_->sol_vector());
}
inline SolDelta::SolDelta(const EasySolver& easy_solver, size_t thread_id,
                          size_t thread_count)
    : Sol(easy_solver.hubo_model()),
      easy_solver_(easy_solver),
      hubo_model_(easy_solver_.hubo_model()),
      delta_(var_count()),
      neg_set_(var_count()),
      pos_set_(var_count()),
      tabu_(var_count(), std::min(static_cast<vindex_t>(7), var_count() / 2)),
      all_best_sol_ptr_(easy_solver_.sol_holder_ptr()),
      time_limited_best_sol_ptr_(easy_solver_.time_limited_best_sol_ptr()),
      thread_id_(thread_id),
      thread_count_(thread_count),
      max_random_flip_count_(
          compute_max_random_flip_count(thread_id, var_count())) {
  for (vindex_t i = 0; i < var_count(); ++i) {
    delta_[i] = hubo_model_.term1(i);
    if (delta_[i] < 0) {
      neg_set_.insert(i, delta_[i]);
    } else if (delta_[i] > 0) {
      pos_set_.insert(i, delta_[i]);
    }
  }
}
inline void SolDelta::move_to(Sol&& destination) {
  misc::MinHeap to_be_flipped(var_count());
  for (vindex_t i = 0; i < var_count(); ++i) {
    if (should_stop()) {
      break;
    }
    if (get(i) != destination.get(i)) {
      to_be_flipped.insert(i, delta_[i]);
    }
  }
  while (!to_be_flipped.empty()) {
    if (should_stop()) {
      break;
    }
    vindex_t min_index = to_be_flipped.pop_first();
    std::vector<vindex_t> updated_vars = flip_with_updated_vars(min_index);
    set_if_better("MoveTo");
    for (const auto& index : updated_vars) {
      if (to_be_flipped.has(index)) {
        to_be_flipped.erase(index);
        to_be_flipped.insert(index, delta_[index]);
      }
    }
  }
}
inline void SolDelta::random_flip(size_t iteration) {
  for (vindex_t i = 0; i < iteration; ++i) {
    if (should_stop()) {
      break;
    }
    vindex_t flip_index = qbpp::random_gen(var_count());
    flip(flip_index);
    set_if_better("Random");
  }
}
inline void SolDelta::random_flip() {
  size_t iteration = (1u << (fail_count_ % 30)) % (var_count() / 2 + 2);
  for (size_t i = 0; i < iteration; ++i) {
    if (should_stop()) {
      break;
    }
    vindex_t flip_index = qbpp::random_gen(var_count());
    flip(flip_index);
    set_if_better("Random");
  }
}
inline void SolDelta::set_if_better(const std::string& info) {
  if (time_limited_best_sol_ptr_->set_if_better(*this, info)) {
    std::optional<double> tts = all_best_sol_ptr_->set_if_better(*this, info);
    if (tts.has_value()) {
      easy_solver_.callback(*this, tts.value(), info);
    }
    set_if_better_neighbor(info);
    if (easy_solver_.best_sols_ptr() != nullptr) {
      easy_solver_.best_sols_ptr()->insert_if_better(*this, neg_set_, pos_set_);
    }
  }
}
inline void SolDelta::set_if_better_neighbor(const std::string& info) {
  if (neg_set_.empty()) {
    return;
  }
  const auto& [min_delta, min_index] = neg_set_.min();
  if (energy() + min_delta < all_best_sol_ptr_->energy()) {
    Sol temp_sol = *this;
    temp_sol.flip_bit_add_delta(min_index, min_delta);
    std::optional<double> tts =
        all_best_sol_ptr_->set_if_better(temp_sol, info + "(neighbor)");
    if (tts.has_value()) {
      easy_solver_.callback(temp_sol, tts.value(), info + "(neighbor)");
    }
  }
}
inline bool SolDelta::reach_time_limit() const {
  return easy_solver_.reach_time_limit();
}
inline double SolDelta::remaining_time() const {
  return easy_solver_.remaining_time();
}
inline bool SolDelta::reach_target_energy() const {
  return easy_solver_.reach_target_energy();
}
inline bool SolDelta::reach_end_time() const {
  return easy_solver_.reach_end_time();
}
}  
}  
