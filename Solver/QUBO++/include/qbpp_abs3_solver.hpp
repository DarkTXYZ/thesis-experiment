/// @file qbpp_abs3_solver.hpp
/// @brief Header file for the ABS3 solver
/// @author Koji Nakano
/// @version 2025.07.16

#pragma once
#include "qbpp.hpp"
#define MAXPARAM 100
namespace qbpp {
namespace abs3 {
class ABS3Solver;
class Params {
  struct Pair {
    char* key;
    char* value;
  };
  Pair pairs_[MAXPARAM];
  Params(const Params&) = delete;
  Params(Params&&) = delete;
  Params& operator=(const Params&) = delete;
  Params& operator=(Params&&) = delete;
 public:
  Params() {
    for (int i = 0; i < MAXPARAM; i++) {
      pairs_[i].key = nullptr;
      pairs_[i].value = nullptr;
    }
  }
  ~Params() {
    for (int i = 0; i < MAXPARAM; i++) {
      if (pairs_[i].key != nullptr) {
        free(pairs_[i].key);
        pairs_[i].key = nullptr;
      }
      if (pairs_[i].value != nullptr) {
        free(pairs_[i].value);
        pairs_[i].value = nullptr;
      }
    }
  }
  void add(const char* key, const std::string& value) {
    add(key, value.c_str());
  }
  void add(const char* key, const char* value) {
    if (pairs_[MAXPARAM - 1].key != nullptr) {
      throw std::runtime_error("Param overflow");
    }
    for (int i = 0; i < MAXPARAM; i++) {
      if (pairs_[i].key == nullptr && pairs_[i].value == nullptr) {
        pairs_[i].key = strdup(key);
        if (pairs_[i].key == nullptr) {
          throw std::runtime_error("strdup failed");
        }
        pairs_[i].value = strdup(value);
        if (pairs_[i].value == nullptr) {
          throw std::runtime_error("strdup failed");
        }
        break;
      }
    }
  }
  const Pair pairs(size_t i) const { return pairs_[i]; }
};
class ABS3Sol {
  qbpp::vindex_t* var_count_ptr_;
  qbpp::vindex_t* size64_ptr_;
  uint64_t* id_ptr_;
  qbpp::energy_t* energy_ptr_;
  uint64_t* bitarray_;
  double* tts_ptr_;
  qbpp::vindex_t* popcount_ptr_;
 public:
  ABS3Sol(qbpp::vindex_t var_count) {
    var_count_ptr_ = new qbpp::vindex_t;
    size64_ptr_ = new qbpp::vindex_t;
    *var_count_ptr_ = var_count;
    *size64_ptr_ = (var_count + 63) / 64;
    id_ptr_ = new uint64_t;
    energy_ptr_ = new qbpp::energy_t;
    bitarray_ = new uint64_t[*size64_ptr_];
    std::fill(bitarray_, bitarray_ + *size64_ptr_, 0ULL);
    tts_ptr_ = new double;
    popcount_ptr_ = new qbpp::vindex_t;
  }
  ABS3Sol(const ABS3Sol&) = delete;
  ABS3Sol& operator=(const ABS3Sol&) = delete;
  ABS3Sol(ABS3Sol&& other) noexcept
      : var_count_ptr_(other.var_count_ptr_),
        size64_ptr_(other.size64_ptr_),
        id_ptr_(other.id_ptr_),
        energy_ptr_(other.energy_ptr_),
        bitarray_(other.bitarray_),
        tts_ptr_(other.tts_ptr_),
        popcount_ptr_(other.popcount_ptr_) {
    other.var_count_ptr_ = nullptr;
    other.size64_ptr_ = nullptr;
    other.id_ptr_ = nullptr;
    other.energy_ptr_ = nullptr;
    other.bitarray_ = nullptr;
    other.tts_ptr_ = nullptr;
    other.popcount_ptr_ = nullptr;
  }
  ABS3Sol& operator=(ABS3Sol&& other) noexcept {
    if (this != &other) {
      delete var_count_ptr_;
      delete size64_ptr_;
      delete id_ptr_;
      delete energy_ptr_;
      delete[] bitarray_;
      delete tts_ptr_;
      delete popcount_ptr_;
      id_ptr_ = other.id_ptr_;
      energy_ptr_ = other.energy_ptr_;
      bitarray_ = other.bitarray_;
      tts_ptr_ = other.tts_ptr_;
      popcount_ptr_ = other.popcount_ptr_;
      var_count_ptr_ = other.var_count_ptr_;
      size64_ptr_ = other.size64_ptr_;
      other.id_ptr_ = nullptr;
      other.energy_ptr_ = nullptr;
      other.bitarray_ = nullptr;
      other.tts_ptr_ = nullptr;
      other.popcount_ptr_ = nullptr;
      other.var_count_ptr_ = nullptr;
      other.size64_ptr_ = nullptr;
    }
    return *this;
  }
  ~ABS3Sol() {
    delete var_count_ptr_;
    delete size64_ptr_;
    delete id_ptr_;
    delete energy_ptr_;
    delete[] bitarray_;
    delete tts_ptr_;
    delete popcount_ptr_;
  }
  uint64_t& id() { return *id_ptr_; }
  const uint64_t& id() const { return *id_ptr_; }
  qbpp::energy_t& energy() { return *energy_ptr_; }
  const qbpp::energy_t& energy() const { return *energy_ptr_; }
  uint64_t* bitarray() { return bitarray_; }
  const uint64_t* bitarray() const { return bitarray_; }
  uint64_t& bitarray(size_t i) { return bitarray_[i]; }
  const uint64_t& bitarray(size_t i) const { return bitarray_[i]; }
  double& tts() { return *tts_ptr_; }
  const double& tts() const { return *tts_ptr_; }
  qbpp::vindex_t& popcount() { return *popcount_ptr_; }
  const qbpp::vindex_t& popcount() const { return *popcount_ptr_; }
  bool get(qbpp::vindex_t index) const {
    return (bitarray_[index / 64] & (1ULL << (index % 64))) != 0;
  }
  void set(qbpp::vindex_t index, bool value) {
    if (value) {
      bitarray_[index / 64] |= (1ULL << (index % 64));
    } else {
      bitarray_[index / 64] &= ~(1ULL << (index % 64));
    }
  }
  bool flip(qbpp::vindex_t index) {
    bitarray_[index / 64] ^= (1ULL << (index % 64));
    return get(index);
  }
};
}  
}  
extern "C" {
int qbpp_abs3_init();
const char* qbpp_abs3_device_prop_name(int device_id);
int qbpp_abs3_device_prop_multiProcessorCount(int device_id);
void* qbpp_abs3_solver_create(const qbpp::abs3::ABS3Solver* abs3solver_ptr,
                              const qbpp::FlatHuboModel* flat_hubo_model_ptr,
                              uint32_t device_count);
void qbpp_abs3_solver_destroy(void* solver);
void qbpp_abs3_solver_search(void* solver, qbpp::abs3::ABS3Sol* sol,
                             const qbpp::abs3::Params* params);
void qbpp_abs3_solver_callback(const qbpp::abs3::ABS3Solver* abs3solver,
                               const qbpp::abs3::ABS3Sol* sol);
}
namespace qbpp {
namespace abs3 {
int init() { return qbpp_abs3_init(); }
std::string device_prop_name(int device_id = 0) {
  return std::string(qbpp_abs3_device_prop_name(device_id));
}
int device_prop_multiProcessorCount(int device_id = 0) {
  return qbpp_abs3_device_prop_multiProcessorCount(device_id);
}
class ABS3Solver {
  qbpp::Model model_;
  const qbpp::HuboModel hubo_model_;
  const qbpp::FlatHuboModel flat_hubo_model_;
  void* pimpl_;
  ABS3Solver() = delete;
  ABS3Solver(const ABS3Solver&) = delete;
  ABS3Solver& operator=(const ABS3Solver&) = delete;
 public:
  explicit ABS3Solver(const Model& model, uint32_t device_count = 0)
      : model_(model),
        hubo_model_(HuboModel(model)),
        flat_hubo_model_(FlatHuboModel(hubo_model_)),
        pimpl_(qbpp_abs3_solver_create(static_cast<ABS3Solver*>(this),
                                       &flat_hubo_model_, device_count)) {}
  explicit ABS3Solver(const Expr& expr, uint32_t device_count)
      : ABS3Solver(Model(expr), device_count) {}
  virtual ~ABS3Solver() noexcept { qbpp_abs3_solver_destroy(pimpl_); }
  qbpp::Sol search(const Params& params) {
    ABS3Sol sol(model_.var_count());
    qbpp_abs3_solver_search(pimpl_, &sol, &params);
    return qbpp::Sol(model_, sol.energy(), sol.bitarray(), sol.tts());
  }
  virtual void callback(const ABS3Sol& sol) const {
    std::cout << "CALLBACK TTS = " << std::fixed << std::setprecision(3)
              << std::setfill('0') << sol.tts() << "s Energy = " << sol.energy()
              << std::endl;
  }
};
extern "C" {
void qbpp_abs3_solver_callback(const qbpp::abs3::ABS3Solver* abs3solver,
                               const qbpp::abs3::ABS3Sol* sol) {
  abs3solver->callback(*sol);
}
}
}  
}  
