/// @file qbpp_defs.hpp
/// @author Koji Nakano
/// @brief Header file for qbpp library definitions.
/// @version 2025.10.30
/// @details This file contains definitions and utility functions used in the
/// qbpp library.
#pragma once
#include <time.h>
#ifndef __CUDACC__
#include <boost/multiprecision/cpp_int.hpp>
#endif
#include <cstdint>
#include <iomanip>
#include <limits>
#include <string>
#ifndef VINDEX_TYPE
#define VINDEX_TYPE uint32_t
#endif
#ifndef qbpp_message
#define qbpp_message(x) x
#endif
#define LICENSE_KEY_MAX_LEN 100
namespace qbpp {
struct LicenseInfoPtrs;
struct LicenseInfo;
using vindex_t = VINDEX_TYPE;
constexpr vindex_t vindex_limit = std::numeric_limits<vindex_t>::max();
constexpr vindex_t vindex_neg_bit =
    1u << (std::numeric_limits<vindex_t>::digits - 1);
constexpr vindex_t vindex_mask = ~vindex_neg_bit;
constexpr size_t SEQ_THRESHOLD = 1000;
}  
extern "C" {
const char* qbpp_const_str(const char* param);
qbpp::vindex_t qbpp_new_var(const char* name);
const char* qbpp_var_str(qbpp::vindex_t index);
qbpp::vindex_t qbpp_var_set_size();
qbpp::vindex_t qbpp_increment_unnamed_var_count();
void* qbpp_bitvector_create(qbpp::vindex_t bit_count);
void* qbpp_bitvector_create_from_array(qbpp::vindex_t bit_count,
                                       const uint64_t* bit_array);
void* qbpp_bitvector_copy(void* src);
void qbpp_bitvector_destroy(void* bv);
void qbpp_bitvector_set(void* bv, qbpp::vindex_t index, bool value);
bool qbpp_bitvector_get(const void* bv, qbpp::vindex_t index);
void qbpp_bitvector_flip(void* bv, qbpp::vindex_t index);
uint64_t qbpp_bitvector_get64(const void* bv, qbpp::vindex_t index64);
void qbpp_bitvector_set64(void* bv, qbpp::vindex_t index64, uint64_t value);
void qbpp_bitvector_clear(void* bv);
qbpp::vindex_t qbpp_bitvector_popcount(const void* bv);
qbpp::vindex_t qbpp_bitvector_size(const void* bv);
qbpp::vindex_t qbpp_bitvector_size64(const void* bv);
bool qbpp_bitvector_equal(const void* a, const void* b);
bool qbpp_bitvector_less_than(const void* a, const void* b);
const uint64_t* qbpp_bitvector_bits_ptr(const void* bv);
char* qbpp_bitvector_str(const void* bv);
void* qbpp_iv_mapper_create(qbpp::vindex_t var_count, const uint8_t* var_flags);
void qbpp_iv_mapper_destroy(void* mapper);
qbpp::vindex_t qbpp_iv_mapper_var_count(const void* mapper);
qbpp::vindex_t qbpp_iv_mapper_index(const void* mapper, qbpp::vindex_t var);
qbpp::vindex_t qbpp_iv_mapper_var(const void* mapper, qbpp::vindex_t index);
int qbpp_iv_mapper_has(const void* mapper, qbpp::vindex_t var);
uint64_t qbpp_hash(size_t size, const uint64_t* data);
void qbpp_queueset_capacity(uint32_t capacity);
size_t qbpp_queueset_size();
uint8_t qbpp_queueset_has(const void* bv);
uint8_t qbpp_queueset_insert(const void* bv);
uint32_t qbpp_random_gen32();
uint64_t qbpp_random_gen64();
double qbpp_random_gen_double();
void qbpp_random_set_seed(uint32_t seed);
void qbpp_license_key(const char* license_key);
void qbpp_license_info(bool activate_if_needed, bool deactivate, bool details,
                       qbpp::LicenseInfoPtrs* info);
void qbpp_license_deactivate(const char* license_key);
}
namespace qbpp {
inline uint32_t random_gen32() { return qbpp_random_gen32(); }
inline uint64_t random_gen64() { return qbpp_random_gen64(); }
inline double random_gen_double() { return qbpp_random_gen_double(); }
inline void random_set_seed(uint32_t seed) { qbpp_random_set_seed(seed); }
template <typename T>
T random_gen(T n) {
  if constexpr (sizeof(T) <= 4 && std::is_integral_v<T>) {
    if (n <= 0) return 0;
    constexpr uint32_t R = std::numeric_limits<uint32_t>::max();
    uint32_t limit = (R / static_cast<uint32_t>(n)) * static_cast<uint32_t>(n);
    uint32_t r;
    do {
      r = random_gen32();
    } while (r >= limit);
    return static_cast<T>(r % static_cast<uint32_t>(n));
  }
  else if constexpr (sizeof(T) <= 8 && std::is_integral_v<T>) {
    if (n <= 0) return 0;
    constexpr uint64_t R = std::numeric_limits<uint64_t>::max();
    uint64_t limit = (R / static_cast<uint64_t>(n)) * static_cast<uint64_t>(n);
    uint64_t r;
    do {
      r = random_gen64();
    } while (r >= limit);
    return static_cast<T>(r % static_cast<uint64_t>(n));
  }
#ifdef __CUDACC__
  else if constexpr (sizeof(T) > 8) {
    static_assert(sizeof(T) == 0, "Unsupported type in random_gen");
  }
#else
  else if constexpr (boost::multiprecision::number_category<T>::value ==
                     boost::multiprecision::number_kind_integer) {
    if (n <= 0) return 0;
    size_t bitlen = boost::multiprecision::msb(n);
    T value;
    do {
      value = 0;
      size_t bits = 0;
      while (bits <= bitlen) {
        uint32_t r = random_gen32();
        for (int i = 0; i < 32 && bits <= bitlen; ++i, ++bits) {
          if (r & (1u << i)) value |= T(1) << bits;
        }
      }
    } while (value >= n);
    return value;
  } else {
    static_assert(sizeof(T) == 0, "Unsupported type in random_gen");
  }
#endif
}
enum class LicenseStatus : int32_t {
  Empty = 0,
  Error = 1,
  Unlicensed = 2,
  ExpiredAnonymousTrial = 3,
  ExpiredLicense = 4,
  AnonymousTrial = 5,
  RegisteredTrial = 6,
  StandardLicense = 7,
  ProfessionalLicense = 8,
  PartnerLicense = 9
};
struct LicenseInfoPtrs {
  LicenseStatus* license_status_ptr_ = nullptr;
  time_t* expiration_date_ptr_ = nullptr;
  size_t* max_var_count_ptr_ = nullptr;
  size_t* gpu_max_var_count_ptr_ = nullptr;
  uint32_t* allowed_activations_ptr_ = nullptr;
  uint32_t* total_activations_ptr_ = nullptr;
  uint32_t* allowed_deactivations_ptr_ = nullptr;
  uint32_t* total_deactivations_ptr_ = nullptr;
  char* license_key_masked_ptr_ = nullptr;
  LicenseInfoPtrs() {
    license_status_ptr_ = new LicenseStatus;
    expiration_date_ptr_ = new time_t;
    max_var_count_ptr_ = new size_t;
    gpu_max_var_count_ptr_ = new size_t;
    allowed_activations_ptr_ = new uint32_t;
    total_activations_ptr_ = new uint32_t;
    allowed_deactivations_ptr_ = new uint32_t;
    total_deactivations_ptr_ = new uint32_t;
    license_key_masked_ptr_ = new char[LICENSE_KEY_MAX_LEN];
    *license_status_ptr_ = LicenseStatus::Empty;
    *expiration_date_ptr_ = 0;
    *max_var_count_ptr_ = 0;
    *gpu_max_var_count_ptr_ = 0;
    *allowed_activations_ptr_ = 0;
    *total_activations_ptr_ = 0;
  }
  ~LicenseInfoPtrs() {
    delete license_status_ptr_;
    delete expiration_date_ptr_;
    delete max_var_count_ptr_;
    delete gpu_max_var_count_ptr_;
    delete allowed_activations_ptr_;
    delete total_activations_ptr_;
    delete[] license_key_masked_ptr_;
  }
  LicenseInfoPtrs& operator=(const LicenseInfo& other);
  LicenseInfoPtrs(const LicenseInfoPtrs&) = delete;
  LicenseInfoPtrs& operator=(const LicenseInfoPtrs&) = delete;
};
struct LicenseInfo {
  LicenseStatus license_status_ = LicenseStatus::Empty;
  time_t expiration_date_ = 0;
  size_t max_var_count_ = 0;
  size_t gpu_max_var_count_ = 0;
  uint32_t allowed_activations_ = 0;
  uint32_t total_activations_ = 0;
  uint32_t allowed_deactivations_ = 0;
  uint32_t total_deactivations_ = 0;
  char license_key_masked_[LICENSE_KEY_MAX_LEN] = {};
  LicenseInfo() = default;
  LicenseInfo(const LicenseInfoPtrs& ptrs)
      : license_status_(*ptrs.license_status_ptr_),
        expiration_date_(*ptrs.expiration_date_ptr_),
        max_var_count_(*ptrs.max_var_count_ptr_),
        gpu_max_var_count_(*ptrs.gpu_max_var_count_ptr_),
        allowed_activations_(*ptrs.allowed_activations_ptr_),
        total_activations_(*ptrs.total_activations_ptr_),
        allowed_deactivations_(*ptrs.allowed_deactivations_ptr_),
        total_deactivations_(*ptrs.total_deactivations_ptr_) {
    std::memcpy(license_key_masked_, ptrs.license_key_masked_ptr_,
                LICENSE_KEY_MAX_LEN);
  }
  static std::string activation_str(uint32_t num) {
    if (num == std::numeric_limits<uint32_t>::max()) {
      return qbpp_message("Unlimited");
    } else {
      return std::to_string(num);
    }
  }
  std::string str(bool details = false) const {
    std::ostringstream oss;
    oss << qbpp_message("QUBO++ (") << qbpp_const_str("version")
        << qbpp_message("): ") << license_status_str();
    if (expiration_date_ > 1) {
      switch (license_status_) {
        case LicenseStatus::ExpiredLicense:
        case LicenseStatus::ExpiredAnonymousTrial:
          oss << qbpp_message(" (expired at: ");
          break;
        default:
          oss << qbpp_message(" (expires at: ");
      }
      oss << std::put_time(std::localtime(&expiration_date_),
                           qbpp_message("%Y/%m/%d %H:%M:%S"))
          << qbpp_message(")");
    }
    if (license_status_ == LicenseStatus::PartnerLicense) {
      oss << qbpp_message(
          "\n*** For authorized partner development use only ***");
    }
    oss << qbpp_message("\nVariable Limit: ") << max_var_count_
        << qbpp_message("/") << gpu_max_var_count_
        << qbpp_message(" (CPU/GPU)");
    if (details) {
      if ((license_status_ == LicenseStatus::StandardLicense ||
           license_status_ == LicenseStatus::ProfessionalLicense ||
           license_status_ == LicenseStatus::RegisteredTrial ||
           license_status_ == LicenseStatus::PartnerLicense)) {
        oss << qbpp_message("\nLicense Key: [") << license_key_masked_
            << qbpp_message("]\n") << qbpp_message("Activations: ")
            << activation_str(total_activations_) << qbpp_message("/")
            << activation_str(allowed_activations_)
            << qbpp_message(" (used/allowed)")
            << qbpp_message("\nDeactivations: ")
            << activation_str(total_deactivations_) << qbpp_message("/")
            << activation_str(allowed_deactivations_)
            << qbpp_message(" (used/allowed)");
      }
    }
    return oss.str();
  }
  std::string license_status_str() const {
    switch (license_status_) {
      case LicenseStatus::Empty:
        return qbpp_message("Empty");
      case LicenseStatus::Error:
        return qbpp_message("Error");
      case LicenseStatus::Unlicensed:
        return qbpp_message("Unlicensed");
      case LicenseStatus::ExpiredAnonymousTrial:
        return qbpp_message("Expired Anonymous Trial");
      case LicenseStatus::ExpiredLicense:
        return qbpp_message("Expired License");
      case LicenseStatus::AnonymousTrial:
        return qbpp_message("Anonymous Trial");
      case LicenseStatus::RegisteredTrial:
        return qbpp_message("Registered Trial");
      case LicenseStatus::StandardLicense:
        return qbpp_message("Standard License");
      case LicenseStatus::ProfessionalLicense:
        return qbpp_message("Professional License");
      case LicenseStatus::PartnerLicense:
        return qbpp_message("Partner License");
      default:
        return qbpp_message("Unknown");
    }
  }
};
inline std::ostream& operator<<(std::ostream& os, const LicenseInfo& info) {
  os << info.str();
  return os;
}
inline void license_key(const char* license_key = nullptr) {
  qbpp_license_key(license_key);
}
inline void license_deactivate(const char* license_key = nullptr) {
  qbpp_license_deactivate(license_key);
}
inline LicenseInfo license_info(bool activate_if_needed = false,
                                bool deactivate = false, bool details = false,
                                const char* license_key = nullptr) {
  LicenseInfoPtrs result_ptrs;
  if (license_key != nullptr) qbpp_license_key(license_key);
  qbpp_license_info(activate_if_needed, deactivate, details, &result_ptrs);
  return LicenseInfo(result_ptrs);
}
inline LicenseInfoPtrs& LicenseInfoPtrs::operator=(const LicenseInfo& other) {
  *license_status_ptr_ = other.license_status_;
  *expiration_date_ptr_ = other.expiration_date_;
  *max_var_count_ptr_ = other.max_var_count_;
  *gpu_max_var_count_ptr_ = other.gpu_max_var_count_;
  *allowed_activations_ptr_ = other.allowed_activations_;
  *total_activations_ptr_ = other.total_activations_;
  *allowed_deactivations_ptr_ = other.allowed_deactivations_;
  *total_deactivations_ptr_ = other.total_deactivations_;
  std::memcpy(license_key_masked_ptr_, &other.license_key_masked_,
              LICENSE_KEY_MAX_LEN);
  return *this;
}
}  