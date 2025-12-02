/// @file qbpp.hpp
/// @brief QUBO++ Toolkit
/// @details This toolkit provides classes and functions for generating and
/// manipulating expressions and polynomials involving binary and spin
/// variables. The library is designed to be simple and easy to use.
/// @note A valid license is required for commercial use of this library.
/// @author Koji Nakano
/// @copyright 2025 Koji Nakano
/// @version 2025.11.22

#pragma once
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <algorithm>
#ifndef __CUDACC__
#include <boost/multiprecision/cpp_int.hpp>
#endif
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include "qbpp_defs.hpp"
namespace qbpp {
#ifndef __CUDACC__
using cpp_int = boost::multiprecision::cpp_int;
using int128_t = boost::multiprecision::int128_t;
using uint128_t = boost::multiprecision::uint128_t;
using int256_t = boost::multiprecision::int256_t;
using uint256_t = boost::multiprecision::uint256_t;
using int512_t = boost::multiprecision::int512_t;
using uint512_t = boost::multiprecision::uint512_t;
using int1024_t = boost::multiprecision::int1024_t;
using uint1024_t = boost::multiprecision::uint1024_t;
#endif
#ifdef COEFF_TYPE
using coeff_t = COEFF_TYPE;
#else
using coeff_t = int32_t;
#endif
#ifdef ENERGY_TYPE
using energy_t = ENERGY_TYPE;
#else
using energy_t = int64_t;
#endif
}  
namespace qbpp {
class Inf;
template <typename T>
class Vector;
class Var;
class Term;
class Terms;
class Expr;
class ExprExpr;
class VarIntCore;
class VarInt;
class VarOnehotCore;
class VarOnehot;
class Sol;
class Model;
class HuboModel;
template <typename T, typename U>
class SolHolderTemplate;
using QuadModel = HuboModel;
namespace impl {
struct var_hash;
struct vars_hash;
template <size_t N>
class VarArray;
class IVMapper;
class BitVector;
}  
template <typename... Args>
std::string file_line(const char* file, int line, Args&&... args) {
  std::ostringstream oss;
  oss << file << "(" << line << ") ";
  using expander = int[];
  (void)expander{0, (void(oss << std::forward<Args>(args)), 0)...};
  return oss.str();
}
#define THROW_MESSAGE(...) file_line(__FILE__, __LINE__, __VA_ARGS__)
constexpr size_t TERM_CAPACITY = 4;
#ifdef MAXDEG
using Vars = impl::VarArray<MAXDEG>;
#else
using Vars = std::vector<Var>;
#endif
using var_val_t = int8_t;
using MapList = std::list<std::pair<std::variant<Var, VarInt>, Expr>>;
using MapDict = std::unordered_map<Var, Expr, impl::var_hash>;
using VarValMap = std::unordered_map<Var, var_val_t, impl::var_hash>;
using VarExprMap = std::unordered_map<Var, Expr, impl::var_hash>;
using VarsCoeffMap = std::unordered_map<Vars, coeff_t, impl::vars_hash>;
}  
namespace qbpp {
namespace impl {
struct var_hash {
  size_t operator()(const Var var) const;
};
struct vars_hash {
  size_t operator()(const Vars& vars) const;
};
}  
}  
namespace qbpp {
Var var(const std::string& name);
Var var();
std::string str(Var var);
inline vindex_t all_var_count() { return qbpp_var_set_size(); }
inline vindex_t new_unnamed_var_index() {
  return qbpp_increment_unnamed_var_count();
}
std::string str(Var var);
std::string str(const Vars& vars);
std::string str(const Term& term);
std::string str(const Expr& expr, const std::string& prefix);
std::string str(const Model& model);
std::string str(const Sol& sol);
std::string str(const MapList& map_list);
std::string str_short(const Expr& expr);
std::ostream& operator<<(std::ostream& os, Var var);
std::ostream& operator<<(std::ostream& os, const Term& term);
std::ostream& operator<<(std::ostream& os, const Terms& terms);
std::ostream& operator<<(std::ostream& os, const Expr& expr);
std::ostream& operator<<(std::ostream& os, const Model& model);
std::ostream& operator<<(std::ostream& os, const Sol& sol);
void sort_vars_in_place(Vars& vars);
Vars sort_vars(const Vars& vars);
Vars sort_vars_as_binary(const Vars& vars);
Vars sort_vars_as_spin(const Vars& vars);
Expr simplify(const Expr& expr, Vars (*sort_vars_func)(const Vars&));
Expr simplify_as_binary(const Expr& expr);
Expr simplify_as_spin(const Expr& expr);
bool is_simplified(const Expr& expr);
bool is_binary(const Expr& expr);
template <typename T>
Vector<Expr> sqr(const Vector<T>& arg);
template <typename T>
auto sqr(const Vector<Vector<T>>& arg) -> Vector<decltype(sqr(arg[0]))>;
energy_t eval(const Expr& expr, const Sol& sol);
energy_t eval(const Expr& expr, const MapList& map_list);
Expr reduce(const Expr& expr);
Expr binary_to_spin(const Expr& expr);
Expr spin_to_binary(const Expr& expr);
template <typename T>
Vector<Expr> binary_to_spin(const Vector<T>& arg);
template <typename T>
auto binary_to_spin(const Vector<Vector<T>>& arg)
    -> Vector<decltype(binary_to_spin(arg[0]))>;
template <typename T>
Vector<Expr> spin_to_binary(const Vector<T>& arg);
template <typename T>
auto spin_to_binary(const Vector<Vector<T>>& arg)
    -> Vector<decltype(spin_to_binary(arg[0]))>;
Vector<Expr> row(const Vector<Vector<Expr>>& vec, vindex_t index);
Vector<Expr> col(const Vector<Vector<Expr>>& vec, size_t index);
double time();
template <typename T>
Expr sum(const Vector<T>& items);
template <typename T>
Vector<Vector<Expr>> transpose(const Vector<Vector<T>>& vec);
Expr operator-(const Expr& expr);
Expr operator-(Expr&& expr);
Expr&& operator-(Expr&& lhs, Expr&& rhs);
Expr sqr(const Expr& expr);
energy_t toInt(const Expr& expr);
VarValMap list_to_var_val(const MapList& map_list);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Term operator*(Var var, T val);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Term operator*(T val, Var var);
Term operator*(Var var1, Var var2);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Term operator*(const Term& term, T val);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Term operator*(T val, const Term& term);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Term operator*(Term&& term, T val);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Term operator*(T val, Term&& term);
Term operator*(const Term& term, Var var);
Term operator*(Var var, const Term& term);
Term operator*(Term&& term, Var var);
Term operator*(Var var, Term&& term);
Term operator*(const Term& term1, const Term& term2);
template <typename T>
Terms operator+(const Terms& lhs, T&& rhs);
template <typename T>
Terms operator+(Terms&& lhs, T&& rhs);
template <typename T>
Terms operator-(const Terms& lhs, T&& rhs);
template <typename T>
Terms operator-(Terms&& lhs, T&& rhs);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Terms operator*(const Terms& lhs, T rhs);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Terms operator*(Terms&& lhs, T rhs);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Terms operator*(T rhs, const Terms& lhs);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Terms operator*(T rhs, Terms&& lhs);
Terms operator*(const Terms& lhs, const Term& rhs);
Terms operator*(const Term& rhs, const Terms& lhs);
Terms operator*(Terms&& lhs, const Term& rhs);
Terms operator*(const Term& lhs, Terms&& rhs);
Terms operator*(const Terms& lhs, const Terms& rhs);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Expr operator*(const Expr& expr, T val);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Expr operator*(T al, const Expr& expr);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Expr operator*(Expr&& expr, T val);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Expr operator*(T val, Expr&& expr);
Expr operator*(const Expr& expr, const Term& term);
Expr operator*(const Term& term, const Expr& expr);
Expr operator*(const Expr& expr, Term&& term);
Expr operator*(Term&& term, const Expr& expr);
Expr&& operator*(Expr&& expr, const Term& term);
Expr&& operator*(const Term& term, Expr&& expr);
Expr&& operator*(Expr&& expr, Term&& rhs);
Expr&& operator*(Term&& rhs, Expr&& expr);
Expr&& operator*(Expr&& lhs, Expr&& rhs);
Expr&& operator*(Expr&& lhs, const Expr& rhs);
Expr&& operator*(const Expr& lhs, Expr&& rhs);
Expr operator*(const Expr& lhs, const Expr& rhs);
Expr&& operator+(Expr&& lhs, Expr&& rhs);
Expr&& operator+(Expr&& lhs, const Expr& rhs);
Expr&& operator+(const Expr& lhs, Expr&& rhs);
Expr operator+(const Expr& lhs, const Expr& rhs);
Expr&& operator-(Expr&& lhs, Expr&& rhs);
Expr&& operator-(Expr&& lhs, const Expr& rhs);
Expr&& operator-(const Expr& lhs, Expr&& rhs);
Expr operator-(const Expr& lhs, const Expr& rhs);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Expr&& operator/(Expr&& expr, T val);
template <typename T,
          typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                  int>::type = 0>
Expr operator/(const Expr& expr, T val);
Expr operator+(Expr&& expr);
const Expr& operator+(const Expr& expr);
Expr operator-(Expr&& expr);
Expr operator-(const Expr& expr);
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
Expr operator+(const Term& term, T val);
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
Expr operator+(Term&& term, T val);
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
Expr operator+(T val, Term&& term);
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
Expr operator+(T val, const Term& term);
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
Expr operator+(Var var, T val);
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
Expr operator+(T val, Var var);
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
Expr operator-(Var var, T val);
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
Expr operator-(T val, Var var);
Expr operator+(Var lhs, Var rhs);
Expr operator+(const Term& lhs, const Term& rhs);
Expr operator+(const Term& lhs, Term&& rhs);
Expr operator+(Term&& lhs, const Term& rhs);
Expr operator+(Term&& lhs, Term&& rhs);
Expr operator-(const Term& lhs, const Term& rhs);
Expr operator-(const Term& lhs, Term&& rhs);
Expr operator-(Term&& lhs, const Term& rhs);
Expr operator-(Term&& lhs, Term&& rhs);
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
Expr operator-(const Expr& expr, T val);
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
Expr operator-(Expr&& expr, T val);
Expr operator+(const Expr& expr, const Term& term);
Expr operator+(Expr&& expr, const Term& term);
Expr operator+(const Expr& expr, Term&& term);
Expr operator+(Expr&& expr, Term&& term);
Expr operator+(const Term& term, const Expr& expr);
Expr operator+(const Term& term, Expr&& expr);
Expr operator+(Term&& term, const Expr& expr);
Expr operator+(Term&& term, Expr&& expr);
Expr operator-(const Term& term, const Expr& expr);
Expr operator-(Term&& term, const Expr& expr);
Expr operator-(const Term& term, Expr&& expr);
Expr operator-(Term&& term, Expr&& expr);
Expr operator-(const Expr& expr, const Term& term);
Expr operator-(Expr&& expr, const Term& term);
Expr operator-(const Expr& expr, Term&& term);
Expr operator-(Expr&& expr, Term&& term);
Expr replace(const Term& term, const MapDict& map_dict);
inline std::string str(const std::string& param) {
  if (param == "all_vars") {
    std::stringstream ss;
    vindex_t var_set_size = qbpp_var_set_size();
    ss << "{";
    for (vindex_t i = 0; i < var_set_size; i++) {
      ss << "{" << i << "," << qbpp_var_str(i) << "}";
      if (i < var_set_size - 1) {
        ss << ",";
      }
    }
    ss << "}";
    return ss.str();
  } else {
    return qbpp_const_str(param.c_str());
  }
}
inline std::string str(int8_t val) {
  return std::to_string(static_cast<int32_t>(val));
}
template <typename T>
auto str(const T& val) -> decltype(std::to_string(val)) {
  return std::to_string(val);
}
template <typename T>
auto str(T val) -> decltype(val.str()) {
  return val.str();
}
template <typename T>
T abs(T val) {
  return val < 0 ? -val : val;
}
inline energy_t gcd(energy_t a, energy_t b) {
  a = abs(a);
  b = abs(b);
  if (a == 0) return b;
  if (b == 0) return a;
  while (b != 0) {
    energy_t r = a % b;
    a = b;
    b = r;
  }
  return a;
}
class Inf {
  bool positive_ = true;
 public:
  Inf() = default;
  bool is_positive() const { return positive_; }
  bool is_negative() const { return !positive_; }
  Inf operator+() const { return *this; }
  Inf operator-() const {
    Inf inf;
    inf.positive_ = !positive_;
    return inf;
  }
};
const Inf inf;
template <typename T>
class Vector {
  std::vector<T> data_;
  template <typename U, typename Op>
  Vector<T>& vector_operation(const Vector<U>& rhs, Op operation) {
    if (size() != rhs.size()) {
      throw std::out_of_range(
          THROW_MESSAGE("Vector size mismatch: ", size(), " != ", rhs.size()));
    }
    tbb::parallel_for(size_t(0), size(), [&](size_t i) {
      operation((*this)[i], rhs[i]);
      (*this)[i] = static_cast<T>((*this)[i]);
    });
    return *this;
  }
  template <typename U, typename Op>
  Vector<T>& vector_operation(Vector<U>&& rhs, Op operation) {
    if (size() != rhs.size()) {
      throw std::out_of_range(
          THROW_MESSAGE("Vector size mismatch: ", size(), " != ", rhs.size()));
    }
    tbb::parallel_for(size_t(0), size(), [&](size_t i) {
      operation((*this)[i], std::move(rhs[i]));
      (*this)[i] = static_cast<T>((*this)[i]);
    });
    return *this;
  }
  template <typename Op>
  Vector<T>& vector_operation(const Expr& rhs, Op operation) {
    tbb::parallel_for(size_t(0), size(),
                      [&](size_t i) { operation((*this)[i], rhs); });
    return *this;
  }
 public:
  Vector() = default;
  Vector(const Vector<T>&) = default;
  Vector(Vector<T>&&) = default;
  Vector(std::initializer_list<T> init) : data_(init) {}
  Vector& operator=(const Vector<T>&) = default;
  Vector& operator=(Vector<T>&&) = default;
  explicit Vector(size_t size) : data_(size) {}
  template <typename U, typename = typename std::enable_if<
                            std::is_integral<U>::value>::type>
  Vector(U size, const T& value) : data_(static_cast<size_t>(size), value) {}
  template <typename U, typename = typename std::enable_if<
                            !std::is_integral<U>::value>::type>
  Vector(U begin, U end) : data_(begin, end) {}
  template <typename U>
  Vector(const Vector<U>& rhs) {
    data_.resize(rhs.size());
    tbb::parallel_for(size_t(0), rhs.size(),
                      [&](size_t i) { data_[i] = rhs[i]; });
  }
  const std::vector<T> data() const { return data_; }
  std::vector<T> data() { return data_; }
  void resize(size_t size) { data_.resize(size); }
  void push_back(const T& t) { data_.push_back(t); }
  void emplace_back(T&& t) { data_.emplace_back(std::forward<T>(t)); }
  void reserve(size_t size) { data_.reserve(size); }
  T& operator[](size_t i) { return data_[i]; }
  const T& operator[](size_t i) const { return data_[i]; }
  size_t size() const { return data_.size(); }
  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }
  auto empty() const { return data_.empty(); }
  typename std::vector<T>::iterator erase(
      typename std::vector<T>::iterator pos) {
    return data_.erase(pos);
  }
  typename std::vector<T>::iterator erase(
      typename std::vector<T>::iterator first,
      typename std::vector<T>::iterator last) {
    return data_.erase(first, last);
  }
  Vector<T>& operator=(const Expr& rhs) {
    tbb::parallel_for(size_t(0), size(), [&](size_t i) { (*this)[i] = rhs; });
    return *this;
  }
  Vector<T>& operator=(energy_t rhs) {
    tbb::parallel_for(size_t(0), size(), [&](size_t i) { (*this)[i] = rhs; });
    return *this;
  }
  template <typename U>
  Vector<T>& operator+=(const Vector<U>& rhs) {
    return vector_operation(rhs, [](T& larg, const U& rarg) { larg += rarg; });
  }
  template <typename U>
  Vector<T>& operator+=(Vector<U>&& rhs) {
    return vector_operation(std::move(rhs),
                            [](T& lhs_elem, U rhs_elem) mutable {
                              lhs_elem += std::move(rhs_elem);
                            });
  }
  template <typename U>
  Vector<T>& operator-=(const Vector<U>& rhs) {
    return vector_operation(rhs, [](T& larg, const U& rarg) { larg -= rarg; });
  }
  template <typename U>
  Vector<T>& operator-=(Vector<U>&& rhs) {
    return vector_operation(std::move(rhs),
                            [](T& lhs_elem, U rhs_elem) mutable {
                              lhs_elem -= std::move(rhs_elem);
                            });
  }
  template <typename U>
  Vector<T>& operator*=(const Vector<U>& rhs) {
    return vector_operation(rhs, [](T& larg, const U& rarg) { larg *= rarg; });
  }
  template <typename U>
  Vector<T>& operator*=(Vector<U>&& rhs) {
    return vector_operation(std::move(rhs),
                            [](T& lhs_elem, U rhs_elem) mutable {
                              lhs_elem *= std::move(rhs_elem);
                            });
  }
  Vector<T>& operator+=(const Expr& expr) {
    return vector_operation(expr,
                            [](T& elem, const Expr& rhs) { elem += rhs; });
  }
  Vector<T>& operator-=(const Expr& expr) {
    return vector_operation(expr,
                            [](T& elem, const Expr& rhs) { elem -= rhs; });
  }
  Vector<T>& operator*=(const Expr& expr) {
    return vector_operation(expr,
                            [](T& elem, const Expr& rhs) { elem *= rhs; });
  }
  template <class U, typename std::enable_if<
                         std::is_convertible<U, coeff_t>::value, int>::type = 0>
  Vector<T>& operator/=(U val) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, size()),
                      [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t i = range.begin(); i < range.end(); ++i) {
                          (*this)[i] /= val;
                        }
                      });
    return *this;
  }
  Vector<T>& sqr() {
    *this *= *this;
    return *this;
  }
  Vector<T>& simplify(Vars (*sort_vars_func)(const Vars&) = sort_vars);
  Vector<T>& simplify_as_binary() {
    *this = simplify(sort_vars_as_binary);
    return *this;
  }
  Vector<T>& simplify_as_spin() {
    *this = simplify(sort_vars_as_spin);
    return *this;
  }
  Vector<T>& binary_to_spin() {
    *this = qbpp::binary_to_spin(*this);
    return *this;
  }
  Vector<T>& spin_to_binary() {
    *this = qbpp::spin_to_binary(*this);
    return *this;
  }
  Vector<T>& replace(const MapList& map_list);
  Vector<T>& reduce();
  Vector<T>& transpose() {
    *this = transpose(*this);
    return *this;
  }
};
class Var {
  vindex_t index_;
 public:
  Var() = default;
  explicit Var(vindex_t index) : index_(index) {}
  vindex_t index() const { return index_; }
  bool operator==(Var var) const { return index_ == var.index_; }
  bool operator!=(Var var) const { return index_ != var.index_; }
  bool operator<(Var var) const {
    vindex_t masked_index_ = index_ & qbpp::vindex_mask;
    vindex_t masked_var_index_ = var.index() & qbpp::vindex_mask;
    if (masked_index_ == masked_var_index_) {
      return index_ < var.index_;
    } else {
      return masked_index_ < masked_var_index_;
    }
  }
  bool operator>(Var var) const {
    vindex_t masked_index_ = index_ & qbpp::vindex_mask;
    vindex_t masked_var_index_ = var.index() & qbpp::vindex_mask;
    if (masked_index_ == masked_var_index_) {
      return index_ > var.index_;
    } else {
      return masked_index_ > masked_var_index_;
    }
  }
  bool operator<=(Var var) const {
    vindex_t masked_index_ = index_ & qbpp::vindex_mask;
    vindex_t masked_var_index_ = var.index() & qbpp::vindex_mask;
    if (masked_index_ == masked_var_index_) {
      return index_ <= var.index_;
    } else {
      return masked_index_ <= masked_var_index_;
    }
  }
  size_t size() const { return 1; }
  var_val_t operator()(Sol& sol) const;
};  
namespace impl {
inline const Var VarVoid{qbpp::vindex_limit};
template <std::size_t N>
class VarArray {
  Var vars_[N];
 public:
  constexpr VarArray() : vars_{} {}
  struct EMPTY {};
  constexpr VarArray(EMPTY) {
    for (std::size_t i = 0; i < N; ++i) vars_[i] = VarVoid;
  }
  constexpr explicit VarArray(Var var) : vars_{} {
    vars_[0] = var;
    for (std::size_t i = 1; i < N; ++i) vars_[i] = VarVoid;
  }
  constexpr explicit VarArray(Var var0, Var var1) : vars_{} {
    vars_[0] = var0;
    vars_[1] = var1;
    for (std::size_t i = 2; i < N; ++i) vars_[i] = VarVoid;
  }
  constexpr VarArray(std::initializer_list<Var> init) : vars_{} {
    std::size_t i = 0;
    for (Var v : init) {
      if (i >= N) break;
      vars_[i++] = v;
    }
    for (; i < N; ++i) vars_[i] = VarVoid;
  }
  void push_back(Var var) {
    for (std::size_t i = 0; i < N; ++i) {
      if (vars_[i] == VarVoid) {
        vars_[i] = var;
        return;
      }
    }
    throw std::out_of_range("VarArray is full");
  }
  std::size_t size() const {
    for (std::size_t i = 0; i < N; ++i)
      if (vars_[i] == VarVoid) return i;
    return N;
  }
  bool empty() const { return vars_[0] == VarVoid; }
  Var& operator[](std::size_t i) { return vars_[i]; }
  const Var& operator[](std::size_t i) const { return vars_[i]; }
  Var* begin() { return vars_; }
  Var* end() { return vars_ + size(); }
  const Var* begin() const { return vars_; }
  const Var* end() const { return vars_ + size(); }
  void clear() {
    for (std::size_t i = 0; i < N; ++i) vars_[i] = VarVoid;
  }
  Var& back() {
    for (std::size_t i = N; i-- > 0;) {
      if (vars_[i] != VarVoid) return vars_[i];
    }
    return vars_[0];  
  }
  void pop_back() {
    for (std::size_t i = N; i-- > 0;) {
      if (vars_[i] != VarVoid) {
        vars_[i] = VarVoid;
        return;
      }
    }
  }
  bool operator==(const VarArray& other) const {
    for (std::size_t i = 0; i < N; ++i)
      if (vars_[i] != other.vars_[i]) return false;
    return true;
  }
  bool operator<(const VarArray& other) const {
    for (std::size_t i = 0; i < N; ++i)
      if (vars_[i] != other.vars_[i]) return vars_[i] < other.vars_[i];
    return false;
  }
  VarArray& operator*=(const VarArray& vars) {
    std::size_t i = 0;
    for (; i < N && vars_[i] != VarVoid; ++i)
      ;
    for (std::size_t j = 0; j < N; ++j) {
      if (vars.vars_[j] == VarVoid) break;
      if (i >= N) {
        throw std::out_of_range("VarArray is full");
      }
      vars_[i++] = vars.vars_[j];
    }
    return *this;
  }
};
}  
class IVMapper {
  void* impl_;
 public:
  explicit IVMapper(const Expr& expr);
  ~IVMapper() {
    if (impl_) {
      qbpp_iv_mapper_destroy(impl_);
    }
  }
  vindex_t var_count() const { return qbpp_iv_mapper_var_count(impl_); }
  vindex_t index(Var var) const {
    return qbpp_iv_mapper_index(impl_, var.index());
  }
  Var var(vindex_t index) const {
    return Var(qbpp_iv_mapper_var(impl_, index));
  }
  bool has(Var var) const {
    return qbpp_iv_mapper_has(impl_, var.index()) != 0;
  }
};
class Term {
  coeff_t coeff_{1};
  Vars vars_;
 public:
  Term() = default;
  Term(const Term&) = default;
  Term(Term&&) noexcept = default;
#ifdef MAXDEG
  template <typename T,
            typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                    int>::type = 0>
  explicit Term(T val)
      : coeff_{static_cast<coeff_t>(val)}, vars_(Vars::EMPTY{}) {}
#else
  template <typename T,
            typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                    int>::type = 0>
  explicit Term(T val) : coeff_{static_cast<coeff_t>(val)} {}
#endif
  explicit Term(Var var) : vars_({var}) {}
  template <typename T,
            typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                    int>::type = 0>
#ifdef MAXDEG
  explicit Term(Var var, T val)
      : coeff_{static_cast<coeff_t>(val)}, vars_(var) {
  }
#else
  explicit Term(Var var, T val)
      : coeff_{static_cast<coeff_t>(val)}, vars_({var}) {
  }
#endif
#if MAXDEG
  explicit Term(Var var1, Var var2) : vars_(var1, var2) {}
#else
  explicit Term(Var var1, Var var2) : vars_({var1, var2}) {}
#endif
  explicit Term(const Vars& vars) : vars_(vars) {}
  explicit Term(Vars&& vars) noexcept : vars_(std::move(vars)) {}
  template <typename T,
            typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                    int>::type = 0>
  explicit Term(const Vars& vars, T val)
      : coeff_(static_cast<coeff_t>(val)), vars_{vars} {}
  template <typename T,
            typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                    int>::type = 0>
  explicit Term(Vars&& vars, T val)
      : coeff_(static_cast<coeff_t>(val)), vars_(std::move(vars)) {}
  Term& operator=(const Term& arg) {
    coeff_ = arg.coeff_;
    vars_ = arg.vars_;
    return *this;
  }
  Term& operator=(Term&& arg) noexcept {
    coeff_ = arg.coeff_;
    vars_ = std::move(arg.vars_);
    return *this;
  }
  coeff_t coeff() const { return coeff_; }
  coeff_t& coeff() { return coeff_; }
  Vars& vars() { return vars_; }
  const Vars& vars() const { return vars_; }
  const Var vars(vindex_t i) const { return vars_[i]; }
  Var& vars(vindex_t i) { return vars_[i]; }
  vindex_t var_count() const { return static_cast<vindex_t>(vars_.size()); }
  Var var(vindex_t i) const { return vars_[i]; }
  Var& var(vindex_t i) { return vars_[i]; }
  bool operator==(const Term& term) const {
    return (coeff_ == term.coeff_) && (vars_ == term.vars_);
  }
  bool operator!=(const Term& term) const { return !(*this == term); }
  Term& negate() {
    coeff_ = -coeff_;
    return *this;
  }
  bool operator<(const Term& term) const {
    if (var_count() < term.var_count())
      return true;
    else if (var_count() > term.var_count())
      return false;
    return vars_ < term.vars_;
  }
  bool operator<=(const Term& term) const {
    return *this < term || *this == term;
  }
  template <typename T,
            typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                    int>::type = 0>
  Term& operator*=(T val) {
    coeff_ *= static_cast<coeff_t>(val);
    return *this;
  }
  Term& operator*=(Var var) {
    vars_.push_back(var);
    return *this;
  }
  Term& operator*=(const Term& term) {
    coeff_ *= term.coeff_;
#ifdef MAXDEG
    vars_ *= term.vars_;
#else
    vars_.reserve(var_count() + term.var_count());
    vars_.insert(vars_.end(), term.vars_.begin(), term.vars_.end());
#endif
    return *this;
  }
  template <typename T,
            typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                    int>::type = 0>
  Term& operator/=(T val) {
    if constexpr (std::is_integral<coeff_t>::value) {
      if ((coeff_ / val) * val != coeff_) {
        throw std::runtime_error(
            THROW_MESSAGE("Indivisible division occurred."));
      }
    }
    coeff_ /= val;
    return *this;
  }
  Term& operator-() && {
    coeff_ = -coeff_;
    return *this;
  }
  Term operator-() const& {
    Term result = *this;
    result.coeff_ = -result.coeff_;
    return result;
  }
};
class Terms {
  std::vector<Term> terms_;
 public:
  Terms() { terms_.reserve(TERM_CAPACITY); };
  explicit Terms(std::initializer_list<Term> init) : Terms() { terms_ = init; }
  explicit Terms(Var var) : Terms() { terms_.push_back(Term(var)); }
  template <typename T,
            typename std::enable_if<std::is_convertible<T, energy_t>::value,
                                    int>::type = 0>
  explicit Terms(Var var, T val) : Terms() {
    terms_.push_back(Term{var, val});
  }
  explicit Terms(const Term& term) : Terms() { terms_.push_back(term); }
  explicit Terms(Term&& term) : Terms() { terms_.push_back(std::move(term)); }
  explicit Terms(const Term& term1, const Term& term2) : Terms() {
    terms_.push_back(term1);
    terms_.push_back(term2);
  }
  explicit Terms(const Term& term1, Term&& term2) : Terms() {
    terms_.push_back(term1);
    terms_.push_back(std::move(term2));
  }
  explicit Terms(Term&& term1, const Term& term2) : Terms() {
    terms_.push_back(std::move(term1));
    terms_.push_back(term2);
  }
  explicit Terms(Term&& term1, Term&& term2) : Terms() {
    terms_.push_back(std::move(term1));
    terms_.push_back(std::move(term2));
  }
  size_t size() const { return terms_.size(); }
  Term& operator[](size_t i) { return terms_[i]; }
  const Term& operator[](size_t i) const { return terms_[i]; }
  auto begin() { return terms_.begin(); }
  auto end() { return terms_.end(); }
  auto begin() const { return terms_.begin(); }
  auto end() const { return terms_.end(); }
  Term& back() { return terms_.back(); }
  const Term& back() const { return terms_.back(); }
  void push_back(const Term& term) { terms_.push_back(term); }
  void push_back(Term&& term) { terms_.push_back(std::move(term)); }
  void pop_back() { terms_.pop_back(); }
  Term& front() { return terms_.front(); }
  const Term& front() const { return terms_.front(); }
  template <typename... Args>
  Term& emplace_back(Args&&... args) {
    return terms_.emplace_back(std::forward<Args>(args)...);
  }
  std::vector<Term>::iterator erase(std::vector<Term>::iterator pos) {
    return terms_.erase(pos);
  }
  std::vector<Term>::iterator erase(std::vector<Term>::iterator first,
                                    std::vector<Term>::iterator last) {
    return terms_.erase(first, last);
  }
  bool empty() const { return terms_.empty(); }
  void reserve(size_t n) { terms_.reserve(n); }
  void clear() { terms_.clear(); }
  void resize(size_t n) { terms_.resize(n); }
  void resize(size_t n, const Term& value) { terms_.resize(n, value); }
  std::vector<Term>::iterator insert(std::vector<Term>::iterator pos,
                                     const Term& value) {
    return terms_.insert(pos, value);
  }
  std::vector<Term>::iterator insert(std::vector<Term>::iterator pos,
                                     Term&& value) {
    return terms_.insert(pos, std::move(value));
  }
  template <typename InputIt>
  std::vector<Term>::iterator insert(std::vector<Term>::iterator pos,
                                     InputIt first, InputIt last) {
    return terms_.insert(pos, first, last);
  }
  Terms& negate() {
    if (size() < SEQ_THRESHOLD) {
      for (auto& t : terms_) {
        t.negate();
      }
      return *this;
    }
    tbb::parallel_for(size_t(0), size(), [&](size_t i) { terms_[i].negate(); });
    return *this;
  }
  Terms& operator+=(const Term& term) {
    if (term.coeff() != 0) {
      push_back(term);
    }
    return *this;
  }
  Terms& operator+=(Term&& term) {
    if (term.coeff() != 0) {
      push_back(std::move(term));
    }
    return *this;
  }
  Terms& operator+=(const Terms& terms) {
    if (terms.size() < SEQ_THRESHOLD) {
      terms_.insert(terms_.end(), terms.begin(), terms.end());
      return *this;
    } else {
      size_t old_size = terms_.size();
      terms_.resize(old_size + terms.size());
      tbb::parallel_for(size_t(0), terms.size(),
                        [&](size_t i) { terms_[old_size + i] = terms[i]; });
      return *this;
    }
  }
  Terms& operator+=(Terms&& terms) {
    if (terms.size() < SEQ_THRESHOLD) {
      std::move(terms.begin(), terms.end(), std::back_inserter(terms_));
    } else {
      size_t old_size = terms_.size();
      terms_.resize(old_size + terms.size());
      tbb::parallel_for(size_t(0), terms.size(), [&](size_t i) {
        terms_[old_size + i] = std::move(terms[i]);
      });
    }
    return *this;
  }
  Terms& operator-=(const Term& term) {
    push_back(-term);
    return *this;
  }
  Terms& operator*=(coeff_t val) {
    if (val == 0) {
      terms_.clear();
      return *this;
    }
    if (val == 1) {
      return *this;
    }
    if (val == -1) {
      for (auto& t : terms_) {
        t.negate();
      }
      return *this;
    }
    for (auto& t : terms_) {
      t *= val;
    }
    return *this;
  }
  Terms& operator*=(const Term& term) {
    if (term.var_count() == 0) {
      *this *= term.coeff();
      return *this;
    }
    for (auto& t : terms_) {
      t *= term;
    }
    return *this;
  }
  template <typename T,
            typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                    int>::type = 0>
  Terms& operator/=(T val) {
    if (val == 1) {
      return *this;
    }
    for (auto& t : terms_) {
      t /= val;
    }
    return *this;
  }
};
class Expr {
  energy_t constant_{0};
  Terms terms_;
  explicit Expr(bool b) = delete;
 public:
  Expr() { terms_.reserve(TERM_CAPACITY); };
  Expr(const Expr& expr) = default;
  Expr(Expr&& expr) noexcept = default;
  template <typename T,
            typename std::enable_if<std::is_convertible<T, energy_t>::value,
                                    int>::type = 0>
  Expr(T value) {
    constant_ = static_cast<energy_t>(value);
    terms_.reserve(TERM_CAPACITY);
  }
  Expr(Var var) : terms_{var} {}
  template <typename T,
            typename std::enable_if<std::is_convertible<T, energy_t>::value,
                                    int>::type = 0>
  Expr(Var var, T val) : terms_{var, val} {}
  explicit Expr(const Term& term) : terms_{term} {}
  explicit Expr(Term&& term) : terms_{std::move(term)} {}
  template <typename T,
            typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
  explicit Expr(const Term& term, T constant = 0)
      : constant_{constant}, terms_{term} {}
  template <typename T,
            typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
  explicit Expr(Term&& term, T constant = 0)
      : constant_{constant}, terms_{std::move(term)} {}
  explicit Expr(const Term& term1, const Term& term2) : terms_{term1, term2} {}
  explicit Expr(const Term& term1, Term&& term2)
      : terms_{term1, std::move(term2)} {}
  explicit Expr(Term&& term1, const Term& term2)
      : terms_{std::move(term1), term2} {}
  explicit Expr(Term&& term1, Term&& term2)
      : terms_{std::move(term1), std::move(term2)} {}
  explicit Expr(energy_t constant, Terms&& terms) noexcept
      : constant_(constant), terms_(std::move(terms)) {}
  explicit Expr(energy_t constant, const Terms& terms) noexcept
      : constant_(constant), terms_(terms) {}
  Expr(const VarInt& var_int);
  Expr(VarInt&& var_int);
  explicit Expr(Var var, energy_t constant) noexcept : constant_(constant) {
    terms_.emplace_back(Term(var));
  }
  Expr& operator=(const Expr& expr) {
    if (this != &expr) {
      constant_ = expr.constant_;
      terms_ = expr.terms_;
    }
    return *this;
  }
  Expr& operator=(Expr&& expr) noexcept {
    if (this != &expr) {
      constant_ = expr.constant_;
      terms_ = std::move(expr.terms_);
    }
    return *this;
  }
  template <typename T,
            typename std::enable_if<std::is_convertible<T, energy_t>::value,
                                    int>::type = 0>
  Expr& operator*=(T val) {
    constant_ *= static_cast<energy_t>(val);
    terms_ *= static_cast<coeff_t>(val);
    return *this;
  }
  Expr& operator*=(const Term& term) {
    if (terms_.size() < SEQ_THRESHOLD) {
      for (auto& t : terms_) {
        t *= term;
      }
    } else {
      tbb::parallel_for(size_t(0), terms_.size(),
                        [&](size_t i) { terms_[i] *= term; });
    }
    if (constant_ != 0) {
      *this += term * static_cast<coeff_t>(constant_);
      constant_ = 0;
    }
    return *this;
  }
  Expr& operator+=(const Terms& terms) {
    terms_ += terms;
    return *this;
  }
  Expr& operator+=(Terms&& terms) {
    terms_ += std::move(terms);
    return *this;
  }
  Expr& operator+=(const Term& term) {
    terms_ += term;
    return *this;
  }
  Expr& operator+=(Term&& term) {
    terms_ += std::move(term);
    return *this;
  }
  Expr& operator-=(const Term& term) {
    terms_ -= term;
    return *this;
  }
  Expr& operator-=(Term&& term) {
    terms_ -= std::move(term);
    return *this;
  }
  Expr& operator*=(const Expr& expr) {
    if (expr.term_count() == 0) {
      return *this *= expr.constant_;
    }
    if (expr.term_count() == 1) {
      terms_ =
          terms_ * expr.terms_[0] + terms_ * expr.constant_ +
          expr.terms_ * constant_;
      constant_ *= expr.constant_;
      return *this;
    }
    Expr result{constant_ * expr.constant_};
    result += terms_ * expr.terms_;
    result += terms_ * static_cast<coeff_t>(expr.constant_);
    result += expr.terms_ * static_cast<coeff_t>(constant_);
    *this = std::move(result);
    return *this;
  }
  Expr& operator+=(const Expr& expr) {
    constant_ += expr.constant_;
    auto expr2 = expr;
    terms_.insert(terms_.end(), expr2.terms_.begin(), expr2.terms_.end());
    return *this;
  }
  Expr& operator-=(const Expr& expr) {
    auto expr2 = -expr;
    *this += std::move(expr2);
    return *this;
  }
  template <typename T,
            typename std::enable_if<std::is_convertible<T, coeff_t>::value,
                                    int>::type = 0>
  Expr& operator/=(T val) {
    if (val == 0) {
      throw std::runtime_error(THROW_MESSAGE("Division by zero."));
    }
    tbb::parallel_for(size_t(0), terms_.size(),
                      [&](size_t i) { terms_[i] /= val; });
    if ((constant_ / val) * val != constant_) {
      throw std::runtime_error(THROW_MESSAGE("Indivisible division occurred."));
    }
    constant_ /= val;
    return *this;
  }
  template <typename T,
            typename std::enable_if<std::is_convertible<T, energy_t>::value,
                                    int>::type = 0>
  Expr& operator+=(T val) {
    constant_ += static_cast<energy_t>(val);
    return *this;
  }
  template <typename T,
            typename std::enable_if<std::is_convertible<T, energy_t>::value,
                                    int>::type = 0>
  Expr& operator-=(T val) {
    constant_ -= static_cast<energy_t>(val);
    return *this;
  }
  Expr& sqr() {
    *this *= *this;
    return *this;
  }
  Expr& simplify(Vars (*sort_vars)(const Vars&) = qbpp::sort_vars);
  Expr& simplify_as_binary();
  Expr& simplify_as_spin();
  energy_t constant() const { return constant_; }
  energy_t& constant() { return constant_; }
  const Terms& terms() const { return terms_; }
  Terms& terms() { return terms_; }
  Term& term(size_t i) { return terms_[i]; }
  const Term& term(size_t i) const { return terms_[i]; }
  size_t term_count() const { return terms_.size(); }
  size_t size() const { return term_count(); }
  Expr& negate() {
    terms_.negate();
    constant_ = -constant_;
    return *this;
  }
  Expr& replace(const MapList& map_list);
  energy_t operator()(const MapList& map_list) const {
    return qbpp::eval(*this, map_list);
  }
  energy_t operator()(const Sol& sol) const { return qbpp::eval(*this, sol); }
  Expr& reduce() { return *this = qbpp::reduce(*this); }
  Expr& binary_to_spin() { return *this = qbpp::binary_to_spin(*this); }
  Expr& spin_to_binary() { return *this = qbpp::spin_to_binary(*this); }
  energy_t pos_sum() const {
    Expr temp = qbpp::simplify_as_binary(*this);
    energy_t sum = temp.constant_;
    for (const auto& term : temp.terms_) {
      if (term.coeff() > 0) {
        sum += term.coeff();
      }
    }
    return sum;
  }
  energy_t neg_sum() const {
    Expr temp = qbpp::simplify_as_binary(*this);
    energy_t sum = temp.constant_;
    for (const auto& term : temp.terms_) {
      if (term.coeff() < 0) {
        sum += term.coeff();
      }
    }
    return sum;
  }
  vindex_t var_count() const;
  size_t term_count(vindex_t size) const {
    return static_cast<size_t>(std::count_if(
        terms_.begin(), terms_.end(),
        [size](const Term& term) { return term.var_count() == size; }));
  }
};
class ExprExpr : public Expr {
  Expr expr2_;
 public:
  template <typename T1, typename T2>
  ExprExpr(T1&& expr1, T2&& expr2)
      : Expr(std::forward<T1>(expr1)), expr2_(std::forward<T2>(expr2)) {}
  ExprExpr() = default;
  ExprExpr(const Expr& expr) : Expr(expr) {}
  Expr& operator*() { return expr2_; }
  const Expr& operator*() const { return expr2_; }
  Expr* operator->() { return &expr2_; }
  const Expr* operator->() const { return &expr2_; }
};
class VarIntCore {
 public:
  const std::string var_str_;
  explicit VarIntCore(const std::string& var_str) : var_str_(var_str) {}
  VarIntCore()
      : var_str_("{" + std::to_string(new_unnamed_var_index()) + "}") {}
};
class VarInt : public Expr {
  const std::string var_str_;
  const energy_t min_val_;
  const energy_t max_val_;
  const std::shared_ptr<std::vector<coeff_t>> coeffs_ptr_;
  const Vector<Var> vars_;
  VarInt(
      const std::tuple<Expr, std::string, energy_t, energy_t,
                       std::shared_ptr<std::vector<coeff_t>>, Vector<Var>>& t)
      : Expr(std::get<0>(t)),
        var_str_(std::get<1>(t)),
        min_val_(std::get<2>(t)),
        max_val_(std::get<3>(t)),
        coeffs_ptr_(std::get<4>(t)),
        vars_(std::get<5>(t)) {}
  std::tuple<Expr, std::string, energy_t, energy_t,
             std::shared_ptr<std::vector<coeff_t>>, Vector<Var>>
  helper_func(const std::string& var_str, energy_t min_val, energy_t max_val,
              std::shared_ptr<std::vector<coeff_t>> coeffs_ptr) const {
    Vector<Var> vars;
    if (coeffs_ptr->size() == 1) {
      vars.push_back(qbpp::var(var_str));
    } else {
      for (size_t i = 0; i < coeffs_ptr->size(); i++) {
        vars.push_back(qbpp::var(var_str + "[" + std::to_string(i) + "]"));
      }
    }
    Expr expr{min_val};
    for (size_t i = 0; i < coeffs_ptr->size(); i++) {
      expr += Term(vars[i], (*coeffs_ptr)[i]);
    }
    return std::make_tuple(expr, var_str, min_val, max_val, coeffs_ptr, vars);
  }
 public:
  VarInt(const std::string& var_str, energy_t min_val, energy_t max_val,
         std::shared_ptr<std::vector<coeff_t>> coeffs_ptr)
      : VarInt(helper_func(var_str, min_val, max_val, coeffs_ptr)) {}
  VarInt(const VarInt&) = default;
  std::string name() const { return var_str_; }
  vindex_t var_count() const { return static_cast<vindex_t>(vars_.size()); }
  energy_t min_val() const { return min_val_; }
  energy_t max_val() const { return max_val_; }
  coeff_t coeff(vindex_t i) const { return (*coeffs_ptr_)[i]; }
  Var var(vindex_t i) const { return vars_[i]; }
  Var operator[](vindex_t i) const { return vars_[i]; }
  VarValMap val_map(energy_t val) const {
    if (val < min_val_ || val > max_val_) {
      throw std::out_of_range(
          THROW_MESSAGE("Value (", str(val), " out of range."));
    }
    val -= min_val_;
    VarValMap result;
    for (size_t i = var_count(); i >= 1; i--) {
      if (val >= (*coeffs_ptr_)[static_cast<size_t>(i) - 1]) {
        result[vars_[i - 1]] = 1;
        val -= (*coeffs_ptr_)[i - 1];
      } else {
        result[vars_[i - 1]] = 0;
      }
    }
    return result;
  }
};
class VarOnehotCore {
 public:
  const std::string var_str_;
  explicit VarOnehotCore(const std::string& var_str) : var_str_(var_str) {}
  VarOnehotCore()
      : var_str_("{" + std::to_string(new_unnamed_var_index()) + "}") {}
};
class VarOnehot {
  const std::string var_str_;
  const energy_t min_val_;
  const energy_t max_val_;
  const Vector<Var> vars_;
  const Expr expr_;
  const Vector<Var> new_vars() const {
    if (min_val_ > max_val_) {
      throw std::runtime_error(
          THROW_MESSAGE("min_val must be less than or equal to max_val."));
    }
    Vector<Var> vars;
    if (min_val_ == max_val_) {
      vars.push_back(var(var_str_));
    } else {
      for (energy_t i = 0; i <= max_val_ - min_val_; i++) {
        vars.push_back(var(var_str_ + "[" + str(i) + "]"));
      }
    }
    return vars;
  }
  const Expr new_expr() const {
    Expr result = Expr{0};
    for (energy_t i = 0; i <= max_val_ - min_val_; i++) {
      result += Term(vars_[static_cast<size_t>(i)],
                     static_cast<coeff_t>(i + min_val_));
    }
    return result;
  }
 public:
  VarOnehot(const std::string& var_str, energy_t min_val, energy_t max_val)
      : var_str_(var_str),
        min_val_(min_val),
        max_val_(max_val),
        vars_(new_vars()),
        expr_(new_expr()) {}
  operator Expr() const { return expr_; }
  Expr operator*() const;
  template <typename T,
            typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
  Expr select(const std::vector<T>& values) const {
    if (values.empty()) {
      throw std::runtime_error(THROW_MESSAGE("values is empty."));
    }
    Expr result;
    for (const auto& val : values) {
      if (val < min_val_ || val > max_val_) {
        throw std::out_of_range(
            THROW_MESSAGE("Value (", str(val), " out of range."));
      }
      result += Term(vars_[static_cast<size_t>(val - min_val_)]);
    }
    return result;
  }
  template <typename T,
            typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
  Expr select(std::initializer_list<T> values) const {
    return select(std::vector<T>(values));
  }
};
class Model {
 protected:
  const std::shared_ptr<const Expr> expr_ptr_;
  const std::shared_ptr<const IVMapper> index_var_ptr_;
  Model() = delete;
 public:
  Model(const Model& model)
      : expr_ptr_(model.expr_ptr_), index_var_ptr_(model.index_var_ptr_) {}
  Model(const Expr& expr)
      : expr_ptr_(std::make_shared<const Expr>(expr)),
        index_var_ptr_(std::make_shared<const IVMapper>(*expr_ptr_)) {}
  Model(const Expr& expr, const std::shared_ptr<const IVMapper>& iv_mapper_ptr)
      : expr_ptr_(std::make_shared<const Expr>(expr)),
        index_var_ptr_(iv_mapper_ptr) {}
  Model(Model&&) = default;
  Model(Expr&& expr) noexcept
      : expr_ptr_(std::make_shared<const Expr>(std::move(expr))),
        index_var_ptr_(std::make_shared<const IVMapper>(*expr_ptr_)) {}
  virtual ~Model() = default;
  vindex_t var_count() const { return index_var_ptr_->var_count(); }
  Var var(vindex_t index) const { return index_var_ptr_->var(index); }
  vindex_t index(Var var) const { return index_var_ptr_->index(var); }
  bool has(Var var) const { return index_var_ptr_->has(var); }
  virtual size_t term_count() const { return expr_ptr_->term_count(); }
  virtual size_t term_count(vindex_t size) const {
    return expr_ptr_->term_count(size);
  }
  const IVMapper& iv_mapper() const { return *index_var_ptr_; }
  const Expr& expr() const { return *expr_ptr_; }
  operator const Expr&() const { return *expr_ptr_; }
  energy_t constant() const { return expr_ptr_->constant(); }
  const Terms& terms() const { return expr_ptr_->terms(); }
};
#ifdef __CUDACC__
#define CUDA_HD __host__ __device__
#else
#define CUDA_HD
#endif
template <size_t N>
struct Indices {
  qbpp::vindex_t indices[N];
  CUDA_HD Indices() {}  
  CUDA_HD Indices(const qbpp::vindex_t (&in)[N]) {
    for (size_t i = 0; i < N; ++i) {
      indices[i] = in[i];  
    }
  }
  CUDA_HD qbpp::vindex_t operator[](size_t i) const { return indices[i]; }
};
template <size_t N>
struct TermVector {
  std::vector<Indices<N>> indices_;
  std::vector<qbpp::coeff_t> coeffs_;
  TermVector() = default;
  explicit TermVector(std::vector<Indices<N>>&& indices,
                      std::vector<qbpp::coeff_t>&& coeffs)
      : indices_(std::move(indices)), coeffs_(std::move(coeffs)) {
    if (indices_.size() != coeffs_.size()) {
      throw std::invalid_argument("Size mismatch between indices and coeffs");
    }
  }
  explicit TermVector(const std::vector<Indices<N>>& indices,
                      const std::vector<qbpp::coeff_t>& coeffs)
      : indices_(indices), coeffs_(coeffs) {
    if (indices_.size() != coeffs_.size()) {
      throw std::invalid_argument("Size mismatch between indices and coeffs");
    }
  }
  void push_back(const Indices<N>& indices, coeff_t coeff) {
    indices_.push_back(indices);
    coeffs_.push_back(coeff);
  }
  void push_back(Indices<N>&& indices, coeff_t coeff) {
    indices_.push_back(std::move(indices));
    coeffs_.push_back(coeff);
  }
  void emplace_back(const qbpp::vindex_t (&in)[N], coeff_t coeff) {
    indices_.emplace_back(in);
    coeffs_.push_back(coeff);
  }
  size_t size() const { return indices_.size(); }
  qbpp::coeff_t coeff(size_t i) const {
    if (i >= coeffs_.size()) {
      throw std::out_of_range("Index out of range in TermVector");
    }
    return coeffs_[i];
  }
  const Indices<N>& indices(size_t i) const { return indices_[i]; }
};
class HuboModel : public qbpp::Model {
  const uint32_t max_power_;
  const qbpp::energy_t constant_;
  const std::vector<qbpp::coeff_t> term1_;
  const std::vector<TermVector<1>> term2_;
  const std::vector<TermVector<2>> term3_;
  const std::vector<TermVector<3>> term4_;
  const std::vector<TermVector<4>> term5_;
  const std::vector<TermVector<5>> term6_;
  const std::vector<TermVector<6>> term7_;
  const std::vector<TermVector<7>> term8_;
  const std::vector<TermVector<8>> term9_;
  const std::vector<TermVector<9>> term10_;
  const std::vector<TermVector<10>> term11_;
  const std::vector<TermVector<11>> term12_;
  const std::vector<TermVector<12>> term13_;
  const std::vector<TermVector<13>> term14_;
  const std::vector<TermVector<14>> term15_;
  const std::vector<TermVector<15>> term16_;
  const qbpp::energy_t all_coeff_sum_;
  const std::vector<qbpp::energy_t> coeff_sum_;
  using all_hubo_terms =
      std::tuple<uint32_t, qbpp::energy_t, std::vector<qbpp::coeff_t>,
                 std::vector<TermVector<1>>, std::vector<TermVector<2>>,
                 std::vector<TermVector<3>>, std::vector<TermVector<4>>,
                 std::vector<TermVector<5>>, std::vector<TermVector<6>>,
                 std::vector<TermVector<7>>, std::vector<TermVector<8>>,
                 std::vector<TermVector<9>>, std::vector<TermVector<10>>,
                 std::vector<TermVector<11>>, std::vector<TermVector<12>>,
                 std::vector<TermVector<13>>, std::vector<TermVector<14>>,
                 std::vector<TermVector<15>>, qbpp::energy_t,
                 std::vector<qbpp::energy_t>>;
  static all_hubo_terms helper_func(const qbpp::Model& model);
  HuboModel(const qbpp::Model& model, all_hubo_terms&& terms)
      : Model(model),
        max_power_(std::get<0>(terms)),
        constant_(std::get<1>(terms)),
        term1_(std::move(std::get<2>(terms))),
        term2_(std::move(std::get<3>(terms))),
        term3_(std::move(std::get<4>(terms))),
        term4_(std::move(std::get<5>(terms))),
        term5_(std::move(std::get<6>(terms))),
        term6_(std::move(std::get<7>(terms))),
        term7_(std::move(std::get<8>(terms))),
        term8_(std::move(std::get<9>(terms))),
        term9_(std::move(std::get<10>(terms))),
        term10_(std::move(std::get<11>(terms))),
        term11_(std::move(std::get<12>(terms))),
        term12_(std::move(std::get<13>(terms))),
        term13_(std::move(std::get<14>(terms))),
        term14_(std::move(std::get<15>(terms))),
        term15_(std::move(std::get<16>(terms))),
        term16_(std::move(std::get<17>(terms))),
        all_coeff_sum_(std::get<18>(terms)),
        coeff_sum_(std::move(std::get<19>(terms))) {}
 public:
  HuboModel(const qbpp::Model& model)
      : HuboModel(model, std::move(helper_func(model))) {}
  HuboModel(const qbpp::Expr& expr) : HuboModel(qbpp::Model(expr)) {}
  HuboModel(const HuboModel& model) = default;
  HuboModel(HuboModel&& model) = default;
  const std::vector<qbpp::coeff_t>& term1() const { return term1_; }
  const std::vector<TermVector<1>>& term2() const { return term2_; }
  const std::vector<TermVector<2>>& term3() const { return term3_; }
  const std::vector<TermVector<3>>& term4() const { return term4_; }
  const std::vector<TermVector<4>>& term5() const { return term5_; }
  const std::vector<TermVector<5>>& term6() const { return term6_; }
  const std::vector<TermVector<6>>& term7() const { return term7_; }
  const std::vector<TermVector<7>>& term8() const { return term8_; }
  const std::vector<TermVector<8>>& term9() const { return term9_; }
  const std::vector<TermVector<9>>& term10() const { return term10_; }
  const std::vector<TermVector<10>>& term11() const { return term11_; }
  const std::vector<TermVector<11>>& term12() const { return term12_; }
  const std::vector<TermVector<12>>& term13() const { return term13_; }
  const std::vector<TermVector<13>>& term14() const { return term14_; }
  const std::vector<TermVector<14>>& term15() const { return term15_; }
  const std::vector<TermVector<15>>& term16() const { return term16_; }
  qbpp::coeff_t term1(qbpp::vindex_t i) const { return term1_[i]; }
  const TermVector<1>& term2(qbpp::vindex_t i) const { return term2_[i]; }
  const TermVector<2>& term3(qbpp::vindex_t i) const { return term3_[i]; }
  const TermVector<3>& term4(qbpp::vindex_t i) const { return term4_[i]; }
  const TermVector<4>& term5(qbpp::vindex_t i) const { return term5_[i]; }
  const TermVector<5>& term6(qbpp::vindex_t i) const { return term6_[i]; }
  const TermVector<6>& term7(qbpp::vindex_t i) const { return term7_[i]; }
  const TermVector<7>& term8(qbpp::vindex_t i) const { return term8_[i]; }
  const TermVector<8>& term9(qbpp::vindex_t i) const { return term9_[i]; }
  const TermVector<9>& term10(qbpp::vindex_t i) const { return term10_[i]; }
  const TermVector<10>& term11(qbpp::vindex_t i) const { return term11_[i]; }
  const TermVector<11>& term12(qbpp::vindex_t i) const { return term12_[i]; }
  const TermVector<12>& term13(qbpp::vindex_t i) const { return term13_[i]; }
  const TermVector<13>& term14(qbpp::vindex_t i) const { return term14_[i]; }
  const TermVector<14>& term15(qbpp::vindex_t i) const { return term15_[i]; }
  const TermVector<15>& term16(qbpp::vindex_t i) const { return term16_[i]; }
  qbpp::energy_t all_coeff_sum() const { return all_coeff_sum_; }
  std::vector<qbpp::energy_t> coeff_sum() const { return coeff_sum_; }
  qbpp::energy_t coeff_sum(qbpp::vindex_t i) const { return coeff_sum_[i]; }
  size_t degree(qbpp::vindex_t index) const {
    return term2_[index].size() + term3_[index].size() + term4_[index].size() +
           term5_[index].size() + term6_[index].size() + term7_[index].size() +
           term8_[index].size() + term9_[index].size() + term10_[index].size() +
           term11_[index].size() + term12_[index].size() +
           term13_[index].size() + term14_[index].size() +
           term15_[index].size() + term16_[index].size();
  }
  uint32_t max_power() const { return max_power_; }
  std::string str() const;
};
using QuboModel = HuboModel;
inline std::ostream& operator<<(std::ostream& os, const HuboModel& model) {
  return os << model.str();
}
template <size_t N>
struct FlatTermVectors {
  const qbpp::vindex_t* const var_count_ptr_;
  const uint64_t* const size_array_;
  const uint64_t* const head_array_;
  const Indices<N>* const indices_array_;
  const qbpp::coeff_t* const coeffs_array_;
  using all_flat_term_vectors =
      std::tuple<const qbpp::vindex_t*, const uint64_t*, const uint64_t*,
                 const Indices<N>*, const qbpp::coeff_t*>;
  static all_flat_term_vectors helper_func(
      const std::vector<TermVector<N>>& term_vector) {
    qbpp::vindex_t* var_count_ptr =
        new qbpp::vindex_t(static_cast<qbpp::vindex_t>(term_vector.size()));
    uint64_t* size_array = new uint64_t[term_vector.size()];
    uint64_t* head_array = new uint64_t[term_vector.size() + 1];
    head_array[0] = 0;
    for (size_t i = 0; i < term_vector.size(); ++i) {
      size_array[i] = term_vector[i].size();
      head_array[i + 1] = head_array[i] + size_array[i];
    }
    size_t total_terms = head_array[term_vector.size()];
    Indices<N>* indices_array = new Indices<N>[total_terms];
    qbpp::coeff_t* coeffs_array = new qbpp::coeff_t[total_terms];
    for (size_t i = 0; i < term_vector.size(); ++i) {
      const auto& tv = term_vector[i];
      for (size_t j = 0; j < tv.size(); ++j) {
        indices_array[head_array[i] + j] = tv.indices(j);
        coeffs_array[head_array[i] + j] = tv.coeff(j);
      }
    }
    return std::make_tuple(var_count_ptr, size_array, head_array, indices_array,
                           coeffs_array);
  }
  FlatTermVectors(all_flat_term_vectors&& args)
      : var_count_ptr_(std::get<0>(args)),
        size_array_(std::get<1>(args)),
        head_array_(std::get<2>(args)),
        indices_array_(std::get<3>(args)),
        coeffs_array_(std::get<4>(args)) {}
  FlatTermVectors(const FlatTermVectors&) = delete;
  FlatTermVectors& operator=(const FlatTermVectors&) = delete;
  FlatTermVectors(FlatTermVectors&&) = delete;
  FlatTermVectors& operator=(FlatTermVectors&&) = delete;
 public:
  FlatTermVectors(const std::vector<TermVector<N>>& term_vector)
      : FlatTermVectors(std::move(helper_func(term_vector))) {}
  ~FlatTermVectors() {
    delete var_count_ptr_;
    delete[] size_array_;
    delete[] head_array_;
    delete[] indices_array_;
    delete[] coeffs_array_;
  }
  uint64_t size(qbpp::vindex_t i) const { return size_array_[i]; }
  uint64_t head(qbpp::vindex_t i) const { return head_array_[i]; }
  const Indices<N>& indices(qbpp::vindex_t i, uint64_t j) const {
    return indices_array_[head(i) + j];
  }
  qbpp::vindex_t index(qbpp::vindex_t i, uint64_t j, uint8_t k) const {
    return indices_array_[head(i) + j].indices[k];
  }
  qbpp::coeff_t coeff(qbpp::vindex_t i, uint64_t j) const {
    return coeffs_array_[head(i) + j];
  }
  std::string str() const {
    std::string result = "FlatTermVectors<" + qbpp::str(N) + ">\n";
    qbpp::vindex_t var_count = *var_count_ptr_;
    for (qbpp::vindex_t i = 0; i < var_count; ++i) {
      result += "Variable " + qbpp::str(i) + "\n";
      result += "  Size: " + qbpp::str(size(i)) + "\n";
      result += "  Head: " + qbpp::str(head(i)) + "\n";
      for (uint64_t j = 0; j < size(i); ++j) {
        result += "  Term " + qbpp::str(j) + ": ";
        for (uint8_t k = 0; k < N; ++k) {
          result += qbpp::str(index(i, j, k)) + " ";
        }
        result += "Coeff: " + qbpp::str(coeff(i, j)) + "\n";
      }
    }
    return result;
  }
};
class FlatHuboModel {
 protected:
  const uint32_t* max_power_ptr_;
  const qbpp::vindex_t* const var_count_ptr_;
  const qbpp::energy_t* const constant_ptr_;
  const qbpp::coeff_t* const term1_array_;
  const FlatTermVectors<1>* const term2_array_;
  const FlatTermVectors<2>* const term3_array_;
  const FlatTermVectors<3>* const term4_array_;
  const FlatTermVectors<4>* const term5_array_;
  const FlatTermVectors<5>* const term6_array_;
  const FlatTermVectors<6>* const term7_array_;
  const FlatTermVectors<7>* const term8_array_;
  const FlatTermVectors<8>* const term9_array_;
  const FlatTermVectors<9>* const term10_array_;
  const FlatTermVectors<10>* const term11_array_;
  const FlatTermVectors<11>* const term12_array_;
  const FlatTermVectors<12>* const term13_array_;
  const FlatTermVectors<13>* const term14_array_;
  const FlatTermVectors<14>* const term15_array_;
  const FlatTermVectors<15>* const term16_array_;
  const qbpp::energy_t* const all_coeff_sum_ptr_;
  const qbpp::energy_t* const coeff_sum_array_;
 private:
  using all_flat_hubo_model =
      std::tuple<const uint32_t*, const qbpp::vindex_t*, const qbpp::energy_t*,
                 const qbpp::coeff_t*, const FlatTermVectors<1>*,
                 const FlatTermVectors<2>*, const FlatTermVectors<3>*,
                 const FlatTermVectors<4>*, const FlatTermVectors<5>*,
                 const FlatTermVectors<6>*, const FlatTermVectors<7>*,
                 const FlatTermVectors<8>*, const FlatTermVectors<9>*,
                 const FlatTermVectors<10>*, const FlatTermVectors<11>*,
                 const FlatTermVectors<12>*, const FlatTermVectors<13>*,
                 const FlatTermVectors<14>*, const FlatTermVectors<15>*,
                 const qbpp::energy_t*, const qbpp::energy_t*>;
  all_flat_hubo_model helper_func(const qbpp::HuboModel& hubo_model) {
    const uint32_t* max_power_ptr = new uint32_t(hubo_model.max_power());
    const qbpp::vindex_t* var_count_ptr =
        new qbpp::vindex_t(hubo_model.var_count());
    const qbpp::energy_t* constant_ptr =
        new qbpp::energy_t(hubo_model.constant());
    qbpp::coeff_t* term1_array = new qbpp::coeff_t[hubo_model.term1().size()];
    std::copy(hubo_model.term1().data(),
              hubo_model.term1().data() + hubo_model.term1().size(),
              term1_array);
    const FlatTermVectors<1>* term2_array =
        new FlatTermVectors<1>(hubo_model.term2());
    const FlatTermVectors<2>* term3_array =
        new FlatTermVectors<2>(hubo_model.term3());
    const FlatTermVectors<3>* term4_array =
        new FlatTermVectors<3>(hubo_model.term4());
    const FlatTermVectors<4>* term5_array =
        new FlatTermVectors<4>(hubo_model.term5());
    const FlatTermVectors<5>* term6_array =
        new FlatTermVectors<5>(hubo_model.term6());
    const FlatTermVectors<6>* term7_array =
        new FlatTermVectors<6>(hubo_model.term7());
    const FlatTermVectors<7>* term8_array =
        new FlatTermVectors<7>(hubo_model.term8());
    const FlatTermVectors<8>* term9_array =
        new FlatTermVectors<8>(hubo_model.term9());
    const FlatTermVectors<9>* term10_array =
        new FlatTermVectors<9>(hubo_model.term10());
    const FlatTermVectors<10>* term11_array =
        new FlatTermVectors<10>(hubo_model.term11());
    const FlatTermVectors<11>* term12_array =
        new FlatTermVectors<11>(hubo_model.term12());
    const FlatTermVectors<12>* term13_array =
        new FlatTermVectors<12>(hubo_model.term13());
    const FlatTermVectors<13>* term14_array =
        new FlatTermVectors<13>(hubo_model.term14());
    const FlatTermVectors<14>* term15_array =
        new FlatTermVectors<14>(hubo_model.term15());
    const FlatTermVectors<15>* term16_array =
        new FlatTermVectors<15>(hubo_model.term16());
    const qbpp::energy_t* all_coeff_sum_ptr =
        new qbpp::energy_t(hubo_model.all_coeff_sum());
    qbpp::energy_t* coeff_sum_array =
        new qbpp::energy_t[hubo_model.var_count()];
    for (qbpp::vindex_t i = 0; i < hubo_model.var_count(); ++i) {
      coeff_sum_array[i] = hubo_model.coeff_sum(i);
    }
    return std::make_tuple(
        max_power_ptr, var_count_ptr, constant_ptr, term1_array, term2_array,
        term3_array, term4_array, term5_array, term6_array, term7_array,
        term8_array, term9_array, term10_array, term11_array, term12_array,
        term13_array, term14_array, term15_array, term16_array,
        all_coeff_sum_ptr, coeff_sum_array);
  }
  FlatHuboModel(all_flat_hubo_model&& args)
      : max_power_ptr_(std::get<0>(args)),
        var_count_ptr_(std::get<1>(args)),
        constant_ptr_(std::get<2>(args)),
        term1_array_(std::get<3>(args)),
        term2_array_(std::get<4>(args)),
        term3_array_(std::get<5>(args)),
        term4_array_(std::get<6>(args)),
        term5_array_(std::get<7>(args)),
        term6_array_(std::get<8>(args)),
        term7_array_(std::get<9>(args)),
        term8_array_(std::get<10>(args)),
        term9_array_(std::get<11>(args)),
        term10_array_(std::get<12>(args)),
        term11_array_(std::get<13>(args)),
        term12_array_(std::get<14>(args)),
        term13_array_(std::get<15>(args)),
        term14_array_(std::get<16>(args)),
        term15_array_(std::get<17>(args)),
        term16_array_(std::get<18>(args)),
        all_coeff_sum_ptr_(std::get<19>(args)),
        coeff_sum_array_(std::get<20>(args)) {}
  FlatHuboModel(const FlatHuboModel&) = delete;
  FlatHuboModel& operator=(const FlatHuboModel&) = delete;
 public:
  explicit FlatHuboModel(const HuboModel& model)
      : FlatHuboModel(std::move(helper_func(model))) {}
  ~FlatHuboModel() {
    delete var_count_ptr_;
    delete constant_ptr_;
    delete[] term1_array_;
    delete term2_array_;
    delete term3_array_;
    delete term4_array_;
    delete term5_array_;
    delete term6_array_;
    delete term7_array_;
    delete term8_array_;
    delete term9_array_;
    delete term10_array_;
    delete term11_array_;
    delete term12_array_;
    delete term13_array_;
    delete term14_array_;
    delete term15_array_;
    delete term16_array_;
    delete all_coeff_sum_ptr_;
    delete[] coeff_sum_array_;
  }
  qbpp::vindex_t var_count() const { return *var_count_ptr_; }
  qbpp::vindex_t size64() const { return (var_count() + 63) / 64; }
  qbpp::energy_t constant() const { return *constant_ptr_; }
  const qbpp::coeff_t* term1_array() const { return term1_array_; }
  const FlatTermVectors<1>& term2_array() const { return *term2_array_; }
  const FlatTermVectors<2>& term3_array() const { return *term3_array_; }
  const FlatTermVectors<3>& term4_array() const { return *term4_array_; }
  const FlatTermVectors<4>& term5_array() const { return *term5_array_; }
  const FlatTermVectors<5>& term6_array() const { return *term6_array_; }
  const FlatTermVectors<6>& term7_array() const { return *term7_array_; }
  const FlatTermVectors<7>& term8_array() const { return *term8_array_; }
  const FlatTermVectors<8>& term9_array() const { return *term9_array_; }
  const FlatTermVectors<9>& term10_array() const { return *term10_array_; }
  const FlatTermVectors<10>& term11_array() const { return *term11_array_; }
  const FlatTermVectors<11>& term12_array() const { return *term12_array_; }
  const FlatTermVectors<12>& term13_array() const { return *term13_array_; }
  const FlatTermVectors<13>& term14_array() const { return *term14_array_; }
  const FlatTermVectors<14>& term15_array() const { return *term15_array_; }
  const FlatTermVectors<15>& term16_array() const { return *term16_array_; }
  uint32_t max_power() const { return *max_power_ptr_; }
  qbpp::energy_t all_coeff_sum() const { return *all_coeff_sum_ptr_; }
  const qbpp::energy_t& coeff_sum_array() const { return *coeff_sum_array_; }
  std::string str() const {
    std::string result = "FlatHuboModel\n";
    result += "Variable Count: " + qbpp::str(*var_count_ptr_) + "\n";
    result += "Constant: " + qbpp::str(*constant_ptr_) + "\n";
    result += "Term1 Coefficients:\n";
    for (qbpp::vindex_t i = 0; i < *var_count_ptr_; ++i) {
      result += "  Variable " + qbpp::str(i) + ": " +
                qbpp::str(term1_array_[i]) + "\n";
    }
    result += "Term2:\n" + term2_array_->str();
    result += "Term3:\n" + term3_array_->str();
    result += "Term4:\n" + term4_array_->str();
    result += "Term5:\n" + term5_array_->str();
    result += "Term6:\n" + term6_array_->str();
    result += "Term7:\n" + term7_array_->str();
    result += "Term8:\n" + term8_array_->str();
    result += "Term9:\n" + term9_array_->str();
    result += "Term10:\n" + term10_array_->str();
    result += "Term11:\n" + term11_array_->str();
    result += "Term12:\n" + term12_array_->str();
    result += "Term13:\n" + term13_array_->str();
    result += "Term14:\n" + term14_array_->str();
    result += "Term15:\n" + term15_array_->str();
    result += "Term16:\n" + term16_array_->str();
    result += "All Coefficients Sum: " + qbpp::str(*all_coeff_sum_ptr_) + "\n";
    result += "Coefficient Sums:\n";
    for (qbpp::vindex_t i = 0; i < *var_count_ptr_; ++i) {
      result += "  Variable " + qbpp::str(i) + ": " +
                qbpp::str(coeff_sum_array_[i]) + "\n";
    }
    return result;
  }
};
inline std::ostream& operator<<(std::ostream& os, const FlatHuboModel& model) {
  os << model.str();
  return os;
}
class BitVector {
  void* impl_;
 public:
  explicit BitVector(vindex_t bit_count)
      : impl_(qbpp_bitvector_create(bit_count)) {}
  explicit BitVector(vindex_t bit_count, const uint64_t* bit_array)
      : impl_(qbpp_bitvector_create_from_array(bit_count, bit_array)) {}
  BitVector(const BitVector& other) : impl_(qbpp_bitvector_copy(other.impl_)) {}
  BitVector(BitVector&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
  }
  BitVector& operator=(const BitVector& other) {
    if (this != &other) {
      qbpp_bitvector_destroy(impl_);
      impl_ = qbpp_bitvector_copy(other.impl_);
    }
    return *this;
  }
  BitVector& operator=(BitVector&& other) noexcept {
    if (this != &other) {
      qbpp_bitvector_destroy(impl_);
      impl_ = other.impl_;
      other.impl_ = nullptr;
    }
    return *this;
  }
  ~BitVector() { qbpp_bitvector_destroy(impl_); }
  void set(vindex_t index, bool value) {
    qbpp_bitvector_set(impl_, index, value);
  }
  bool get(vindex_t index) const { return qbpp_bitvector_get(impl_, index); }
  void flip(vindex_t index) { qbpp_bitvector_flip(impl_, index); }
  vindex_t size() const { return qbpp_bitvector_size(impl_); }
  vindex_t size64() const { return qbpp_bitvector_size64(impl_); }
  const uint64_t* bits_ptr() const { return qbpp_bitvector_bits_ptr(impl_); }
  uint64_t get64(vindex_t index64) const {
    return qbpp_bitvector_get64(impl_, index64);
  }
  void set64(vindex_t index64, uint64_t value) {
    qbpp_bitvector_set64(impl_, index64, value);
  }
  void clear() { qbpp_bitvector_clear(impl_); }
  vindex_t popcount() const { return qbpp_bitvector_popcount(impl_); }
  bool operator==(const BitVector& other) const {
    return qbpp_bitvector_equal(impl_, other.impl_);
  }
  bool operator<(const BitVector& other) const {
    return qbpp_bitvector_less_than(impl_, other.impl_);
  }
  void* impl() const { return impl_; }
  std::string str() const {
    std::string result;
    vindex_t size = this->size();
    for (vindex_t i = 0; i < size; ++i) {
      result += (get(i) ? '1' : '0');
    }
    return result;
  }
};
class Sol {
 protected:
  const Model model_;
  const Expr& expr_ = model_.expr();
  const IVMapper& iv_mapper_ = model_.iv_mapper();
  const energy_t constant_ = model_.constant();
  BitVector bit_vector_;
  std::optional<energy_t> energy_ = std::nullopt;
  double tts_ = -1.0;
  MapList map_list() const;
  Sol() = delete;
 public:
  Sol(const Sol& sol) = default;
  Sol(Sol&& other) = default;
  explicit Sol(const Model& model)
      : model_(model),
        bit_vector_(model.var_count()),
        energy_(model.constant()) {}
  explicit Sol(const Model& model, const BitVector& bit_vector)
      : model_(model), bit_vector_(bit_vector), energy_(model.constant()) {}
  explicit Sol(const Model& model, energy_t energy, const uint64_t* bit_array,
               double tts = -1.0)
      : model_(model),
        bit_vector_(model.var_count(), bit_array),
        energy_(energy),
        tts_(tts) {}
  virtual ~Sol() = default;
  Sol& operator=(const Sol& sol) {
    if (this != &sol) {
      bit_vector_ = sol.bit_vector_;
      energy_ = sol.energy_;
      tts_ = sol.tts_;
    }
    return *this;
  }
  bool operator==(const Sol& sol) const {
    if (energy() != sol.energy()) return false;
    return bit_vector_ == sol.bit_vector_;
  }
  bool operator<(const Sol& sol) const {
    if (energy() < sol.energy()) return true;
    if (energy() == sol.energy())
      return bit_vector_ < sol.bit_vector_;
    else
      return false;
  }
  var_val_t get(vindex_t index) const { return bit_vector_.get(index); }
  var_val_t get(Var var) const { return get(iv_mapper_.index(var)); }
  energy_t get(const Expr& expr) const { return eval(expr, *this); }
  energy_t operator()(const Expr& expr) const { return get(expr); }
  energy_t operator()(vindex_t index) const { return get(index); }
  bool has(Var var) const { return iv_mapper_.has(var); }
  void clear() {
    bit_vector_.clear();
    energy_ = constant_;
  }
  void nullify() { energy_ = std::nullopt; }
  bool is_null() const { return !energy_.has_value(); }
  template <typename T>
  auto get(const Vector<T>& vec) const {
    Vector<decltype(get(std::declval<T>()))> result;
    result.reserve(vec.size());
    for (const auto& elem : vec) {
      result.push_back(get(elem));
    }
    return result;
  }
  void set(vindex_t index, bool value) {
    energy_.reset();
    bit_vector_.set(index, value);
  }
  void set(Var var, bool value) {
    energy_.reset();
    set(iv_mapper_.index(var), value);
  }
  void set64(vindex_t index64, uint64_t value) {
    energy_.reset();
    bit_vector_.set64(index64, value);
  }
  void bit_vector_flip(vindex_t index) { bit_vector_.flip(index); }
  virtual void flip(vindex_t index) {
    energy_.reset();
    bit_vector_.flip(index);
  }
  virtual void flip_bit_add_delta(vindex_t index, energy_t delta) {
    energy_ = energy_.value() + delta;
    bit_vector_.flip(index);
  }
  vindex_t popcount() const { return bit_vector_.popcount(); }
  energy_t energy() const {
    if (!energy_) {
      throw std::runtime_error(THROW_MESSAGE("Energy is not set."));
    }
    return *energy_;
  }
  energy_t comp_energy() const { return energy_ ? *energy_ : expr_(*this); }
  void energy(energy_t energy) { energy_ = energy; }
  void energy(const Expr& expr) { energy_ = expr(*this); }
  bool energy_has_value() const { return energy_.has_value(); }
  energy_t add_energy(energy_t energy) { return *energy_ += energy; }
  const BitVector& bit_vector() const { return bit_vector_; }
  BitVector& bit_vector() { return bit_vector_; }
  vindex_t var_count() const { return iv_mapper_.var_count(); }
  energy_t constant() const { return constant_; }
  Var var(vindex_t index) const { return iv_mapper_.var(index); }
  vindex_t index(Var var) const { return iv_mapper_.index(var); }
  operator MapList() const { return map_list(); }
  Sol& set(const Sol& sol) {
    for (vindex_t i = 0; i < sol.var_count(); ++i) {
      set(sol.var(i), sol.get(i));
    }
    return *this;
  }
  Sol& set(const MapList& map_list) {
    VarValMap var_val_map = list_to_var_val(map_list);
    for (const auto& [var, val] : var_val_map) {
      set(var, val);
    }
    if (!energy_) energy_ = comp_energy();
    return *this;
  }
  void tts(double tts) { tts_ = tts; }
  double tts() const { return tts_; }
};
template <typename T, typename U>
class SolHolderTemplate {
 protected:
  T sol_;
  std::optional<energy_t> bound_ = std::nullopt;
  double startx_ = time();
  double tts_{-1.0f};
  std::string solver_ = "N/A";
  mutable std::mutex mtx_;
 public:
  explicit SolHolderTemplate(const Model& model) : sol_(model) {}
  explicit SolHolderTemplate(const Expr& expr) : sol_(Model(expr)) {}
  explicit SolHolderTemplate(const T& sol) : sol_(sol) {}
  SolHolderTemplate(const SolHolderTemplate<T, U>& other) = delete;
  virtual ~SolHolderTemplate() = default;
  operator Sol() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return sol_;
  }
  virtual std::optional<double> set_if_better(const T& new_sol,
                                              const std::string& solver = "") {
    if (new_sol.energy() >= sol_.energy()) {
      return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    if (new_sol.energy() < sol_.energy()) {
      tts_ = time() - startx_;
      sol_ = new_sol;
      solver_ = solver;
      return std::make_optional(tts_);
    } else {
      return std::nullopt;
    }
  }
  virtual std::optional<double> set_if_better(T&& new_sol,
                                              const std::string& solver = "") {
    if (new_sol.energy() >= sol_.energy()) {
      return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    if (new_sol.energy() < sol_.energy()) {
      tts_ = time() - startx_;
      sol_ = std::move(new_sol);
      solver_ = solver;
      return std::make_optional(tts_);
    } else {
      return std::nullopt;
    }
  }
  virtual std::optional<T> get_if_better(U my_energy) {
    if (sol_.energy() >= my_energy) return std::nullopt;
    std::lock_guard<std::mutex> lock(mtx_);
    if (sol_.energy() < my_energy) {
      return sol_;
    } else {
      return std::nullopt;
    }
  }
  std::optional<T> get_if_better(const T& my_sol) {
    return get_if_better(my_sol.energy());
  }
  std::tuple<T, double, std::string> get_sol_tts_solver() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return std::make_tuple(sol_, tts_, solver_);
  }
  vindex_t var_count() const { return sol_.var_count(); }
  const U energy() const { return sol_.energy(); }
  const T& sol() const { return sol_; }
  const T& sol(const T& sol) {
    std::lock_guard<std::mutex> lock(mtx_);
    sol_ = sol;
    return sol_;
  }
  T copy_sol() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return sol_;
  }
  void clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    sol_.clear();
    tts_ = 0;
    solver_ = "N/A";
  }
  double tts() const { return tts_; }
  const std::string& solver() const { return solver_; }
  void bound(U bound) { bound_ = bound; }
  std::optional<U> bound() const { return bound_; }
};
using SolHolder = SolHolderTemplate<Sol, energy_t>;
inline std::string str_impl(const Vars& vars,
                            std::function<std::string(Var)> str) {
  std::string result;
  for (const auto& var : vars) {
    if (!result.empty()) {
      result += "*";
    }
    result += str(var);
  }
  return result;
}
inline std::string str_impl(const Term& term,
                            std::function<std::string(Var)> str) {
  std::string result;
  if (term.var_count() == 0) return qbpp::str(term.coeff());
  if (term.coeff() == -1)
    result += "-";
  else if (term.coeff() != 1)
    result += qbpp::str(term.coeff()) + "*";
  result += str_impl(term.vars(), str);
  return result;
}
inline std::string str_impl(const Terms& terms,
                            std::function<std::string(Var)> str) {
  bool first = true;  
  std::string result;
  for (const auto& term : terms) {
    if (first) {
      result += str_impl(term, str);
      first = false;
    } else {
      if (term.coeff() < 0)
        result += " " + str_impl(term, str);
      else
        result += " +" + str_impl(term, str);
    }
  }
  return result;
}
inline std::string str_impl(const Expr& expr,
                            std::function<std::string(Var)> str) {
  bool first = true;  
  std::string result;
  if (expr.constant() != 0) {
    result += qbpp::str(expr.constant());
    first = false;
  } else if (expr.term_count() == 0)
    return "0";
  for (const auto& term : expr.terms()) {
    if (first) {
      result += str_impl(term, str);
      first = false;
    } else {
      if (term.coeff() < 0)
        result += " " + str_impl(term, str);
      else
        result += " +" + str_impl(term, str);
    }
  }
  return result;
}
inline std::string str(const Vars& vars) {
  return str_impl(vars, static_cast<std::string (*)(Var)>(str));
}
inline std::string str(const BitVector& bit_vector) {
  std::string result;
  for (vindex_t i = 0; i < bit_vector.size(); i++) {
    result += bit_vector.get(i) ? "1" : "0";
  }
  return result;
}
inline std::string str(const Term& term) {
  return str_impl(term, static_cast<std::string (*)(Var)>(str));
}
inline std::string str(const Expr& expr) {
  return str_impl(expr, static_cast<std::string (*)(Var)>(str));
}
inline std::string str(const Terms& terms) {
  return str_impl(terms, static_cast<std::string (*)(Var)>(str));
}
inline std::string str(const Expr& expr, const std::string& prefix,
                       const std::string& separator [[maybe_unused]] = ",") {
  if (prefix == "") {
    return "{" + str_impl(expr, static_cast<std::string (*)(Var)>(str)) + "}";
  } else {
    return "{" + prefix + "," +
           str_impl(expr, static_cast<std::string (*)(Var)>(str)) + "}";
  }
}
inline std::string str(const Model& model) {
  auto to_str = [&model](Var var) -> std::string {
    vindex_t index = model.index(var);  
    std::string var_string = std::to_string(index & qbpp::vindex_mask);
    return "<" + var_string + ">";
  };
  return str_impl(model.expr(), to_str);
}
inline std::string str(const MapList& map_list) {
  std::ostringstream oss;
  oss << "{";
  bool first = true;
  for (const auto& [key, val] : map_list) {
    if (!first) {
      oss << ",";
    }
    first = false;
    auto str_val = str(val);
    std::visit(
        [&](const auto& arg) {
          using T = std::decay_t<decltype(arg)>;
          oss << "{";
          if constexpr (std::is_same_v<T, VarInt>) {
            oss << arg.name();
          } else {
            oss << str(arg);
          }
          oss << "," << str_val << "}";
        },
        key);
  }
  oss << "}";
  return oss.str();
}
inline std::string str(const Sol& sol) {
  if (sol.tts() >= 0.0) {
    return std::to_string(sol.tts()) + ":" + str(sol.comp_energy()) + ":" +
           str(static_cast<MapList>(sol));
  }
  return str(sol.comp_energy()) + ":" + str(static_cast<MapList>(sol));
}
template <typename T>
std::string str(const Vector<T>& vec, const std::string& prefix = "",
                const std::string& separator = ",") {
  std::string result;
  for (size_t i = 0; i < vec.size(); i++) {
    result += str(vec[i], prefix + "[" + qbpp::str(i) + "]", separator);
    if (i < vec.size() - 1) result += separator;
  }
  return result;
}
inline std::ostream& operator<<(std::ostream& os, Var var) {
  return os << str(var);
}
inline std::ostream& operator<<(std::ostream& os, const VarInt& var_int) {
  return os << qbpp::Expr(var_int);
}
inline std::ostream& operator<<(std::ostream& os, const Term& term) {
  return os << str(term);
}
inline std::ostream& operator<<(std::ostream& os, const Terms& terms) {
  return os << str(terms);
}
inline std::ostream& operator<<(std::ostream& os, const Expr& expr) {
  return os << str(expr);
}
inline std::ostream& operator<<(std::ostream& os, const Model& model) {
  return os << str(model);
}
inline std::ostream& operator<<(std::ostream& os, const MapList& map_list) {
  return os << str(map_list);
}
inline std::ostream& operator<<(std::ostream& os, const Sol& sol) {
  return os << str(sol);
}
template <typename T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& vec) {
  os << str(vec);
  return os;
}
template <typename T>
std::ostream& operator<<(std::ostream& os,
                         const std::pair<std::string, Vector<T>>& vec) {
  os << str(vec.second, vec.first);
  return os;
}
}  
namespace qbpp::impl {
inline std::ostream& operator<<(std::ostream& os,
                                const qbpp::BitVector& bit_vector) {
  os << bit_vector.str();
  return os;
}
}  
namespace qbpp {
inline std::string str_short(const Expr& expr) {
  if (expr.term_count() <= 5) return str(expr);
  std::string result;
  if (expr.constant() != 0) result += qbpp::str(expr.constant()) + " + ";
  auto it = expr.terms().begin();
  result += str(*it) + " + ";
  ++it;
  result += str(*it) + " + ... + ";
  it = expr.terms().end();
  --it;
  --it;
  result += str(*it) + " + ";
  ++it;
  result += str(*it);
  result += " (" + qbpp::str(expr.term_count()) + " terms)";
  return result;
}
namespace impl {
inline size_t var_hash::operator()(const Var var) const {
  std::hash<vindex_t> hasher;
  return hasher(var.index());
}
inline size_t vars_hash::operator()(const Vars& vars) const {
  std::hash<vindex_t> hasher;
  size_t seed = 0;
  for (const auto& item : vars) {
    seed ^= hasher(item.index()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}
}  
namespace impl {
template <typename T>
Expr equal(const T& lhs, const T& rhs) {
  return lhs - rhs == 0;
}
template <typename T>
Expr equal(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::runtime_error(THROW_MESSAGE("Vector dimension mismatch"));
  }
  Expr result = 0;
  for (size_t i = 0; i < lhs.size(); i++) {
    result += equal(lhs[i], rhs[i]);
  }
  return result;
}
}  
inline VarValMap list_to_var_val(const MapList& map_list) {
  VarValMap var_int_map;
  for (const auto& [key, val] : map_list) {
    if (val.term_count() >= 1) {
      throw std::runtime_error(THROW_MESSAGE("Value must be an integer."));
    }
    auto constant = val.constant();
    std::visit(
        [&](auto&& arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, VarInt>) {
            for (const auto& [var, bin_val] :
                 arg.val_map(static_cast<coeff_t>(constant))) {
              var_int_map[var] = bin_val;
            }
          } else {
            var_int_map[arg] = static_cast<var_val_t>(constant);
          }
        },
        key);
  }
  return var_int_map;
}
inline MapDict list_to_dict(const MapList& map_list) {
  MapDict map_dict;
  for (const auto& [key, val] : map_list) {
    auto val_local = val;
    std::visit(
        [&](auto&& arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, VarInt>) {
            if (val_local.term_count() >= 1) {
              throw std::runtime_error(
                  THROW_MESSAGE("VarInt must have a constant."));
            }
            for (const auto& [var, bin_val] :
                 arg.val_map(static_cast<coeff_t>(val_local.constant()))) {
              map_dict[var] = bin_val;
            }
          } else {
            map_dict[arg] = val_local;
          }
        },
        key);
  }
  return map_dict;
}
inline energy_t eval(const Term& term, const VarValMap& var_val_map) {
  energy_t result = term.coeff();
  for (const auto& item : term.vars()) {
    try {
      result *= var_val_map.at(item);
    } catch (const std::out_of_range& e) {
      throw std::runtime_error(THROW_MESSAGE("Variable ", qbpp::str(item),
                                             " is not found in the map."));
    }
  }
  return result;
}
inline energy_t eval(const Term& term, const Sol& sol) {
  energy_t result = term.coeff();
  for (const auto& item : term.vars()) {
    result *= sol.get(item);
  }
  return result;
}
inline Expr replace(const Expr& expr, const MapList& map_list) {
  auto result = expr;
  return result.replace(map_list);
}
inline energy_t eval_var_val_map(const Expr& expr,
                                 const VarValMap& var_val_map) {
  energy_t result = expr.constant();
  tbb::combinable<energy_t> local_results(
      [] { return static_cast<energy_t>(0); });
  tbb::parallel_for(size_t(0), expr.terms().size(), [&](size_t i) {
    local_results.local() += eval(expr.terms()[i], var_val_map);
  });
  result += local_results.combine(
      [](const energy_t& a, const energy_t& b) { return a + b; });
  return result;
}
inline energy_t eval(const Expr& expr, const MapList& map_list) {
  VarValMap var_val_map = list_to_var_val(map_list);
  return eval_var_val_map(expr, var_val_map);
}
inline energy_t eval(const Expr& expr, const Sol& sol) {
  tbb::combinable<energy_t> local_results(
      [] { return static_cast<energy_t>(0); });
  tbb::parallel_for(tbb::blocked_range<size_t>(0, expr.terms().size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                      auto& local = local_results.local();
                      for (size_t i = range.begin(); i < range.end(); ++i) {
                        local += eval(expr.terms()[i], sol);
                      }
                    });
  energy_t result = expr.constant();
  result += local_results.combine(
      [](const energy_t& a, const energy_t& b) { return a + b; });
  return result;
}
inline Var var(const std::string& name) {
  vindex_t index = qbpp_new_var(name.c_str());
  if (index == qbpp::vindex_limit) {
    throw std::runtime_error(
        THROW_MESSAGE("License check failed: trial period expired or no valid "
                      "license found."));
  }
  return Var(index);
}
inline Var var() { return Var(qbpp_new_var(nullptr)); }
inline std::string str(Var var) {
  return std::string(qbpp_var_str(var.index()));
}
template <typename T, typename... Args,
          typename = typename std::enable_if<std::is_integral<T>::value>::type>
auto var(const std::string& var_str, T size, Args... args) {
  if constexpr (sizeof...(Args) == 0) {
    Vector<Var> vars;
    if (size == 1) {
      vars.emplace_back(var(var_str));
    } else {
      for (vindex_t i = 0; i < static_cast<size_t>(size); ++i) {
        vars.emplace_back(var(var_str + "[" + qbpp::str(i) + "]"));
      }
    }
    return vars;
  } else {
    Vector<decltype(var(var_str, args...))> vars;
    vars.reserve(static_cast<size_t>(size));
    for (size_t i = 0; i < static_cast<size_t>(size); ++i) {
      vars.emplace_back(var(var_str + "[" + qbpp::str(i) + "]",
                            static_cast<size_t>(args)...));
    }
    return vars;
  }
}
template <typename T, typename... Args,
          typename = typename std::enable_if<std::is_integral<T>::value>::type>
auto var(T size, Args... args) {
  return var("{" + std::to_string(new_unnamed_var_index()) + "}", size,
             args...);
}
inline VarIntCore var_int(const std::string& var_str) {
  return VarIntCore(var_str);
}
inline VarIntCore var_int() {
  return var_int("{" + std::to_string(new_unnamed_var_index()) + "}");
}
template <typename T, typename... Args>
auto var_int(const std::string& var_str, T size, Args... args) {
  if constexpr (sizeof...(args) == 0) {
    Vector<VarIntCore> vars;
    vars.reserve(static_cast<size_t>(size));
    for (size_t i = 0; i < static_cast<size_t>(size); ++i) {
      vars.emplace_back(VarIntCore(var_str + "[" + qbpp::str(i) + "]"));
    }
    return vars;
  } else {
    Vector<decltype(var_int(var_str, args...))> vars;
    vars.reserve(static_cast<size_t>(size));
    for (size_t i = 0; i < static_cast<size_t>(size); ++i) {
      vars.emplace_back(var_int(var_str + "[" + qbpp::str(i) + "]",
                                static_cast<size_t>(args)...));
    }
    return vars;
  }
}
inline VarOnehotCore var_onehot(const std::string& var_str) {
  return VarOnehotCore(var_str);
}
inline VarOnehotCore var_onehot() {
  return var_onehot("{" + std::to_string(new_unnamed_var_index()) + "}");
}
template <typename T, typename... Args>
auto var_onehot(const std::string& var_str, T size, Args... args) {
  if constexpr (sizeof...(args) == 0) {
    Vector<VarOnehotCore> vars;
    vars.reserve(static_cast<size_t>(size));
    for (size_t i = 0; i < static_cast<size_t>(size); ++i) {
      vars.emplace_back(VarOnehotCore(var_str + "[" + qbpp::str(i) + "]"));
    }
    return vars;
  } else {
    Vector<decltype(var_onehot(var_str, args...))> vars;
    vars.reserve(static_cast<size_t>(size));
    for (size_t i = 0; i < static_cast<size_t>(size); ++i) {
      vars.emplace_back(var_onehot(var_str + "[" + qbpp::str(i) + "]",
                                   static_cast<size_t>(args)...));
    }
    return vars;
  }
}
inline auto operator<=(energy_t lhs, VarOnehotCore&& rhs) {
  return std::make_pair(lhs, std::move(rhs));
}
inline auto operator<=(const std::pair<energy_t, VarOnehotCore> lhs,
                       energy_t rhs) {
  return qbpp::VarOnehot(lhs.second.var_str_, lhs.first, rhs);
}
inline std::vector<coeff_t> comp_coeffs(energy_t min_val, energy_t max_val,
                                        energy_t base_coeff = 1) {
  std::vector<coeff_t> coeffs;
  coeff_t val = static_cast<coeff_t>(max_val - min_val);
  if (val <= 0) {
    throw std::invalid_argument(THROW_MESSAGE(
        "max_val(", max_val, ") must be greater than min_val(", min_val, ")"));
  }
  coeff_t current_coeff = static_cast<coeff_t>(base_coeff);
  while (val > 0) {
    if (current_coeff < val) {
      coeffs.push_back(current_coeff);
    } else {
      coeffs.push_back(val);
    }
    val -= current_coeff;
    current_coeff *= 2;
  }
  return coeffs;
}
inline VarInt new_var_int(const std::string& var_str, energy_t min_val,
                          energy_t max_val, energy_t base_coeff = 1) {
  std::shared_ptr<std::vector<coeff_t>> coeffs_ptr =
      std::make_shared<std::vector<coeff_t>>(
          comp_coeffs(min_val, max_val, base_coeff));
  return VarInt(var_str, min_val, max_val, coeffs_ptr);
}
inline VarInt new_var_int(const VarIntCore& var_int_core, energy_t min_val,
                          energy_t max_val, energy_t base_coeff = 1) {
  return new_var_int(var_int_core.var_str_, min_val, max_val, base_coeff);
}
inline auto operator<=(energy_t lhs, VarIntCore&& rhs) {
  return std::make_pair(lhs, std::move(rhs));
}
template <typename T>
auto operator<=(energy_t lhs, Vector<T>&& rhs) {
  return std::make_pair(lhs, std::forward<Vector<T>>(rhs));
}
inline auto operator<=(const std::pair<energy_t, VarIntCore> lhs,
                       energy_t rhs) {
  return new_var_int(lhs.second.var_str_, lhs.first, rhs);
}
inline std::pair<energy_t, Vector<VarIntCore>> operator<=(
    energy_t lhs, const Vector<VarIntCore>& rhs) {
  return std::make_pair(lhs, rhs);
}
inline auto operator<=(const std::pair<energy_t, Vector<VarIntCore>> lhs,
                       energy_t rhs) {
  Vector<VarInt> result;
  result.reserve(lhs.second.size());
  for (const auto& item : lhs.second) {
    result.push_back(new_var_int(item, lhs.first, rhs));
  }
  return result;
}
void operator<=(energy_t , Var ) = delete;
void operator<=(Var , energy_t ) = delete;
inline Expr expr() { return Expr(); }
inline Vector<Expr> expr(vindex_t size) { return Vector<Expr>(size); }
template <typename T, typename... Args>
auto expr(T size, Args... args) {
  if constexpr (sizeof...(Args) == 0) {
    return Vector<Expr>(static_cast<size_t>(size));
  } else {
    Vector<decltype(expr(args...))> result(static_cast<size_t>(size));
    for (size_t i = 0; i < static_cast<size_t>(size); ++i)
      result[i] = expr(static_cast<size_t>(args)...);
    return result;
  }
}
template <typename T>
Expr toExpr(const T& arg) {
  return Expr(arg);
}
inline Expr toExpr(const Expr& arg) { return arg; }
inline Vector<Expr> toExpr(const Vector<Expr>& arg) { return arg; }
template <typename T>
auto toExpr(const Vector<T>& arg) {
  Vector<decltype(toExpr(arg[0]))> result;
  result.reserve(arg.size());
  for (const auto& item : arg) {
    result.push_back(toExpr(item));
  }
  return result;
}
template <typename T>
auto toExpr(const std::vector<T>& arg) {
  Vector<decltype(toExpr(arg[0]))> result;
  result.reserve(arg.size());
  for (const auto& item : arg) {
    result.push_back(toExpr(item));
  }
  return result;
}
template <typename T>
auto toExpr(const Vector<T>& arg) ->
    typename std::enable_if<std::is_same<T, decltype(toExpr(arg[0]))>::value,
                            Vector<T>>::type {
  return arg;
}
inline Vector<Expr> toExpr(const std::initializer_list<Expr>& list) {
  return Vector<Expr>(list);
}
template <typename T>
auto toExpr(const std::initializer_list<T>& list) {
  Vector<decltype(toExpr(std::declval<T>()))> result;
  result.reserve(list.size());
  for (const auto& item : list) {
    result.push_back(toExpr(item));
  }
  return result;
}
inline Vector<Vector<Expr>> toExpr(
    const std::initializer_list<std::initializer_list<Expr>>& list) {
  Vector<Vector<Expr>> result;
  result.reserve(list.size());
  for (const auto& item : list) {
    result.push_back(toExpr(item));
  }
  return result;
}
inline Vector<Vector<Vector<Expr>>> toExpr(
    const std::initializer_list<
        std::initializer_list<std::initializer_list<Expr>>>& list) {
  Vector<Vector<Vector<Expr>>> result;
  result.reserve(list.size());
  for (const auto& item : list) {
    result.push_back(toExpr(item));
  }
  return result;
}
inline Vector<Vector<Vector<Vector<Expr>>>> toExpr(
    const std::initializer_list<std::initializer_list<
        std::initializer_list<std::initializer_list<Expr>>>>& list) {
  Vector<Vector<Vector<Vector<Expr>>>> result;
  result.reserve(list.size());
  for (const auto& item : list) {
    result.push_back(toExpr(item));
  }
  return result;
}
inline Expr sqr(const Expr &expr) { return expr * expr; }
inline coeff_t gcd(const Expr &expr) {
  const auto &terms = expr.terms();
  tbb::combinable<energy_t> local_gcds([&] {
    return expr.constant();  
  });
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, terms.size()),
      [&](const tbb::blocked_range<size_t> &range) {
        energy_t &local = local_gcds.local();
        for (size_t i = range.begin(); i < range.end(); ++i) {
          const auto &term = terms[i];
          local = gcd(local, term.coeff());
        }
      });
  energy_t result = 0;
  local_gcds.combine_each([&](const energy_t &val) {
    result = gcd(result, val);
  });
  return static_cast<coeff_t>(abs(result));
}
inline ExprExpr operator==(const Expr& expr, energy_t val) {
  return ExprExpr(sqr(expr - val), expr);
}
inline ExprExpr comparison(const Expr& expr, energy_t minimum,
                           energy_t maximum) {
  energy_t val = maximum - minimum;
  if (val < 0) {
    throw std::runtime_error(THROW_MESSAGE("RHS (", maximum,
                                           ") must be greater than LHS (",
                                           minimum, ") in operator <=."));
  } else if (val == 0) {
    return ExprExpr((expr - minimum) == 0, expr);
  } else if (val == 1) {
    return ExprExpr((expr - minimum) * (expr - maximum), expr);
  } else {
    VarInt slack_var =
        new_var_int("{" + std::to_string(new_unnamed_var_index()) + "}",
                    minimum, maximum - 1, 2);
    return ExprExpr((expr - slack_var) * (expr - (slack_var + 1)), expr);
  }
}
inline std::pair<energy_t, Expr> operator<=(energy_t min_val,
                                            const Expr& expr) {
  return std::make_pair(min_val, expr);
}
inline ExprExpr operator<=(const std::pair<energy_t, Expr>& pair,
                           energy_t max_val) {
  energy_t min_val = pair.first;
  const Expr& result = pair.second;
  return comparison(result, min_val, max_val);
}
inline ExprExpr operator<=(const std::pair<energy_t, Expr>& pair, Inf val) {
  energy_t min_val = pair.first;
  const Expr& result = pair.second;
  if (val.is_negative()) {
    throw std::runtime_error(
        THROW_MESSAGE("RHS of operator <= must not be -inf."));
  }
  return comparison(result, min_val, result.pos_sum());
}
inline std::pair<energy_t, Expr> operator<=(Inf val, const Expr& expr) {
  if (val.is_positive()) {
    throw std::runtime_error(
        THROW_MESSAGE("LHS of operator <= must not be inf."));
  }
  return std::make_pair(expr.neg_sum(), expr);
}
inline void sort_vars_in_place(Vars& vars) {
#if MAXDEG == 2
  if (vars[0] > vars[1]) {
    std::swap(vars[0], vars[1]);
    return;
  }
#else
  std::size_t size = vars.size();
  switch (size) {
    case 0:
    case 1:
      break;
    case 2:
      if (vars[0] > vars[1]) std::swap(vars[0], vars[1]);
      break;
    case 3:
      if (vars[0] > vars[1]) std::swap(vars[0], vars[1]);
      if (vars[1] > vars[2]) std::swap(vars[1], vars[2]);
      if (vars[0] > vars[1]) std::swap(vars[0], vars[1]);
      break;
    case 4:
      if (vars[0] > vars[1]) std::swap(vars[0], vars[1]);
      if (vars[2] > vars[3]) std::swap(vars[2], vars[3]);
      if (vars[0] > vars[2]) std::swap(vars[0], vars[2]);
      if (vars[1] > vars[3]) std::swap(vars[1], vars[3]);
      if (vars[1] > vars[2]) std::swap(vars[1], vars[2]);
      break;
    default:
      for (size_t i = 1; i < size; ++i) {
        Var key = vars[i];
        size_t j = i;
        while (j > 0 && key < vars[j - 1]) {
          vars[j] = vars[j - 1];
          --j;
        }
        vars[j] = key;
      }
  }
#endif
}
inline void compare_swap_binary_vars(Var& a, Var& b) {
  if (a == b) {
    b = qbpp::impl::VarVoid;
  } else if (a > b) {
    std::swap(a, b);
  }
}
inline void compare_swap(Var& a, Var& b) {
  if (a > b) {
    std::swap(a, b);
  }
}
inline void unique_vars(size_t size, const Vars& src, Vars& dest) {
  auto prev = qbpp::impl::VarVoid;
  size_t j = 0;
  for (size_t i = 0; i < size; ++i) {
    if (src[i] != prev) {
      dest[j++] = src[i];
      prev = src[i];
    }
  }
  for (; j < size; ++j) {
    dest[j] = qbpp::impl::VarVoid;
  }
}
inline void erase_two_vars(size_t size, const Vars& src, Vars& dest) {
  auto prev = qbpp::impl::VarVoid;
  size_t j = 0;
  for (size_t i = 0; i < size; ++i) {
    if (src[i] != prev) {
      dest[j++] = src[i];
      prev = src[i];
    } else {
      if (j > 0) --j;
      prev = qbpp::impl::VarVoid;
    }
  }
  for (; j < size; ++j) {
    dest[j] = qbpp::impl::VarVoid;
  }
}
inline void sort_binary_vars_in_place(Vars& vars) {
#if MAXDEG == 2
  compare_swap_binary_vars(vars[0], vars[1]);
#else
  size_t size = vars.size();
  if (size == 0 || size == 1) {
    return;
  }
  if (size == 2) {
    compare_swap_binary_vars(vars[0], vars[1]);
    return;
  }
  if (size == 3) {
    compare_swap_binary_vars(vars[0], vars[1]);
    compare_swap_binary_vars(vars[1], vars[2]);
    compare_swap_binary_vars(vars[0], vars[1]);
    compare_swap_binary_vars(vars[1], vars[2]);
    return;
  }
  Vars temp = vars;
  if (size == 4) {
    compare_swap(temp[0], temp[1]);
    compare_swap(temp[2], temp[3]);
    compare_swap(temp[0], temp[2]);
    compare_swap(temp[1], temp[3]);
    compare_swap(temp[1], temp[2]);
    unique_vars(size, temp, vars);
    return;
  }
  for (size_t i = 1; i < size; ++i) {
    for (size_t j = i; j > 0; --j) {
      compare_swap(temp[j - 1], temp[j]);
    }
  }
  unique_vars(size, temp, vars);
#endif
}
inline void sort_spin_vars_in_place(Vars& vars) {
  Vars temp = vars;
  size_t size = vars.size();
  for (size_t i = 1; i < size; ++i) {
    for (size_t j = i; j > 0; --j) {
      compare_swap(temp[j - 1], temp[j]);
    }
  }
  erase_two_vars(size, temp, vars);
}
inline Vars sort_vars(const Vars& vars) {
  auto result = vars;
  sort_vars_in_place(result);
  return result;
}
inline Vars sort_vars_as_binary(const Vars& vars) {
#if defined(MAXDEG) && MAXDEG == 2
  if (vars[0] != qbpp::impl::VarVoid) {
    if (vars[1] != qbpp::impl::VarVoid) {
      if (vars[0] < vars[1]) {
        return vars;
      } else if (vars[0] == vars[1]) {
        return Vars(vars[0]);
      } else {
        return Vars(vars[1], vars[0]);
      }
    }
  }
  return vars;
#elif defined(MAXDEG)
  Vars result = vars;
  sort_binary_vars_in_place(result);
  return result;
#else
  Vars result = vars;
  sort_vars_in_place(result);
  if (result.size() >= 2)
    result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
#endif
}
inline Vars sort_vars_as_spin(const Vars& vars) {
#if defined(MAXDEG) && MAXDEG == 2
  if (vars[0] != qbpp::impl::VarVoid) {
    if (vars[1] != qbpp::impl::VarVoid) {
      if (vars[0] < vars[1]) {
        return vars;
      } else if (vars[0] == vars[1]) {
        return Vars(qbpp::impl::VarArray<MAXDEG>::EMPTY{});
      } else {
        return Vars(vars[1], vars[0]);
      }
    }
  }
  return vars;
#elif defined(MAXDEG)
  auto result = vars;
  sort_binary_vars_in_place(result);
  return result;
#else
  auto result = vars;
  sort_vars_in_place(result);
  if (result.size() >= 2) {
    for (auto it = result.begin();
         it != result.end() && it + 1 != result.end();) {
      if (*it == *(it + 1)) {
        it = result.erase(it, it + 2);
      } else {
        ++it;
      }
    }
  }
  return result;
#endif
}
inline Expr simplify_seq(const Expr& expr,
                         Vars (*sort_vars_func)(const Vars&) = sort_vars) {
  VarsCoeffMap vars_val_map;
  Expr result = expr.constant();
  for (const auto& term : expr.terms()) {
    auto new_vars = (*sort_vars_func)(term.vars());
    if (new_vars.size() != 0) {
      vars_val_map[new_vars] += term.coeff();
    } else {  
      result += term.coeff();
    }
  }
  for (const auto& [vars, val] : vars_val_map) {
    if (val != 0) {
      result += Term{vars, val};
    }
  }
  std::sort(result.terms().begin(), result.terms().end());
  return result;
}
inline Expr simplify(const Expr& expr,
                     Vars (*sort_vars_func)(const Vars&) = sort_vars) {
  if (expr.size() < SEQ_THRESHOLD) return simplify_seq(expr, sort_vars_func);
  Terms simplified_terms;
  simplified_terms.resize(expr.terms().size());
  tbb::parallel_for(size_t(0), expr.terms().size(), [&](size_t i) {
    simplified_terms[i] =
        Term(sort_vars_func(expr.terms()[i].vars()), expr.terms()[i].coeff());
  });
  tbb::parallel_sort(simplified_terms.begin(), simplified_terms.end());
  const size_t size = simplified_terms.size();
  constexpr size_t min_chunk_size = 32;
  size_t max_chunk_count = std::thread::hardware_concurrency();
  size_t chunk_count =
      std::min((size + min_chunk_size - 1) / min_chunk_size, max_chunk_count);
  size_t chunk_size = (size + chunk_count - 1) / chunk_count;
  std::vector<Terms> chunk(chunk_count);
  std::vector<energy_t> constant(chunk_count, 0);
  std::vector<size_t> indices(chunk_count);
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, chunk_count),
      [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
          size_t start = i * chunk_size;
          size_t end = std::min(start + chunk_size, size);
          for (size_t j = start; j < end; ++j) {
            if (simplified_terms[j].coeff() == 0) {
              continue;
            }
            if (simplified_terms[j].var_count() == 0) {
              constant[i] += simplified_terms[j].coeff();
              continue;
            }
            if (chunk[i].empty()) {
              chunk[i].push_back(std::move(simplified_terms[j]));
              continue;
            }
            if (chunk[i].back().vars() == simplified_terms[j].vars()) {
              chunk[i].back().coeff() += simplified_terms[j].coeff();
              if (chunk[i].back().coeff() == 0) {
                chunk[i].pop_back();
              }
              continue;
            }
            chunk[i].emplace_back(std::move(simplified_terms[j]));
          }
        }
      });
  for (size_t i = 1; i < chunk_count; ++i) {
    if (!chunk[i - 1].empty()) {
      if (chunk[i].empty()) {
        chunk[i].push_back(std::move(chunk[i - 1].back()));
        chunk[i - 1].pop_back();
        continue;
      }
      if (chunk[i - 1].back().vars() == chunk[i].front().vars()) {
        chunk[i].front().coeff() += chunk[i - 1].back().coeff();
        chunk[i - 1].pop_back();
        if (chunk[i].front().coeff() == 0) {
          chunk[i].erase(chunk[i].begin());
        }
      }
    }
  }
  std::vector<size_t> prefix(chunk_count + 1, 0);
  energy_t result_constant = expr.constant();
  for (size_t i = 0; i < chunk_count; ++i) {
    prefix[i + 1] = prefix[i] + chunk[i].size();
    result_constant += constant[i];
  }
  Terms result_terms;
  result_terms.resize(prefix.back());
  tbb::parallel_for(size_t(0), size_t(chunk_count), [&](size_t i) {
    std::copy(chunk[i].begin(), chunk[i].end(),
              result_terms.begin() + static_cast<std::ptrdiff_t>(prefix[i]));
  });
  return Expr(result_constant, std::move(result_terms));
}
template <typename T>
Vector<T> simplify(const Vector<T>& vec,
                   Vars (*sort_vars_func)(const Vars&) = sort_vars) {
  Vector<T> result(vec.size());
  tbb::parallel_for(size_t(0), vec.size(), [&](size_t i) {
    result[i] = simplify(vec[i], sort_vars_func);
  });
  return result;
}
inline Expr simplify_as_binary(const Expr& expr) {
  return simplify(expr, sort_vars_as_binary);
}
template <typename T>
auto simplify_as_binary(const Vector<T>& arg) {
  return simplify(arg, sort_vars_as_binary);
}
inline Expr simplify_as_spin(const Expr& expr) {
  return simplify(expr, sort_vars_as_spin);
}
template <typename T>
auto simplify_as_spin(const Vector<T>& arg) {
  return simplify(arg, sort_vars_as_spin);
}
inline bool is_simplified(const Expr& expr) {
  for (auto it = expr.terms().begin(); it != expr.terms().end(); ++it) {
    if (it->coeff() == 0) {
      return false;
    }
    for (auto it_var = it->vars().begin(); it_var != it->vars().end();
         ++it_var) {
      if (it_var != it->vars().begin() && *it_var < *(std::prev(it_var))) {
        return false;
      }
    }
    if (it != expr.terms().begin() && *it < *(std::prev(it))) {
      return false;
    }
  }
  return true;
}
inline bool is_binary(const Expr& expr) {
  for (const auto& term : expr.terms()) {
    if (term.var_count() > 2) {
      return false;
    }
    if (term.var_count() == 2 && term.vars()[0] == term.vars()[1]) {
      return false;
    }
  }
  return true;
}
inline MapList Sol::map_list() const {
  MapList result;
  for (vindex_t i = 0; i < iv_mapper_.var_count(); ++i) {
    result.push_back({var(i), static_cast<var_val_t>(get(i))});
  }
  return result;
}
inline Expr reduce_sum(const Term& term) {
  coeff_t num_var = static_cast<coeff_t>(term.var_count());
  if (num_var <= 2) {  
    return Expr{term};
  }
  Expr sum_vars =
      std::accumulate(term.vars().begin(), term.vars().end(), Expr(0));
  if (term.coeff() < 0) {  
    return term.coeff() * qbpp::var() * (sum_vars - (num_var - 1));
  }
  VarInt aux_var = new_var_int(
      "{" + std::to_string(new_unnamed_var_index()) + "}", 0, num_var - 2, 2);
  Expr result =
      term.coeff() * (sum_vars - aux_var) * (sum_vars - (aux_var + 1));
  return simplify_as_binary(result) / 2;
}
inline Expr reduce_cascade(const Term& term) {
  coeff_t var_count = static_cast<coeff_t>(term.var_count());
  if (var_count <= 2) {  
    return Expr{term};
  }
  Term temp = term;
  auto aux_var = qbpp::var();
  auto tail1 = temp.vars().back();
  temp.vars().pop_back();
  auto tail0 = temp.vars().back();
  temp.vars().pop_back();
  temp.vars().push_back(aux_var);
  return reduce_cascade(temp) +
         abs(term.coeff()) *
             (tail0 * tail1 - 2 * (tail0 + tail1) * aux_var + 3 * aux_var);
}
inline Expr reduce(const Expr& expr) {
  Expr result = expr.constant();
  for (const auto& term : expr.terms()) {
    if (term.coeff() < 0 || term.var_count() <= 4) {
      result += reduce_sum(term);
    } else
      result += reduce_cascade(term);
  }
  return result;
}
inline Expr binary_to_spin(const Expr& expr) {
  Expr result = 4 * expr.constant();
  for (const auto& term : expr.terms()) {
    if (term.var_count() == 0) {
      result += term.coeff();
    } else if (term.var_count() == 1) {
      result += 2 * term.coeff() * (term.vars()[0] + 1);
    } else if (term.var_count() == 2) {
      result += term.coeff() * (term.vars()[0] + 1) * (term.vars()[1] + 1);
    } else {
      throw std::runtime_error(
          THROW_MESSAGE("The term must be linear or quadratic, but degree is ",
                        term.var_count(), "."));
    }
  }
  return result.simplify_as_spin();
}
inline Expr spin_to_binary(const Expr& expr) {
  Expr result = expr.constant();
  for (const auto& term : expr.terms()) {
    const auto& vars = term.vars();
    Expr spin_expr = 1;
    for (const auto& v : vars) {
      spin_expr *= (2 * v - 1);
    }
    result += term.coeff() * spin_expr;
  }
  return result.simplify_as_binary();
}
template <typename T, typename U>
Vector<Expr> operator+(const Vector<T>& lhs, const Vector<U>& rhs) {
  Vector<Expr> result = lhs;
  result += rhs;
  return result;
}
template <typename T, typename U>
auto operator+(const Vector<Vector<T>>& lhs, const Vector<Vector<U>>& rhs) {
  Vector<decltype(lhs[0] + rhs[0])> result = lhs;
  result += rhs;
  return result;
}
template <typename T>
Vector<Expr> operator+(const Vector<T>& lhs, const Expr& rhs) {
  Vector<Expr> result = lhs;
  result += rhs;
  return result;
}
template <typename T>
Vector<Expr> operator+(Vector<T>&& lhs, const Expr& rhs) {
  lhs += rhs;
  return std::move(lhs);
}
template <typename T>
auto operator+(const Vector<Vector<T>>& lhs, const Expr& rhs) {
  Vector<decltype(lhs[0] + rhs)> result = lhs;
  result += rhs;
  return result;
}
template <typename T>
auto operator+(Vector<Vector<T>>&& lhs, const Expr& rhs) {
  lhs += rhs;
  return std::move(lhs);
}
template <typename T>
auto operator+(const Expr& lhs, const Vector<T>& rhs) {
  return rhs + lhs;
}
template <typename T>
auto operator+(const Expr& lhs, Vector<T>&& rhs) {
  rhs += lhs;
  return std::move(rhs);
}
template <typename T, typename U>
Vector<Expr> operator*(const Vector<T>& lhs, const Vector<U>& rhs) {
  Vector<Expr> result = lhs;
  result *= rhs;
  return result;
}
template <typename T, typename U>
auto operator*(const Vector<Vector<T>>& lhs, const Vector<Vector<U>>& rhs) {
  Vector<decltype(lhs[0] * rhs[0])> result = lhs;
  result *= rhs;
  return result;
}
template <typename T>
Vector<Expr> operator*(const Vector<T>& lhs, const Expr& rhs) {
  Vector<Expr> result = lhs;
  result *= rhs;
  return result;
}
template <typename T>
Vector<Expr> operator*(Vector<T>&& lhs, const Expr& rhs) {
  lhs *= rhs;
  return std::move(lhs);
}
template <typename T>
auto operator*(const Vector<Vector<T>>& lhs, const Expr& rhs) {
  Vector<decltype(lhs[0] * rhs)> result = lhs;
  result *= rhs;
  return result;
}
template <typename T>
auto operator*(Vector<Vector<T>>&& lhs, const Expr& rhs) {
  lhs *= rhs;
  return lhs;
}
template <typename T>
auto operator*(const Expr& lhs, const Vector<T>& rhs) {
  return rhs * lhs;
}
template <typename T>
auto operator*(const Expr& lhs, Vector<T>&& rhs) {
  return std::move(rhs) * lhs;
}
template <typename T, typename U>
Vector<Expr> operator-(const Vector<T>& lhs, const Vector<U>& rhs) {
  Vector<Expr> result = lhs;
  result -= rhs;
  return result;
}
template <typename T, typename U>
auto operator-(const Vector<Vector<T>>& lhs, const Vector<Vector<U>>& rhs) {
  Vector<decltype(lhs[0] - rhs[0])> result = lhs;
  result -= rhs;
  return result;
}
template <typename T>
Vector<Expr> operator-(const Vector<T>& lhs, const Expr& rhs) {
  Vector<Expr> result = lhs;
  result -= rhs;
  return result;
}
template <typename T>
Vector<Expr> operator-(Vector<T>&& lhs, const Expr& rhs) {
  lhs -= rhs;
  return std::move(lhs);
}
template <typename T>
auto operator-(const Vector<Vector<T>>& lhs, const Expr& rhs) {
  Vector<decltype(lhs[0] - rhs)> result = lhs;
  result -= rhs;
  return result;
}
template <typename T>
auto operator-(Vector<Vector<T>>&& lhs, const Expr& rhs) {
  lhs -= rhs;
  return std::move(lhs);
}
template <typename T>
auto operator-(const Expr& lhs, const Vector<T>& rhs) {
  return -(rhs - lhs);
}
template <typename T, class U,
          typename std::enable_if<std::is_convertible<U, coeff_t>::value,
                                  int>::type = 0>
auto operator/(const Vector<T>& lhs, U rhs) {
  Vector<decltype(lhs[0] / rhs)> result = lhs;
  result /= rhs;
  return result;
}
template <typename T>
auto operator+(const Vector<T>& lhs) {
  return qbpp::toExpr(lhs);
}
template <typename T>
auto operator-(const Vector<T>& lhs) {
  return lhs * (-1);
}
template <typename T>
auto operator==(const Vector<T>& lhs, energy_t rhs) {
  Vector<decltype(lhs[0] == rhs)> result = qbpp::sqr(lhs - rhs);
  return result;
}
template <typename T>
auto operator<=(energy_t lhs, const Vector<T>& rhs) {
  return std::make_pair(lhs, rhs);
}
template <typename T>
auto operator<=(Inf lhs, const Vector<T>& rhs) {
  return std::make_pair(lhs, rhs);
}
template <typename T>
auto operator<=(const std::pair<energy_t, Vector<T>>& lhs, energy_t rhs) {
  Vector<ExprExpr> result;
  result.resize(lhs.second.size());
  tbb::parallel_for(size_t(0), lhs.second.size(), [&](size_t i) {
    result[i] = (lhs.first <= lhs.second[i] <= rhs);
  });
  return result;
}
template <typename T>
auto operator<=(const std::pair<energy_t, Vector<T>>& lhs, Inf rhs) {
  Vector<ExprExpr> result;
  result.resize(lhs.second.size());
  tbb::parallel_for(size_t(0), lhs.second.size(), [&](size_t i) {
    result[i] = (lhs.first <= lhs.second[i] <= rhs);
  });
  return result;
}
template <typename T>
auto operator<=(const std::pair<Inf, Vector<T>>& lhs, energy_t rhs) {
  Vector<ExprExpr> result;
  result.resize(lhs.second.size());
  tbb::parallel_for(size_t(0), lhs.second.size(), [&](size_t i) {
    result[i] = (lhs.first <= lhs.second[i] <= rhs);
  });
  return result;
}
template <typename T>
auto operator<=(energy_t lhs, const Vector<Vector<T>>& rhs) {
  return std::make_pair(lhs, rhs);
}
template <typename T>
auto operator<=(const std::pair<energy_t, Vector<Vector<T>>>& lhs,
                energy_t rhs) {
  Vector<decltype(lhs.first <= lhs.second[0] <= rhs)> result;
  result.resize(lhs.second.size());
  tbb::parallel_for(size_t(0), lhs.second.size(), [&](size_t i) {
    result[i] = (lhs.first <= lhs.second[i] <= rhs);
  });
  return result;
}
template <typename T>
auto operator<=(const std::pair<energy_t, Vector<Vector<T>>>& lhs, Inf rhs) {
  Vector<Vector<T>> result;
  result.resize(lhs.second.size());
  tbb::parallel_for(size_t(0), lhs.second.size(), [&](size_t i) {
    result[i] = (lhs.first <= lhs.second[i] <= rhs);
  });
  return result;
}
template <typename T>
auto operator<=(const std::pair<Inf, Vector<Vector<T>>>& lhs, energy_t rhs) {
  Vector<decltype(lhs.second[0] <= rhs)> result;
  result.resize(lhs.second.size());
  tbb::parallel_for(size_t(0), lhs.second.size(), [&](size_t i) {
    result[i] = (lhs.first <= lhs.second[i] <= rhs);
  });
  return result;
}
template <typename T>
Expr sum(const T& arg [[maybe_unused]]) {
  throw std::runtime_error(
      "qbpp::sum cannot have a scalar argment. It must be a vector.");
  return toExpr(0);
}
template <typename T>
Expr sum(const Vector<Vector<T>>& arg [[maybe_unused]]) {
  throw std::runtime_error(
      "qbpp::sum cannot have a 2-d or higher array as an argument. It "
      "must be a 1-d vector. For 2-d or higher arrays, use qbpp::total_sum "
      "or "
      "qbpp::vector_sum.");
  return toExpr(0);
}
inline Expr sum(const Vector<Var>& vars) {
  Expr result;
  if (vars.size() < SEQ_THRESHOLD) {
    for (const auto& var : vars) {
      result += var;
    }
  } else {
    auto& terms = result.terms();
    terms.resize(vars.size());
    tbb::parallel_for(size_t(0), vars.size(),
                      [&](size_t i) { terms[i] = Term(vars[i]); });
  }
  return result;
}
inline Expr sum(const Vector<Expr>& expr) {
  Expr result;
  size_t n = expr.size();
  std::vector<size_t> indices(n + 1, 0);
  for (size_t i = 0; i < n; ++i) {
    indices[i + 1] = indices[i] + expr[i].term_count();
    result.constant() += expr[i].constant();
  }
  size_t total_terms = indices.back();
  result.terms().resize(total_terms);
  tbb::parallel_for(size_t(0), n, [&](size_t i) {
    auto& dst = result.terms();
    auto& src = expr[i].terms();
    std::copy(src.begin(), src.end(),
              dst.begin() + static_cast<std::ptrdiff_t>(indices[i]));
  });
  return result;
}
template <typename T>
Expr sum(const Vector<T>& items) {
  std::vector<Expr> result;
  result.reserve(items.size());
  for (const auto& item : items) {
    result.push_back(toExpr(item));
  }
  std::vector<size_t> indices(result.size() + 1);
  energy_t constant = 0;
  for (size_t i = 0; i < result.size(); ++i) {
    indices[i + 1] = indices[i] + result[i].term_count();
    constant += result[i].constant();
  }
  Terms terms;
  terms.resize(indices.back());
  tbb::parallel_for(size_t(0), result.size(), [&](size_t i) {
    std::copy(result[i].terms().begin(), result[i].terms().end(),
              terms.begin() + static_cast<std::ptrdiff_t>(indices[i]));
  });
  return Expr(constant, std::move(terms));
}
template <typename T>
Expr total_sum_impl(const T& item) {
  return toExpr(item);
}
template <typename T>
Expr total_sum_impl(const Vector<T>& items) {
  std::vector<Expr> result;
  result.resize(items.size());
  tbb::parallel_for(size_t(0), items.size(),
                    [&](size_t i) { result[i] = total_sum_impl(items[i]); });
  std::vector<size_t> indices(result.size() + 1);
  energy_t constant = 0;
  for (size_t i = 0; i < result.size(); ++i) {
    indices[i + 1] = indices[i] + result[i].term_count();
    constant += result[i].constant();
  }
  Terms terms;
  terms.resize(indices.back());
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, result.size()),
      [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
          std::copy(result[i].terms().begin(), result[i].terms().end(),
                    terms.begin() + static_cast<std::ptrdiff_t>(indices[i]));
        }
      });
  return Expr(constant, std::move(terms));
}
template <typename T>
qbpp::Expr total_sum(const T& arg [[maybe_unused]]) {
  throw std::runtime_error(
      "qbpp::total_sum cannot have a scalar argment. It must be a vector.");
  return toExpr(0);
}
template <typename T>
qbpp::Expr total_sum(const Vector<T>& arg [[maybe_unused]]) {
  throw std::runtime_error(
      "qbpp::total_sum cannot have a 1-d vector as an argument. For "
      "1-d vector, use qbpp::sum.");
  return toExpr(0);
}
template <typename T>
Expr total_sum(const Vector<Vector<T>>& items) {
  return total_sum_impl(items);
}
template <typename T>
Expr vector_sum(const T& items [[maybe_unused]]) {
  throw std::runtime_error(
      "qbpp::vector_sum function does not support scalar values.");
  return toExpr(0);
}
template <typename T>
Expr vector_sum(const Vector<T>& items [[maybe_unused]]) {
  throw std::runtime_error(
      "qbpp::vector_sum function does not support 1-d vectors. Use "
      "qbpp::sum instead.");
  return toExpr(0);
}
template <typename T>
auto vector_sum_impl(const T& items) {
  return toExpr(items);
}
template <typename T>
auto vector_sum_impl(const Vector<T>& items) {
  return sum(items);
}
template <typename T>
auto vector_sum_impl(const Vector<Vector<T>>& items) {
  using SubResultType = decltype(vector_sum_impl(std::declval<Vector<T>>()));
  size_t size = items.size();
  if (size < SEQ_THRESHOLD) {
    Vector<SubResultType> result(size);
    for (size_t i = 0; i < size; ++i) {
      result[i] = vector_sum_impl(items[i]);
    }
    return result;
  }
  Vector<SubResultType> result(size);
  tbb::parallel_for(size_t(0), size,
                    [&](size_t i) { result[i] = vector_sum_impl(items[i]); });
  return result;
}
template <typename T>
auto vector_sum(const Vector<Vector<T>>& items) {
  return vector_sum_impl(items);
}
template <typename T>
Expr product(const T& arg [[maybe_unused]]) {
  throw std::runtime_error(
      "qbpp::product cannot have a scalar argment. It must be a vector.");
  return toExpr(0);
}
template <typename T>
Expr product(const Vector<Vector<T>>& arg [[maybe_unused]]) {
  throw std::runtime_error(
      "qbpp::product cannot have a 2-d or higher array as an argument. It "
      "must be a 1-d vector. For 2-d or higher arrays, use "
      "qbpp::total_product "
      "or "
      "qbpp::vector_product.");
  return toExpr(0);
}
template <typename T>
Expr product(const Vector<T>& items) {
  tbb::combinable<Expr> local_products([] { return Expr{1}; });
  tbb::parallel_for(tbb::blocked_range<size_t>(0, items.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                      Expr& local = local_products.local();
                      for (size_t i = range.begin(); i < range.end(); ++i) {
                        local *= items[i];
                      }
                    });
  Expr result{1};
  local_products.combine_each([&](const Expr& val) { result *= val; });
  return result;
}
template <typename T>
Expr total_product_impl(const T& item) {
  return toExpr(item);
}
template <typename T>
Expr total_product_impl(const Vector<T>& items) {
  tbb::combinable<Expr> local_products([] { return Expr{1}; });
  tbb::parallel_for(tbb::blocked_range<size_t>(0, items.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                      Expr& local = local_products.local();
                      for (size_t i = range.begin(); i < range.end(); ++i) {
                        local *= total_product_impl(items[i]);
                      }
                    });
  Expr result{1};
  local_products.combine_each([&](const Expr& val) { result *= val; });
  return result;
}
template <typename T>
Expr total_product(const Vector<Vector<T>>& items) {
  return total_product_impl(items);
}
template <typename T>
Expr total_product(const T& arg [[maybe_unused]]) {
  throw std::runtime_error(
      "qbpp::total_product cannot have a scalar argment. It must be a "
      "vector.");
  return toExpr(0);
}
template <typename T>
Expr total_product(const Vector<T>& arg [[maybe_unused]]) {
  throw std::runtime_error(
      "qbpp::total_product cannot have a 1-d vector as an argument. For "
      "1-d vector, use qbpp::product.");
  return toExpr(0);
}
template <typename T>
Expr vector_product(const T& items [[maybe_unused]]) {
  throw std::runtime_error(
      "qbpp::vector_product function does not support scalar values.");
  return toExpr(0);
}
template <typename T>
Expr vector_product(const Vector<T>& items [[maybe_unused]]) {
  throw std::runtime_error(
      "qbpp::vector_product function does not support 1-d vectors. Use "
      "qbpp::product instead.");
  return toExpr(0);
}
template <typename T>
auto vector_product_impl(const T& items) {
  return toExpr(items);
}
template <typename T>
auto vector_product_impl(const Vector<T>& items) {
  return product(items);
}
template <typename T>
auto vector_product_impl(const Vector<Vector<T>>& items) {
  using SubResultType = decltype(vector_product(std::declval<Vector<T>>()));
  Vector<SubResultType> result(items.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, items.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                      for (size_t i = range.begin(); i < range.end(); ++i) {
                        result[i] = vector_product_impl(items[i]);
                      }
                    });
  return result;
}
template <typename T>
auto vector_product(const Vector<Vector<T>>& items) {
  return vector_product_impl(items);
}
template <typename T>
Expr diff(const Vector<T>& items, energy_t head = 0, energy_t tail = 0) {
  if (items.empty()) {
    return 0;
  }
  Expr result = sqr(head - items[0]);
  for (size_t i = 0; i < items.size() - 1; ++i) {
    result += sqr(items[i] - items[i + 1]);
  }
  result += sqr(items[items.size() - 1] - tail);
  return result;
}
template <typename T>
auto diff(const Vector<Vector<T>>& items, energy_t head = 0,
          energy_t tail = 0) {
  using SubResultType = decltype(diff(std::declval<Vector<T>>()));
  Vector<SubResultType> result(items.size());
  tbb::parallel_for(size_t(0), items.size(),
                    [&](size_t i) { result[i] = diff(items[i], head, tail); });
  return result;
}
inline energy_t toInt(const Expr& expr) {
  if (expr.term_count() == 0) {
    return static_cast<energy_t>(expr.constant());
  } else
    throw std::runtime_error(
        THROW_MESSAGE("The expression is not a constant."));
}
template <typename T>
auto toInt(const Vector<T>& arg) {
  using ResultType = decltype(toInt(arg[0]));
  Vector<ResultType> result(arg.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, arg.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                      for (size_t i = range.begin(); i < range.end(); ++i) {
                        result[i] = toInt(arg[i]);
                      }
                    });
  return result;
}
template <typename T>
Vector<energy_t> eval_var_val_map(const Vector<T>& arg,
                                  const VarValMap& var_val_map) {
  Vector<energy_t> result(arg.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, arg.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                      for (size_t i = range.begin(); i < range.end(); ++i) {
                        result[i] = eval_var_val_map(arg[i], var_val_map);
                      }
                    });
  return result;
}
template <typename T>
Vector<energy_t> eval(const Vector<T>& arg, const MapList& map_list) {
  VarValMap var_val_map = list_to_var_val(map_list);
  return eval_var_val_map(arg, var_val_map);
}
template <typename T>
auto eval_var_val_map(const Vector<Vector<T>>& arg,
                      const VarValMap& var_val_map) {
  using ResultType = decltype(eval_var_val_map(arg[0], var_val_map));
  Vector<ResultType> result(arg.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, arg.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                      for (size_t i = range.begin(); i < range.end(); ++i) {
                        result[i] = eval_var_val_map(arg[i], var_val_map);
                      }
                    });
  return result;
}
template <typename T>
auto eval(const Vector<Vector<T>>& arg, const MapList& map_list) {
  VarValMap var_val_map = list_to_var_val(map_list);
  return eval_var_val_map(arg, var_val_map);
}
template <typename T>
Vector<Expr> replace(const Vector<T>& arg, const MapList& map_list) {
  Vector<Expr> result(arg.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, arg.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                      for (size_t i = range.begin(); i < range.end(); ++i) {
                        result[i] = replace(arg[i], map_list);
                      }
                    });
  return result;
}
template <typename T>
auto replace(const Vector<Vector<T>>& arg, const MapList& map_list) {
  using ResultType = decltype(replace(arg[0], map_list));
  Vector<ResultType> result(arg.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, arg.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                      for (size_t i = range.begin(); i < range.end(); ++i) {
                        result[i] = replace(arg[i], map_list);
                      }
                    });
  return result;
}
template <typename T, typename F>
auto element_wise(const Vector<T>& arg, F func) {
  Vector<decltype(func(arg[0]))> result(arg.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, arg.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                      for (size_t i = range.begin(); i < range.end(); ++i) {
                        result[i] = func(arg[i]);
                      }
                    });
  return result;
}
template <typename T>
Vector<Expr> sqr(const Vector<T>& arg) {
  return element_wise(arg, [](const T& item) { return sqr(item); });
}
template <typename T>
auto sqr(const Vector<Vector<T>>& arg) -> Vector<decltype(sqr(arg[0]))> {
  return element_wise(arg, [](const Vector<T>& item) { return sqr(item); });
}
template <typename T>
Vector<Expr> reduce(const Vector<T>& arg) {
  return element_wise(arg, [](const T& item) { return reduce(item); });
}
template <typename T>
auto reduce(const Vector<Vector<T>>& arg) {
  return element_wise(arg, [](const Vector<T>& item) { return reduce(item); });
}
template <typename T>
Vector<Expr> binary_to_spin(const Vector<T>& arg) {
  return element_wise(arg, [](const T& item) { return binary_to_spin(item); });
}
template <typename T>
auto binary_to_spin(const Vector<Vector<T>>& arg)
    -> Vector<decltype(binary_to_spin(arg[0]))> {
  return element_wise(
      arg, [](const Vector<T>& item) { return binary_to_spin(item); });
}
template <typename T>
Vector<Expr> spin_to_binary(const Vector<T>& arg) {
  return element_wise(arg, [](const T& item) { return spin_to_binary(item); });
}
template <typename T>
auto spin_to_binary(const Vector<Vector<T>>& arg)
    -> Vector<decltype(spin_to_binary(arg[0]))> {
  return element_wise(
      arg, [](const Vector<T>& item) { return spin_to_binary(item); });
}
inline Vector<Expr> row(const Vector<Vector<Expr>>& vec, vindex_t index) {
  Vector<Expr> result = vec[index];
  return result;
}
inline Vector<Expr> col(const Vector<Vector<Expr>>& vec, size_t index) {
  const size_t n = vec.size();
  if (n < SEQ_THRESHOLD) {
    Vector<Expr> result;
    result.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      result.push_back(vec[i][index]);
    }
    return result;
  } else {
    Vector<Expr> result(n);
    tbb::parallel_for(size_t(0), n,
                      [&](size_t i) { result[i] = vec[i][index]; });
    return result;
  }
}
template <typename T>
coeff_t gcd(const Vector<T>& vec) {
  tbb::combinable<energy_t> local_gcds([] { return static_cast<energy_t>(0); });
  tbb::parallel_for(tbb::blocked_range<size_t>(0, vec.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                      energy_t& local = local_gcds.local();
                      for (size_t i = range.begin(); i < range.end(); ++i) {
                        local = gcd(local, gcd(vec[i]));
                      }
                    });
  energy_t result = 0;
  local_gcds.combine_each([&](energy_t val) { result = gcd(result, val); });
  return static_cast<coeff_t>(result);
}
template <typename T>
Vector<Vector<Expr>> transpose(const Vector<Vector<T>>& vec) {
  const size_t rows = vec.size();
  const size_t cols = vec[0].size();
  Vector<Vector<Expr>> result(cols, Vector<Expr>(rows));  
  tbb::parallel_for(size_t(0), rows, [&](size_t i) {
    for (size_t j = 0; j < cols; ++j) {
      result[j][i] = vec[i][j];
    }
  });
  return result;
}
inline int32_t onehot_to_int(const Vector<var_val_t>& vec) {
  int32_t total = 0;
  int32_t result = -1;
  for (vindex_t i = 0; i < vec.size(); i++) {
    if (vec[i] == 1) {
      result = static_cast<int32_t>(i);
    }
    total += vec[i];
  }
  if (total != 0) {
    return result;
  } else {
    return -1;
  }
}
template <typename T>
auto onehot_to_int(const Vector<Vector<T>>& vec) {
  using ResultType = decltype(onehot_to_int(vec[0]));
  Vector<ResultType> result(vec.size());
  tbb::parallel_for(size_t(0), vec.size(),
                    [&](size_t i) { result[i] = onehot_to_int(vec[i]); });
  return result;
}
template <typename T>
Vector<T>& Vector<T>::simplify(Vars (*sort_vars_func)(const Vars&)) {
  *this = qbpp::simplify(*this, sort_vars_func);
  return *this;
}
template <typename T>
Vector<T>& Vector<T>::replace(const MapList& map_list) {
  *this = qbpp::replace(*this, map_list);
  return *this;
}
template <typename T>
Vector<T>& Vector<T>::reduce() {
  *this = qbpp::reduce(*this);
  return *this;
}
inline var_val_t Var::operator()(Sol& sol) const { return sol.get(*this); }
inline IVMapper::IVMapper(const Expr& expr) {
  std::vector<uint8_t> var_flags(all_var_count(), 0);
  if (expr.term_count() == 0) {
    throw std::runtime_error(THROW_MESSAGE("Error: Empty expressoin."));
  };
  tbb::parallel_for(size_t(0), expr.terms().size(), [&](size_t i) {
    const auto& term = expr.terms()[i];
    for (const auto& var : term.vars()) {
      var_flags[var.index() & qbpp::vindex_mask] = 1;
    }
  });
  impl_ = qbpp_iv_mapper_create(all_var_count(), var_flags.data());
}
inline Expr& Expr::simplify(Vars (*sort_vars_func)(const Vars&)) {
  return *this = qbpp::simplify(*this, sort_vars_func);
}
inline Expr& Expr::simplify_as_binary() {
  return *this = qbpp::simplify_as_binary(*this);
}
inline Expr& Expr::simplify_as_spin() {
  return *this = qbpp::simplify_as_spin(*this);
}
inline Expr::Expr(const VarInt& var_int)
    : constant_(var_int.constant()), terms_(var_int.terms()) {
  if (var_int.max_val() == 0) {
    throw std::runtime_error(
        THROW_MESSAGE("Wrong VarInt object. Max value is not specified."));
  }
}
inline Expr::Expr(VarInt&& var_int)
    : constant_(var_int.constant()), terms_(std::move(var_int.terms())) {
  if (var_int.max_val() == 0) {
    throw std::runtime_error(
        THROW_MESSAGE("Wrong VarInt object. Max value is not specified."));
  }
}
inline Expr& Expr::replace(const MapList& map_list) {
  MapDict map_dict = list_to_dict(map_list);
  Vector<Expr> exprs;
  exprs.resize(terms_.size());
  tbb::parallel_for(size_t(0), terms_.size(), [&](size_t i) {
    exprs[i] = qbpp::replace(terms_[i], map_dict);
  });
  Expr result = qbpp::sum(exprs);
  result += constant_;
  *this = std::move(result);
  return *this;
}
inline vindex_t Expr::var_count() const {
  qbpp::BitVector bits(all_var_count());
  for (const auto& term : terms_) {
    for (const auto& var : term.vars()) {
      bits.set(var.index(), 1);
    }
  }
  return bits.popcount();
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
Term operator*(Var var, T val) {
  return Term(var, val);
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
Term operator*(T val, Var var) {
  return Term(var, val);
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
Term operator*(const Term& term, T val) {
  Term result = term;
  result *= val;
  return result;
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
Term operator*(T val, const Term& term) {
  Term result = term;
  result *= val;
  return result;
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
Term operator*(Term&& term, T val) {
  term *= val;
  return std::move(term);
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
Term operator*(T val, Term&& term) {
  term *= val;
  return std::move(term);
}
inline Term operator*(Var var1, Var var2) { return Term(var1, var2); }
inline Term operator*(const Term& term, Var var) {
  Term result = term;
  result *= var;
  return result;
}
inline Term operator*(Term&& term, Var var) {
  term *= var;
  return std::move(term);
}
inline Term operator*(Var var, const Term& term) { return term * var; }
inline Term operator*(Var var, Term&& term) { return std::move(term) * var; }
inline Term operator*(const Term& term1, const Term& term2) {
  Term result = term1;
  result.coeff() *= term2.coeff();
#ifdef MAXDEG
  result.vars() *= term2.vars();
#else
  result.vars().insert(result.vars().end(), term2.vars().begin(),
                       term2.vars().end());
#endif
  return result;
}
inline Term operator*(Term&& term1, const Term& term2) {
  term1.coeff() *= term2.coeff();
#ifdef MAXDEG
  term1.vars() *= term2.vars();
#else
  term1.vars().insert(term1.vars().end(), term2.vars().begin(),
                      term2.vars().end());
#endif
  return term1;
}
inline Term operator*(const Term& term1, Term&& term2) {
  return std::move(term2) * term1;
}
inline Term operator*(Term&& term1, Term&& term2) {
  return std::move(term1) * term2;
}
template <typename T>
Terms operator+(const Terms& lhs, T&& rhs) {
  Terms result = lhs;
  result += std::forward<T>(rhs);
  return result;
}
template <typename T>
Terms operator+(Terms&& lhs, T&& rhs) {
  lhs += std::forward<T>(rhs);
  return std::move(lhs);
}
template <typename T>
Terms operator-(const Terms& lhs, T&& rhs) {
  Terms result = lhs;
  result -= std::forward<T>(rhs);
  return result;
}
template <typename T>
Terms operator-(Terms&& lhs, T&& rhs) {
  lhs -= std::forward<T>(rhs);
  return std::move(lhs);
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
Terms operator*(const Terms& lhs, T rhs) {
  Terms result;
  if (rhs == 0) {
    return result;
  }
  const size_t n = lhs.size();
  if (n < SEQ_THRESHOLD) {
    result.reserve(n);
    for (const auto& term : lhs) {
      result += term * rhs;
    }
  } else {
    result.resize(n);
    tbb::parallel_for(size_t(0), n,
                      [&](size_t i) { result[i] = lhs[i] * rhs; });
  }
  return result;
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
Terms operator*(Terms&& lhs, T rhs) {
  if (rhs == 0) {
    return Terms{};
  } else {
    lhs *= rhs;
  }
  return std::move(lhs);
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
inline Terms operator*(T rhs, const Terms& lhs) {
  return lhs * rhs;
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
inline Terms operator*(T rhs, Terms&& lhs) {
  return lhs * rhs;
}
inline Terms operator*(const Terms& lhs, const Term& rhs) {
  Terms result;
  const size_t n = lhs.size();
  if (n < SEQ_THRESHOLD) {
    result.reserve(n);
    for (const auto& term : lhs) {
      result += term * rhs;
    }
  } else {
    result.resize(n);
    tbb::parallel_for(size_t(0), n,
                      [&](size_t i) { result[i] = lhs[i] * rhs; });
  }
  return result;
}
inline Terms operator*(const Term& lhs, const Terms& rhs) { return lhs * rhs; }
inline Terms operator*(Terms&& lhs, const Term& rhs) {
  lhs *= rhs;
  return std::move(lhs);
}
inline Terms operator*(const Term& lhs, Terms&& rhs) {
  rhs *= lhs;
  return std::move(rhs);
}
inline Terms operator*(const Terms& lhs, const Terms& rhs) {
  Terms result;
  const size_t n = lhs.size(), m = rhs.size();
  if (n * m < SEQ_THRESHOLD) {
    result.reserve(n * m);
    for (const auto& term : rhs) {
      result += lhs * term;
    }
  } else {
    result.resize(n * m);
    const Terms& small_term = (n <= m) ? lhs : rhs;
    const Terms& large_term = (n > m) ? lhs : rhs;
    tbb::parallel_for(size_t(0), large_term.size(), [&](size_t i) {
      for (size_t j = 0; j < small_term.size(); ++j) {
        result[i * small_term.size() + j] = large_term[i] * small_term[j];
      }
    });
  }
  return result;
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
Expr operator*(const Expr& expr, T val) {
  Expr result = expr;
  result *= val;
  return result;
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
Expr operator*(T val, const Expr& expr) {
  return expr * val;
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
Expr operator*(Expr&& expr, T val) {
  expr *= val;
  return std::move(expr);
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
Expr operator*(T val, Expr&& expr) {
  return std::move(expr) * val;
}
inline Expr operator*(const Expr& expr, const Term& term) {
  Expr result = expr;
  result *= term;
  return result;
}
inline Expr operator*(const Term& term, const Expr& expr) {
  Expr result = expr;
  result *= term;
  return result;
}
inline Expr operator*(const Expr& expr, Term&& term) {
  Expr result = expr;
  result *= std::move(term);
  return result;
}
inline Expr operator*(Term&& term, const Expr& expr) {
  Expr result = expr;
  result *= std::move(term);
  return result;
}
inline Expr&& operator*(Expr&& expr, const Term& term) {
  expr *= term;
  return std::move(expr);
}
inline Expr&& operator*(const Term& term, Expr&& expr) {
  expr *= term;
  return std::move(expr);
}
inline Expr&& operator*(Expr&& expr, Term&& rhs) {
  expr *= std::move(rhs);
  return std::move(expr);
}
inline Expr&& operator*(Term&& rhs, Expr&& expr) {
  expr *= std::move(rhs);
  return std::move(expr);
}
inline Expr&& operator*(Expr&& lhs, Expr&& rhs) {
  if (lhs.term_count() >= rhs.term_count()) {
    lhs *= rhs;
    return std::move(lhs);
  } else {
    rhs *= lhs;
    return std::move(rhs);
  }
}
inline Expr&& operator*(Expr&& lhs, const Expr& rhs) {
  lhs *= rhs;
  return std::move(lhs);
}
inline Expr&& operator*(const Expr& lhs, Expr&& rhs) {
  rhs *= lhs;
  return std::move(rhs);
}
inline Expr operator*(const Expr& lhs, const Expr& rhs) {
  auto result = lhs;
  result *= rhs;
  return result;
}
inline Expr&& operator+(Expr&& lhs, Expr&& rhs) {
  if (lhs.term_count() >= rhs.term_count()) {
    lhs += rhs;
    return std::move(lhs);
  } else {
    rhs += lhs;
    return std::move(rhs);
  }
}
inline Expr&& operator+(Expr&& lhs, const Expr& rhs) {
  lhs += rhs;
  return std::move(lhs);
}
inline Expr&& operator+(const Expr& lhs, Expr&& rhs) {
  rhs += lhs;
  return std::move(rhs);
}
inline Expr operator+(const Expr& lhs, const Expr& rhs) {
  auto result = lhs;
  return result += rhs;
}
inline Expr&& operator-(Expr&& lhs, Expr&& rhs) {
  lhs -= rhs;
  return std::move(lhs);
}
inline Expr&& operator-(Expr&& lhs, const Expr& rhs) {
  lhs -= rhs;
  return std::move(lhs);
}
inline Expr&& operator-(const Expr& lhs, Expr&& rhs) {
  rhs *= -1;
  rhs += lhs;
  return std::move(rhs);
}
inline Expr operator-(const Expr& lhs, const Expr& rhs) {
  auto result = lhs;
  return result -= rhs;
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
inline Expr&& operator/(Expr&& expr, T val) {
  expr /= val;
  return std::move(expr);
}
template <typename T, typename std::enable_if<
                          std::is_convertible<T, coeff_t>::value, int>::type>
inline Expr operator/(const Expr& expr, T val) {
  auto temp = expr;
  return temp /= val;
}
inline Expr operator+(Expr&& expr) { return std::move(expr); }
inline const Expr& operator+(const Expr& expr) { return expr; }
inline Expr operator-(Expr&& expr) {
  expr.constant() = -expr.constant();
  if (expr.term_count() < SEQ_THRESHOLD) {
    for (auto& term : expr.terms()) {
      term.negate();
    }
  } else {
    tbb::parallel_for(size_t(0), expr.size(),
                      [&](size_t i) { expr.term(i).negate(); });
  }
  return std::move(expr);
}
inline Expr operator-(const Expr& expr) {
  Expr result{-expr.constant()};
  const size_t n = expr.size();
  if (n < SEQ_THRESHOLD) {
    result.terms().reserve(n);
    for (const auto& term : expr.terms()) {
      result.terms().emplace_back(-term);
    }
  } else {
    result.terms().resize(n);
    tbb::parallel_for(size_t(0), n,
                      [&](size_t i) { result.term(i) = -expr.term(i); });
  }
  return result;
}
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
Expr operator+(const Term& term, T val) {
  return Expr{term, val};
}
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
Expr operator+(Term&& term, T val) {
  return Expr{std::move(term), val};
}
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
Expr operator+(T val, Term&& term) {
  return Expr{std::move(term), val};
}
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
Expr operator+(T val, const Term& term) {
  return Expr{term, val};
}
inline Expr operator+(const Term& lhs, const Term& rhs) {
  return Expr{lhs, rhs};
}
inline Expr operator+(const Term& lhs, Term&& rhs) {
  return Expr{lhs, std::move(rhs)};
}
inline Expr operator+(Term&& lhs, const Term& rhs) {
  return Expr{std::move(lhs), rhs};
}
inline Expr operator+(Term&& lhs, Term&& rhs) {
  return Expr(std::move(lhs), std::move(rhs));
}
inline Expr operator-(const Term& lhs, const Term& rhs) {
  Expr result{lhs};
  result -= rhs;
  return result;
}
inline Expr operator-(const Term& lhs, Term&& rhs) {
  Expr result{lhs};
  result -= std::move(rhs);
  return result;
}
inline Expr operator-(Term&& lhs, const Term& rhs) {
  Expr result{std::move(lhs)};
  result -= rhs;
  return result;
}
inline Expr operator-(Term&& lhs, Term&& rhs) {
  Expr result{std::move(lhs)};
  result -= std::move(rhs);
  return result;
}
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
Expr operator+(Var var, T val) {
  return Expr(var, static_cast<energy_t>(val));
}
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
Expr operator+(T val, Var var) {
  return var + val;
}
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
Expr operator-(Var var, T val) {
  return Expr(var, static_cast<energy_t>(-val));
}
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
Expr operator-(T val, Var var) {
  return Expr{Term(var, static_cast<coeff_t>(-1)) + val};
}
inline Expr operator+(Var lhs, Var rhs) { return Term(lhs) + Term(rhs); }
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
Expr operator-(const Expr& expr, T val) {
  return expr + (-val);
}
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
inline Expr operator-(Expr&& expr, T val) {
  return expr + (-val);
}
inline Expr operator+(const Expr& expr, const Term& term) {
  Expr result = expr;
  return result += term;
}
inline Expr operator+(Expr&& expr, const Term& term) { return expr += term; }
inline Expr operator+(const Expr& expr, Term&& term) {
  Expr result = expr;
  return result += std::move(term);
}
inline Expr operator+(Expr&& expr, Term&& term) {
  return expr += std::move(term);
}
inline Expr operator+(const Term& term, const Expr& expr) {
  Expr result = expr;
  return result += term;
}
inline Expr operator+(const Term& term, Expr&& expr) { return expr += term; }
inline Expr operator+(Term&& term, const Expr& expr) {
  Expr result = expr;
  return result += std::move(term);
}
inline Expr operator+(Term&& term, Expr&& expr) {
  return expr += std::move(term);
}
inline Expr operator-(const Term& term, const Expr& expr) {
  return Expr(term) - expr;
}
inline Expr operator-(Term&& term, const Expr& expr) {
  return Expr(std::move(term)) - expr;
}
inline Expr operator-(const Term& term, Expr&& expr) {
  expr.negate();
  return expr += term;
}
inline Expr operator-(Term&& term, Expr&& expr) {
  expr.negate();
  return expr += std::move(term);
}
inline Expr operator-(const Expr& expr, const Term& term) {
  Expr result = expr;
  return result -= term;
}
inline Expr operator-(Expr&& expr, const Term& term) {
  expr -= term;
  return std::move(expr);
}
inline Expr operator-(const Expr& expr, Term&& term) {
  Expr result = expr;
  return result -= std::move(term);
}
inline Expr operator-(Expr&& expr, Term&& term) {
  expr -= std::move(term);
  return std::move(expr);
}
inline Expr replace(const Term& term, const MapDict& map_dict) {
  Expr result{term.coeff()};
  for (auto item : term.vars()) {
    vindex_t index = item.index();
    if (index & qbpp::vindex_neg_bit) {
      throw std::runtime_error(
          THROW_MESSAGE("The replace function cannot be applied to variables "
                        "with a negation operator (~)."));
    } else {
      auto it = map_dict.find(item);
      if (it == map_dict.end()) {
        result *= item;
      } else {
        result *= it->second;
      }
    }
  }
  return result;
}
inline Expr VarOnehot::operator*() const { return sqr(sum(vars_) - 1); }
inline std::tuple<bool, size_t, size_t, coeff_t, coeff_t>
check_if_simplified_as_binary(const Expr& expr) {
  if (expr.term_count() == 0) {
    return {false, 0, 0, 0, 0};
  }
  const auto& terms = expr.terms();
  bool error = false;
  size_t linear_count = 0;
  coeff_t min_coeff = terms[0].coeff();
  coeff_t max_coeff = terms[0].coeff();
  std::mutex mtx;
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, terms.size()),
      [&](const tbb::blocked_range<size_t>& range) {
        coeff_t local_min = min_coeff;
        coeff_t local_max = max_coeff;
        for (size_t i = range.begin(); i < range.end(); ++i) {
          const auto& term = terms[i];
          if (local_min > term.coeff()) local_min = term.coeff();
          if (local_max < term.coeff()) local_max = term.coeff();
          if (term.var_count() == 0 || term.var_count() >= 3) {
            error = true;
            return;
          }
          if (term.var_count() == 2 && term.vars()[1] <= term.vars()[0]) {
            error = true;
            return;
          }
          if (i > 0) {
            if (term <= terms[i - 1]) {
              error = true;
              return;
            }
            if (terms[i - 1].var_count() == 1 && term.var_count() == 2) {
              linear_count = i;
            }
          }
        }
        std::lock_guard<std::mutex> lock(mtx);
        if (local_min < min_coeff) min_coeff = local_min;
        if (local_max > max_coeff) max_coeff = local_max;
      });
  if (terms.back().var_count() == 1) {
    linear_count = terms.size();
  }
  return {error, linear_count, terms.size() - linear_count, min_coeff,
          max_coeff};
}
template <size_t K, class Model, class Term, class Buckets>
inline void add_term_rotations(const Model& model, const Term& term,
                               Buckets& buckets) {
  static_assert(K >= 2, "K must be >= 2");
  for (vindex_t i = 0; i < K; ++i) {
    vindex_t idx[K - 1];
    for (vindex_t j = 0; j < K - 1; ++j) {
      idx[j] = model.index(term.var((i + 1 + j) % K));
    }
    buckets[model.index(term.var(i))].emplace_back(idx, term.coeff());
  }
}
inline HuboModel::all_hubo_terms HuboModel::helper_func(
    const qbpp::Model& model) {
  uint32_t max_power = 0;
  std::vector<qbpp::coeff_t> term1(model.var_count(), 0);
  std::vector<TermVector<1>> term2(model.var_count());
  std::vector<TermVector<2>> term3(model.var_count());
  std::vector<TermVector<3>> term4(model.var_count());
  std::vector<TermVector<4>> term5(model.var_count());
  std::vector<TermVector<5>> term6(model.var_count());
  std::vector<TermVector<6>> term7(model.var_count());
  std::vector<TermVector<7>> term8(model.var_count());
  std::vector<TermVector<8>> term9(model.var_count());
  std::vector<TermVector<9>> term10(model.var_count());
  std::vector<TermVector<10>> term11(model.var_count());
  std::vector<TermVector<11>> term12(model.var_count());
  std::vector<TermVector<12>> term13(model.var_count());
  std::vector<TermVector<13>> term14(model.var_count());
  std::vector<TermVector<14>> term15(model.var_count());
  std::vector<TermVector<15>> term16(model.var_count());
  for (const auto& term : model.expr().terms()) {
    if (term.var_count() > max_power) {
      max_power = term.var_count();
    }
    switch (term.var_count()) {
      case 0:
        break;
      case 1:
        if (term.coeff() != 0) {
          if (term.var(0).index() & qbpp::vindex_neg_bit) {
            term1[model.index(Var(term.var(0).index() & qbpp::vindex_mask))] -=
                term.coeff();
          } else {
            term1[model.index(term.var(0))] += term.coeff();
          }
        }
        break;
      case 2:
        add_term_rotations<2>(model, term, term2);
        break;
      case 3:
        add_term_rotations<3>(model, term, term3);
        break;
      case 4:
        add_term_rotations<4>(model, term, term4);
        break;
      case 5:
        add_term_rotations<5>(model, term, term5);
        break;
      case 6:
        add_term_rotations<6>(model, term, term6);
        break;
      case 7:
        add_term_rotations<7>(model, term, term7);
        break;
      case 8:
        add_term_rotations<8>(model, term, term8);
        break;
      case 9:
        add_term_rotations<9>(model, term, term9);
        break;
      case 10:
        add_term_rotations<10>(model, term, term10);
        break;
      case 11:
        add_term_rotations<11>(model, term, term11);
        break;
      case 12:
        add_term_rotations<12>(model, term, term12);
        break;
      case 13:
        add_term_rotations<13>(model, term, term13);
        break;
      case 14:
        add_term_rotations<14>(model, term, term14);
        break;
      case 15:
        add_term_rotations<15>(model, term, term15);
        break;
      case 16:
        add_term_rotations<16>(model, term, term16);
        break;
      default:
        throw std::runtime_error(
            THROW_MESSAGE("The HuboModel only supports up to degree 16."));
    }
  }
  std::vector<energy_t> coeff_sum(model.var_count(), 0);
  for (vindex_t i = 0; i < model.var_count(); ++i) {
    coeff_sum[i] = term1[i];
    for (size_t j = 0; j < term2[i].size(); ++j)
      coeff_sum[i] += term2[i].coeff(j);
    for (size_t j = 0; j < term3[i].size(); ++j)
      coeff_sum[i] += term3[i].coeff(j);
    for (size_t j = 0; j < term4[i].size(); ++j)
      coeff_sum[i] += term4[i].coeff(j);
    for (size_t j = 0; j < term5[i].size(); ++j)
      coeff_sum[i] += term5[i].coeff(j);
    for (size_t j = 0; j < term6[i].size(); ++j)
      coeff_sum[i] += term6[i].coeff(j);
    for (size_t j = 0; j < term7[i].size(); ++j)
      coeff_sum[i] += term7[i].coeff(j);
    for (size_t j = 0; j < term8[i].size(); ++j)
      coeff_sum[i] += term8[i].coeff(j);
    for (size_t j = 0; j < term9[i].size(); ++j)
      coeff_sum[i] += term9[i].coeff(j);
    for (size_t j = 0; j < term10[i].size(); ++j)
      coeff_sum[i] += term10[i].coeff(j);
    for (size_t j = 0; j < term11[i].size(); ++j)
      coeff_sum[i] += term11[i].coeff(j);
    for (size_t j = 0; j < term12[i].size(); ++j)
      coeff_sum[i] += term12[i].coeff(j);
    for (size_t j = 0; j < term13[i].size(); ++j)
      coeff_sum[i] += term13[i].coeff(j);
    for (size_t j = 0; j < term14[i].size(); ++j)
      coeff_sum[i] += term14[i].coeff(j);
    for (size_t j = 0; j < term15[i].size(); ++j)
      coeff_sum[i] += term15[i].coeff(j);
    for (size_t j = 0; j < term16[i].size(); ++j)
      coeff_sum[i] += term16[i].coeff(j);
  }
  energy_t all_coeff_sum = model.constant();
  for (const auto& term : model.expr().terms()) {
    all_coeff_sum += term.coeff();
  }
  return std::make_tuple(
      max_power, model.constant(), std::move(term1), std::move(term2),
      std::move(term3), std::move(term4), std::move(term5), std::move(term6),
      std::move(term7), std::move(term8), std::move(term9), std::move(term10),
      std::move(term11), std::move(term12), std::move(term13),
      std::move(term14), std::move(term15), std::move(term16), all_coeff_sum,
      std::move(coeff_sum));
}
inline std::string HuboModel::str() const {
  std::ostringstream oss;
  bool is_first = true;
  oss << "(constant) " << constant() << "\n";
  for (vindex_t i = 0; i < term1_.size(); ++i) {
    if (term1_[i] != 0) {
      if (is_first) {
        oss << "(degree 1)\n";
        is_first = false;
      }
      oss << var(i) << "*" << term1_[i] << "\n";
    }
  }
  is_first = true;
  for (vindex_t i = 0; i < term2_.size(); ++i) {
    auto& t2 = term2_[i];
    for (size_t j = 0; j < t2.size(); ++j) {
      if (is_first) {
        oss << "(degree 2)\n";
        is_first = false;
      }
      oss << var(i) << "*" << var(t2.indices(j)[0]) << "*" << t2.coeff(j)
          << "\n";
    }
  }
  is_first = true;
  for (vindex_t i = 0; i < term3_.size(); ++i) {
    auto& t3 = term3_[i];
    for (size_t j = 0; j < t3.size(); ++j) {
      if (is_first) {
        oss << "(degree 3)\n";
        is_first = false;
      }
      oss << var(i) << "*" << var(t3.indices(j)[0]) << "*"
          << var(t3.indices(j)[1]) << "*" << t3.coeff(j) << "\n";
    }
  }
  return oss.str();
}
inline double time() {
  static const auto start_ = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::high_resolution_clock::now() - start_;
  return std::chrono::duration<double>(elapsed).count();
}
inline Vector<Expr> operator*(const Vector<ExprExpr>& arg) {
  Vector<Expr> result(arg.size());
  tbb::parallel_for(size_t(0), arg.size(),
                    [&](size_t i) { result[i] = *arg[i]; });
  return result;
}
template <typename T>
auto operator*(const Vector<Vector<T>>& arg) {
  using InnerResultType = decltype(operator*(*std::declval<Vector<T>&>()));
  Vector<InnerResultType> result(arg.size());
  tbb::parallel_for(size_t(0), arg.size(),
                    [&](size_t i) { result[i] = *arg[i]; });
  return result;
}
inline Vector<Expr> operator*(const Vector<Var>& lhs, const Vector<Var>& rhs) {
  Vector<Expr> result;
  result.resize(lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    result.push_back(qbpp::Expr(qbpp::Term(lhs[i], rhs[i])));
  }
  return result;
}
inline void queueset_capacity(uint32_t capacity) {
  qbpp_queueset_capacity(capacity);
}
inline size_t queueset_size() { return qbpp_queueset_size(); }
inline bool queueset_has(const BitVector& bv) {
  return qbpp_queueset_has(static_cast<const void*>(bv.impl())) == 1;
}
inline bool queueset_insert(const BitVector& bv) {
  return qbpp_queueset_insert(static_cast<const void*>(bv.impl())) == 1;
}
}  
