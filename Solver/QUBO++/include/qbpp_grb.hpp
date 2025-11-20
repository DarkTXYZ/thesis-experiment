/// @file  qbpp_grb.hpp
/// @brief QUBO++ interface to call Gurobi Optimizer
/// @details This file provides interfaces to call Gurobi Optimizer from QUBO++
/// @author Koji Nakano
/// @version 2025.05.21

#pragma once
#include <cmath>
#include "gurobi_c++.h"
#include "qbpp.hpp"
#define GRB_SAFE_CALL(func)                                               \
  try {                                                                   \
    func;                                                                 \
  } catch (GRBException e) {                                              \
    std::cerr << e.getErrorCode() << ": " << e.getMessage() << std::endl; \
    exit(1);                                                              \
  }
namespace qbpp_grb {
class HuboModel;
class Sol;
class Callback;
class HuboModel : public qbpp::HuboModel {
 protected:
  GRBEnv grb_env;
  std::shared_ptr<GRBModel> grb_model_ptr;
  GRBVar *grb_x;
 public:
  HuboModel(const qbpp::HuboModel &hubo_model, bool verbose = false);
  HuboModel(const HuboModel &grb_model) = default;
  void set(const std::string &key, const std::string &val) {
    grb_model_ptr->getEnv().set(key, val);
  };
  void time_limit(uint32_t time_limit) {
    set("TimeLimit", std::to_string(time_limit));
  }
  void set(Callback &cb);
  GRBVar grb_var(qbpp::vindex_t index) const { return grb_x[index]; };
  GRBModel &grb_model() { return *grb_model_ptr; }
  const GRBModel &grb_model() const { return *grb_model_ptr; }
  Sol optimize();
  void write(std::string filename) {
    GRB_SAFE_CALL(grb_model_ptr->write(filename));
  }
};
class Sol : public qbpp::Sol {
  qbpp::energy_t bound_;
 public:
  Sol(const qbpp::HuboModel &hubo_model) : qbpp::Sol(hubo_model) {}
  qbpp::energy_t bound() const { return bound_; }
  qbpp::energy_t bound(qbpp::energy_t eval) {
    bound_ = eval;
    return bound_;
  }
};
class Callback : public GRBCallback {
 protected:
  const HuboModel hubo_model_;
  const GRBModel &grb_model_;
  std::optional<qbpp::energy_t> target_energy_ = std::nullopt;
  mutable std::mutex mtx_;
  void abort_if_target_energy(qbpp::energy_t energy) {
    if (target_energy_.has_value() && energy <= target_energy_.value()) {
      abort();
    }
  }
 public:
  Callback(const HuboModel &hubo_model)
      : hubo_model_(hubo_model), grb_model_(hubo_model.grb_model()) {
    qbpp::time();
  };
  virtual ~Callback() = default;
  Sol sol();
  virtual void callback() override {
    if (where == GRB_CB_MIPSOL) {
      qbpp::energy_t energy = sol().energy();
      std::cout << "TTS = " << std::fixed << std::setprecision(3)
                << std::setfill('0') << qbpp::time() << "s Energy = " << energy
                << std::endl;
      abort_if_target_energy(energy);
    }
  }
  void target_energy(qbpp::energy_t target_energy) {
    target_energy_ = target_energy;
  }
  double getDoubleInfoPublic(int what) { return getDoubleInfo(what); }
  double getSolutionPublic(GRBVar v) { return getSolution(v); }
};
inline HuboModel::HuboModel(const qbpp::HuboModel &hubo_model, bool verbose)
    : qbpp::HuboModel(hubo_model), grb_env(true) {
  GRB_SAFE_CALL(grb_env.set(GRB_IntParam_OutputFlag, 0));
  if (verbose) {
    GRB_SAFE_CALL(grb_env.set("OutputFlag", "1"));
  }
  GRB_SAFE_CALL(grb_env.start());
  if (hubo_model.max_power() > 2) {
    std::cerr << "HuboModel has a term with maximum power of "
              << hubo_model.max_power()
              << ". Gurobi supports only quadratic models." << std::endl;
  }
  grb_model_ptr = std::make_unique<GRBModel>(grb_env);
  GRB_SAFE_CALL(grb_model_ptr->set(GRB_IntAttr_ModelSense, GRB_MINIMIZE));
  grb_x = grb_model_ptr->addVars(static_cast<int>(var_count()), GRB_BINARY);
  GRBQuadExpr obj;
  obj += static_cast<double>(hubo_model.constant());
  for (qbpp::vindex_t i = 0; i < var_count(); ++i) {
    if (hubo_model.term1()[i] != 0)
      obj += static_cast<double>(hubo_model.term1()[i]) * grb_x[i];
  }
  for (qbpp::vindex_t i = 0; i < var_count(); ++i) {
    qbpp::vindex_t degree =
        static_cast<qbpp::vindex_t>(hubo_model.term2(i).size());
    for (qbpp::vindex_t j = 0; j < degree; ++j) {
      auto k = hubo_model.term2(i).indices(j)[0];
      auto coeff = hubo_model.term2(i).coeff(j);
      if (i < k) obj += coeff * grb_x[i] * grb_x[k];
    }
  }
  GRB_SAFE_CALL(grb_model_ptr->setObjective(obj));
}
inline void HuboModel::set(Callback &cb) { grb_model_ptr->setCallback(&cb); }
inline Sol HuboModel::optimize() {
  Sol sol(*this);
  GRB_SAFE_CALL(grb_model_ptr->optimize());
  for (qbpp::vindex_t i = 0; i < var_count(); ++i)
    sol.set(i, int(grb_x[i].get(GRB_DoubleAttr_X)));
  sol.energy(static_cast<qbpp::energy_t>(
      std::round(grb_model_ptr->get(GRB_DoubleAttr_ObjVal))));
  sol.bound(static_cast<qbpp::energy_t>(
      std::round(grb_model_ptr->get(GRB_DoubleAttr_ObjBound))));
  return sol;
}
inline Sol Callback::sol() {
  Sol sol(hubo_model_);
  std::lock_guard<std::mutex> lock(mtx_);
  sol.bound(static_cast<qbpp::energy_t>(
      std::round(getDoubleInfoPublic(GRB_CB_MIPSOL_OBJBND))));
  for (qbpp::vindex_t i = 0; i < hubo_model_.var_count(); ++i) {
    sol.set(i, int(getSolutionPublic(hubo_model_.grb_var(i))));
  }
  sol.energy(eval(hubo_model_, sol));
  return sol;
}
}  
