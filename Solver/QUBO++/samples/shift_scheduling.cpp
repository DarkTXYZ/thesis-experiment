/// @file shift_scheduling.cpp
/// @brief This program solves the simpleshift scheduling problem.
/// @author Koji Nakano
/// @version 2025.06.14
/// @details This program solves the simple shift scheduling problem using
/// - Problem Setting
/// - There are `employees` employees and `days` days.
/// - Each day, at least `required` or `required+1`employees must be assigned to
/// work.
/// - Each employee can work for a maximum of `consecutive` consecutive days.
/// - Every employee work for two ore more consecutive days.
/// - The objective is to minimize the total of the square of the number of
/// working days of each employee to minimze the total number of working days
/// and the variance of the number of working days among employees.

#include <boost/program_options.hpp>
#include "qbpp.hpp"
#include "qbpp_easy_solver.hpp"
namespace po = boost::program_options;
po::variables_map vm;
int main(int argc, char **argv) {
  po::options_description desc("Shift Scheduling");
desc.add_options()
    ("help,h", "Show this help message.")
    ("employees,e", po::value<uint32_t>()->default_value(5), "The number of employees.")
    ("days,d", po::value<uint32_t>()->default_value(30), "The number of days.")
    ("required,r", po::value<uint32_t>()->default_value(3), "The required number of employees per day.")
    ("consecutive,c", po::value<uint32_t>()->default_value(6), "The maximum number of consecutive working days.")
    ("time,t", po::value<double>()->default_value(1.0), "Time limit in second.");
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const std::exception &e) {
    std::cerr << "Invalid arguments: " << e.what()
              << "\nUse -h or --help to see usage.\n";
    return 1;
  }
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  auto employees = vm["employees"].as<uint32_t>();
  auto days = vm["days"].as<uint32_t>();
  auto required = vm["required"].as<uint32_t>();
  auto consecutive = vm["consecutive"].as<uint32_t>();
  auto time_limit = vm["time"].as<double>();
  auto x = qbpp::var("x", employees, days + 2);
  qbpp::Expr constraint;
  qbpp::Expr objective;
  qbpp::Expr c_required;
  qbpp::Expr c_employees;
  qbpp::Expr c_consecutive;
  qbpp::Expr c_no2days;
  qbpp::Expr c_no1day;
  qbpp::MapList fix;
  for (size_t i = 0; i < employees; ++i) {
    fix.push_back({x[i][0], 0});
    fix.push_back({x[i][days + 1], 0});
  }
  auto assigned = qbpp::expr(days + 1);
  for (size_t j = 1; j < days + 1; ++j) {
    for (size_t i = 0; i < employees; ++i) {
      assigned[j] += x[i][j];
    }
    c_required += required <= assigned[j] <= required + 1;
  }
  for (size_t i = 0; i < employees; ++i) {
    for (size_t j = 0; j < days + 2 - consecutive; ++j) {
      qbpp::Expr temp = 1;
      for (size_t k = 0; k < consecutive; ++k) {
        temp *= x[i][j + k];
      }
      c_consecutive += temp;
    }
  }
  c_consecutive.replace(fix);
  for (size_t i = 0; i < employees; ++i) {
    for (size_t j = 0; j < days; ++j) {
      c_no1day += (1 - x[i][j]) * x[i][j + 1] * (1 - x[i][j + 2]);
    }
    for (size_t j = 0; j < days - 1; ++j) {
      c_no2days +=
          (1 - x[i][j]) * x[i][j + 1] * x[i][j + 2] * (1 - x[i][j + 3]);
    }
  }
  c_no1day.replace(fix);
  c_no2days.replace(fix);
  constraint = c_required + c_consecutive + c_no1day + c_no2days;
  auto o_working_days = qbpp::expr(employees);
  for (size_t i = 0; i < employees; ++i) {
    for (size_t j = 1; j < days + 1; ++j) {
      o_working_days[i] += x[i][j];
    }
    objective += qbpp::sqr(o_working_days[i]);
  }
  qbpp::Expr f = constraint * 10000 + objective;
  f.simplify_as_binary();
  std::cout << "f = " << f << std::endl;
  auto solver = qbpp::easy_solver::EasySolver(f);
  solver.time_limit(time_limit);
  auto sol = solver.search();
  for (size_t i = 0; i < employees; ++i) {
    std::cout << "Employee " << i << ": ";
    for (size_t j = 1; j < days + 1; ++j) {
      std::cout << sol(x[i][j]);
    }
    std::cout << " " << sol(o_working_days[i]) << std::endl;
  }
  std::cout << "Assigned  : ";
  for (size_t j = 1; j < days + 1; ++j) {
    std::cout << sol(assigned[j]);
  }
  std::cout << std::endl;
  std::cout << "Objective: " << sol(objective) << std::endl;
  std::cout << "Constraint: " << sol(constraint) << std::endl;
}