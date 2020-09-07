/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <limits>
#include <memory>

#include <Eigen/Core>

namespace tvm
{
// forward declarations
namespace constraint
{
namespace abstract
{
class Constraint;
class LinearConstraint;
} // namespace abstract
} // namespace constraint
namespace function
{
namespace abstract
{
class Function;
class LinearFunction;
} // namespace abstract
} // namespace function
namespace requirements
{
class SolvingRequirements;
class SolvingRequirementsWithCallbacks;
} // namespace requirements
namespace task_dynamics
{
namespace abstract
{
class TaskDynamicsImpl;
}
} // namespace task_dynamics
class Clock;
class Range;
class Robot;
class Variable;
class VariableVector;

// definitions
using MatrixConstRef = Eigen::Ref<const Eigen::MatrixXd>;
using MatrixRef = Eigen::Ref<Eigen::MatrixXd>;
using VectorConstRef = Eigen::Ref<const Eigen::VectorXd>;
using VectorRef = Eigen::Ref<Eigen::VectorXd>;

using MatrixPtr = std::shared_ptr<Eigen::MatrixXd>;
using VectorPtr = std::shared_ptr<Eigen::VectorXd>;

using ConstraintPtr = std::shared_ptr<constraint::abstract::Constraint>;
using FunctionPtr = std::shared_ptr<function::abstract::Function>;
using LinearFunctionPtr = std::shared_ptr<function::abstract::LinearFunction>;
using LinearConstraintPtr = std::shared_ptr<constraint::abstract::LinearConstraint>;
using ClockPtr = std::shared_ptr<Clock>;
using RangePtr = std::shared_ptr<Range>;
using RobotPtr = std::shared_ptr<Robot>;
using SolvingRequirementsPtr = std::shared_ptr<requirements::SolvingRequirementsWithCallbacks>;
using TaskDynamicsPtr = std::shared_ptr<task_dynamics::abstract::TaskDynamicsImpl>;
using VariablePtr = std::shared_ptr<Variable>;

// constants
namespace constant
{
namespace internal
{
/** Constexpr integer power base^exp
 * Adapted from https://stackoverflow.com/a/17728525
 */
template<typename T>
constexpr T pow(T base, unsigned int exp, T result = 1)
{
  return exp <= 1 ? (exp == 0 ? 1 : result * base) : pow(base * base, exp / 2, (exp % 2) ? result * base : result);
}

/* Constexpr version of the square root of x
 * curr is the initial guess for the square root
 * Adapted from https://gist.github.com/alexshtf/eb5128b3e3e143187794
 */
constexpr double sqrtNewtonRaphson(double x, double curr, double prev = 0)
{
  return curr == prev ? curr : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
}

/** \internal We compute the square root of std::numeric_limits<double>::max()
 * We start with an approximation 2^{max_exponent/2}
 */
static constexpr double sqrtGuess = pow(2., std::numeric_limits<double>::max_exponent / 2);
static constexpr double sqrtOfMax = sqrtNewtonRaphson(std::numeric_limits<double>::max(), sqrtGuess, 0);
} // namespace internal

/** We take as a default big number sqrt(std::numeric_limits<double>::max())/2 */
static constexpr double big_number = internal::sqrtOfMax / 2;
// assert that the big_number value is correct (no std::abs here because it is not constexpr)
static_assert(-(big_number * big_number - std::numeric_limits<double>::max() / 4)
                      < (2 * std::numeric_limits<double>::epsilon()) * std::numeric_limits<double>::max()
                  && (big_number * big_number - std::numeric_limits<double>::max() / 4)
                         < (2 * std::numeric_limits<double>::epsilon()) * std::numeric_limits<double>::max(),
              "big_number was not computed at compile time or its value was not correct");

/** Pi (from boost/math/constant)*/
constexpr double pi = 3.141592653589793238462643383279502884e+00;

/** Constant for specifying that the matrix in front of the variable is full
 * rank.
 */
constexpr int fullRank = -1;

/** Default gravity vector */
static const Eigen::Vector3d gravity{0, 0, 9.81};
} // namespace constant

} // namespace tvm
