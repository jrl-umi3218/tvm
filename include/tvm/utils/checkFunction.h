/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>

#include <optional>
#include <vector>

namespace tvm
{

namespace utils
{

/** A small structure to specify options for the checks in \ref checkGroup.
 *
 *  - \a step is the increment that will be taken for finite difference schemes
 *  - \a prec is the precision with which the equality of two vectors is
 *  tested. It corresponds to the \a prec parameter of Eigen's \a isApprox
 *  method.
 *  - if \a verbose is true, the functions will display some indications when a
 *  mismatch is detected.
 *  - if \a samples is > 0 then the check will generate \a samples random
 *  configurations
 *  - if \a configs are provided then these specific configurations are also
 *  tested, see \ref CheckOptions::CheckConfiguration
 */

class CheckOptions
{
public:
  /** A specific configuration to check
   * - \a value will be stored in the function's variable value
   * - \a velocity will be stored in the function's variable's derivative value
   * - \a acceleration will be stored in the function's variable's derivative's
   *   derivative value
   * - \a samples if any of value/velocity/acceleration is null then \a samples
   *   random configuration are generated for those (if it makes sense for this test)
   */
  struct CheckConfiguration
  {
    std::optional<Eigen::VectorXd> value = std::nullopt;
    std::optional<Eigen::VectorXd> velocity = std::nullopt;
    std::optional<Eigen::VectorXd> acceleration = std::nullopt;
    size_t samples = 1;
  };

  CheckOptions() : step(1e-7), prec(1e-6), verbose(true) {}
  CheckOptions(double s, double p, bool v) : step(s), prec(p), verbose(v) {}
  double step;
  double prec;
  bool verbose;
  size_t samples = 1;
  std::vector<CheckConfiguration> configs = {};
};

/** \defgroup checkGroup */
/**@{*/
/** Check the jacobian matrices of function \a f by forward finite differences.*/
bool TVM_DLLAPI checkJacobian(FunctionPtr f, CheckOptions opt = CheckOptions());

/** Check the velocity of the function \a f by comparing it to J*\dot{x}.
 * Assume that the jacobian matrices are correct.
 */
bool TVM_DLLAPI checkVelocity(FunctionPtr f, CheckOptions opt = CheckOptions());

/** Check the normal acceleration of the function \a f.
 * Noting v=f(x), this is done by comparing it to ddot{v}-J\ddot{x}, where
 * \ddot{x} is taken constant over the interval opt.step, and ddot{v} is obtained
 * by finite differences.
 * Assume that the jacobian matrices and the velocity are correct.
 */
bool TVM_DLLAPI checkNormalAcceleration(FunctionPtr f, CheckOptions opt = CheckOptions());

/** Check the jacobian matrices, velocity and normal acceleration of the
 * function \a f
 */
bool TVM_DLLAPI checkFunction(FunctionPtr f, CheckOptions opt = CheckOptions());
/**@}*/
} // namespace utils

} // namespace tvm
