/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <limits>
#include <memory>

#include <Eigen/Core>

#include <tvm/internal/math.h>

namespace tvm
{
  //forward declarations
  namespace constraint
  {
    namespace abstract
    {
      class Constraint;
      class LinearConstraint;
    }
  }
  namespace function
  {
    namespace abstract
    {
      class Function;
      class LinearFunction;
    }
  }
  namespace requirements
  {
    class SolvingRequirements;
  }
  namespace task_dynamics
  {
    namespace abstract
    {
      class TaskDynamicsImpl;
    }
  }
  class Clock;
  class Range;
  class Robot;
  class Variable;
  class VariableVector;

  //definitions
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
  using SolvingRequirementsPtr = std::shared_ptr<requirements::SolvingRequirements>;
  using TaskDynamicsPtr = std::shared_ptr<task_dynamics::abstract::TaskDynamicsImpl>;
  using VariablePtr = std::shared_ptr<Variable>;

  //constants
  namespace constant
  {
    namespace internal
    {
      /** \internal We compute the square root of std::numeric_limits<double>::max()
        * We start with an approximation 2^{max_exponent/2}
        */
      static constexpr double sqrtGuess = tvm::internal::pow(2., std::numeric_limits<double>::max_exponent / 2);
      static constexpr double sqrtOfMax = tvm::internal::sqrtNewtonRaphson(std::numeric_limits<double>::max(), sqrtGuess, 0);
    }

    /** We take as a default big number sqrt(std::numeric_limits<double>::max())/2 */
    static constexpr double big_number = internal::sqrtOfMax/2;
    //assert that the big_number value is correct (no std::abs here because it is not constexpr)
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

    /** Default gravity vector 
      *
      * \internal Should we really have this here and why is there no - sign in front of 9.81?
      */
    static const Eigen::Vector3d gravity {0, 0, 9.81}; 
  } // namespace constant

}  // namespace tvm
