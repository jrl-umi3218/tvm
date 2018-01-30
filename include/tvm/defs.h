#pragma once

#include <limits>
#include <memory>

#include <Eigen/Core>

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
  class Range;
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
  using LinearConstraintPtr = std::shared_ptr<constraint::abstract::LinearConstraint>;
  using RangePtr = std::shared_ptr<Range>;
  using SolvingRequirementsPtr = std::shared_ptr<requirements::SolvingRequirements>;
  using TaskDynamicsPtr = std::shared_ptr<task_dynamics::abstract::TaskDynamicsImpl>;
  using VariablePtr = std::shared_ptr<Variable>;

  //constants
  namespace constant
  {
    namespace internal
    {
      /** Constexpr integer power base^exp
        * Taken from https://stackoverflow.com/a/17728525
        */
      template <typename T>
      constexpr T pow(T base, int exp, T result = 1) 
      {
        return exp < 1 ? result : pow(base*base, exp / 2, (exp % 2) ? result*base : result);
      }

      /* Constexpr version of the square root of x
       * curr is the initial guess for the square root
       * Adapted from https://gist.github.com/alexshtf/eb5128b3e3e143187794
       */
      double constexpr sqrtNewtonRaphson(double x, double curr, double prev=0)
      {
        return curr == prev ? curr : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
      }
      
      /** \internal We compute the square root of std::numeric_limits<double>::max()
        * We start with an approximation 2^{max_exponent/2}
        */
      static constexpr double sqrtGuess = pow(2., std::numeric_limits<double>::max_exponent / 2);
      static constexpr double sqrtOfMax = sqrtNewtonRaphson(std::numeric_limits<double>::max(), sqrtGuess, 0);
    }

    /** We take as a default big number sqrt(std::numeric_limits<double>::max())/2 */
    static constexpr double big_number = internal::sqrtOfMax/2;
  };

}  // namespace tvm
