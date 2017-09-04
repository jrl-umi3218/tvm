#pragma once

#include <memory>

#include <Eigen/Core>

namespace tvm
{
  //forward declarations
  class Variable;
  class VariableVector;
  class Range;
  class Function;
  class TaskDynamics;
  class LinearConstraint;

  //definitions
  typedef Eigen::Ref<Eigen::MatrixXd>       MatrixRef;
  typedef Eigen::Ref<const Eigen::MatrixXd> MatrixConstRef;
  typedef Eigen::Ref<Eigen::VectorXd>       VectorRef;
  typedef Eigen::Ref<const Eigen::VectorXd> VectorConstRef;

  typedef std::shared_ptr<Eigen::MatrixXd>  MatrixPtr;
  typedef std::shared_ptr<Eigen::VectorXd>  VectorPtr;

  typedef std::shared_ptr<Variable>         VariablePtr;
  typedef std::shared_ptr<Range>            RangePtr;
  typedef std::shared_ptr<Function>         FunctionPtr;
  typedef std::shared_ptr<LinearConstraint> LinearConstraintPtr;
}
