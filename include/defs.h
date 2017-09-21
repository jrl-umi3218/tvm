#pragma once

#include <memory>

#include <Eigen/Core>

namespace tvm
{
  //forward declarations
  class Constraint;
  class Function;
  class LinearConstraint;
  class Range;
  class SolvingRequirements;
  class TaskDynamics;
  class Variable;
  class VariableVector;

  //definitions
  typedef Eigen::Ref<const Eigen::MatrixXd> MatrixConstRef;
  typedef Eigen::Ref<Eigen::MatrixXd>       MatrixRef;
  typedef Eigen::Ref<const Eigen::VectorXd> VectorConstRef;
  typedef Eigen::Ref<Eigen::VectorXd>       VectorRef;

  typedef std::shared_ptr<Eigen::MatrixXd>  MatrixPtr;
  typedef std::shared_ptr<Eigen::VectorXd>  VectorPtr;

  typedef std::shared_ptr<Constraint>           ConstraintPtr;
  typedef std::shared_ptr<Function>             FunctionPtr;
  typedef std::shared_ptr<LinearConstraint>     LinearConstraintPtr;
  typedef std::shared_ptr<Range>                RangePtr;
  typedef std::shared_ptr<SolvingRequirements>  SolvingRequirementsPtr;
  typedef std::shared_ptr<Variable>             VariablePtr;
}
