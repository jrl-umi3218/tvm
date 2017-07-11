#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include <Eigen/Core>

#include <tvm/api.h>
#include "AssignmentTarget.h"
#include "CompiledAssignmentWrapper.h"
#include "ConstraintEnums.h"
#include "SolvingRequirements.h"
#include "Variable.h" //for Range

namespace tvm
{
  class LinearConstraint;
  class VariableVector;

  /** A class whose role is to assign efficiently the matrix and vector(s) of a
    * LinearConstraint to a part of matrix and vector(s) specified by the a
    * ResolutionScheme and the mapping of variables. This is done while taking
    * into account the possible convention differences between the constraint
    * and the scheme, as well as the requirements on the constraint.
    */
  class TVM_DLLAPI Assignment
  {
  public:
    typedef const Eigen::VectorXd& (LinearConstraint::*RHSFunction)() const;
    typedef MatrixRef(AssignmentTarget::*MatrixFunction)(int, int) const;
    typedef VectorRef (AssignmentTarget::*VectorFunction)() const;

    Assignment(std::shared_ptr<LinearConstraint> source, const SolvingRequirements& req,
               const AssignmentTarget& target, const VariableVector& variables);
  
    /** To be called when the source has been resized*/
    void refreshSource();
    /** To be called when the target has been resized and/or range has changed*/
    void refreshTarget();
    /** To be called when the variables change.*/
    void refreshMapping(const VariableVector& variables);
    /** */
    void changeWeight(double alpha);
    void changeWeight(const Eigen::VectorXd& w);

    /** Perform the assignment.*/
    void run();

  private:
    struct MatrixAssignment
    {
      utils::CompiledAssignmentWrapper<Eigen::MatrixXd> assignment;
      Variable* x;
      Range colRange;
      MatrixFunction getTargetMatrix;
    };

    struct VectorAssignment
    {
      utils::CompiledAssignmentWrapper<Eigen::VectorXd> assignment;
      bool useSource;
      RHSFunction getSourceVector;
      VectorFunction getTargetVector;
    };


    static bool checkTarget();

    /** Where the magic happens*/
    void build(const VariableVector& variables);
    void processRequirements();
    void addMatrixAssignment(Variable* x, MatrixFunction M, const Range& range, bool flip);
    void addVectorAssignment(RHSFunction f, VectorFunction v, bool flip);
    void addConstantAssignment(double d, VectorFunction v);

    template<typename T, typename U>
    utils::CompiledAssignmentWrapper<T> createAssignment(const U& from, const Eigen::Ref<T>& to, bool flip);

    std::shared_ptr<LinearConstraint> source_;
    AssignmentTarget target_;
    SolvingRequirements requirements_;
    std::vector<MatrixAssignment> matrixAssignments_;
    std::vector<VectorAssignment> vectorAssignments_;

    /** Processed requirements*/
    double alpha_;
    Eigen::VectorXd weight_;
    Eigen::VectorXd minusWeight_;
    Eigen::MatrixXd mult_; //unused for now, will serve when substituting variables
  };

  template<typename T, typename U>
  inline utils::CompiledAssignmentWrapper<T> Assignment::createAssignment(const U& from, const Eigen::Ref<T>& to, bool flip)
  {
    using namespace utils;
    typedef typename utils::CompiledAssignmentWrapper<typename std::conditional<std::is_arithmetic<U>::value, Eigen::VectorXd, T>::type> Wrapper;

    if (requirements_.anisotropicWeight().isDefault())
    {
      if (requirements_.weight().isDefault())
      {
        if (flip)
          return Wrapper::template make<REPLACE, MINUS, IDENTITY, PRE>(from, to);
        else
          return Wrapper::template make<REPLACE, NONE, IDENTITY, PRE>(from, to);
      }
      else
      {
        if (flip)
          return Wrapper::template make<REPLACE, SCALAR, IDENTITY, PRE>(from, to, -alpha_);
        else
          return Wrapper::template make<REPLACE, SCALAR, IDENTITY, PRE>(from, to, alpha_);
      }
    }
    else
    {
      if (flip)
        return Wrapper::template make<REPLACE, NONE, DIAGONAL, PRE>(from, to, 1, &minusWeight_);
      else
        return Wrapper::template make<REPLACE, NONE, DIAGONAL, PRE>(from, to, 1, &weight_);
    }
  }
}
