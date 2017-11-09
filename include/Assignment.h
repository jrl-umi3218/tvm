#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include <Eigen/Core>

#include <tvm/api.h>
#include "AssignmentTarget.h"
#include "tvm/utils/CompiledAssignmentWrapper.h"
#include "ConstraintEnums.h"
#include "defs.h"
#include "SolvingRequirements.h"
#include "Variable.h" //for Range

namespace tvm
{
  /** A class whose role is to assign efficiently the matrix and vector(s) of a
    * LinearConstraint to a part of matrix and vector(s) specified by a
    * ResolutionScheme and a mapping of variables. This is done while taking
    * into account the possible convention differences between the constraint
    * and the scheme, as well as the requirements on the constraint.
    */
  class TVM_DLLAPI Assignment
  {
  public:
    using RHSFunction = const Eigen::VectorXd& (LinearConstraint::*)() const;
    using MatrixFunction = MatrixRef (AssignmentTarget::*)(int, int) const;
    using VectorFunction = VectorRef (AssignmentTarget::*)() const;

    /** Assignment constructor
      * \param source The linear constraints whose matrix and vector(s) will be
      * assigned.
      * \param req Solving requirements attached to this constraint.
      * \param target The target of the assignment.
      * \param variables The vector of variables corresponding to the target.
      * It must be such that its total dimension is equal to the column size of
      * the target matrix.
      * \param scalarizationWeight An additional scalar weight to apply on the
      * constraint, used by the solver to emulate priority.
      */
    Assignment(LinearConstraintPtr source, std::shared_ptr<SolvingRequirements> req,
               const AssignmentTarget& target, const VariableVector& variables, double scalarizationWeight = 1);

    /** To be called when the source has been resized*/
    void onUpdatedSource();
    /** To be called when the target has been resized and/or range has changed*/
    void onUpdatedTarget();
    /** To be called when the variables change.*/
    void onUpdatedMapping(const VariableVector& variables);
    /** Change the weight of constraint. TODO: how to specify the constraint?*/
    void weight(double alpha);
    void weight(const Eigen::VectorXd& w);

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


    bool checkTarget();

    /** Where the magic happens*/
    void build(const VariableVector& variables);
    void processRequirements();
    void addMatrixAssignment(Variable* x, MatrixFunction M, const Range& range, bool flip);
    void addVectorAssignment(RHSFunction f, VectorFunction v, bool flip);
    void addConstantAssignment(double d, VectorFunction v);

    template<typename T, typename U>
    utils::CompiledAssignmentWrapper<T> createAssignment(const U& from, const Eigen::Ref<T>& to, bool flip);

    LinearConstraintPtr source_;
    AssignmentTarget target_;
    double scalarizationWeight_;
    std::shared_ptr<SolvingRequirements> requirements_;
    std::vector<MatrixAssignment> matrixAssignments_;
    std::vector<VectorAssignment> vectorAssignments_;

    /** Processed requirements*/
    double scalarWeight_;
    Eigen::VectorXd anisotropicWeight_;
    Eigen::VectorXd minusAnisotropicWeight_;
    Eigen::MatrixXd mult_; //unused for now, will serve when substituting variables
  };

  template<typename T, typename U>
  inline utils::CompiledAssignmentWrapper<T> Assignment::createAssignment(const U& from, const Eigen::Ref<T>& to, bool flip)
  {
    using namespace utils;
    typedef typename utils::CompiledAssignmentWrapper<typename std::conditional<std::is_arithmetic<U>::value, Eigen::VectorXd, T>::type> Wrapper;

    if (requirements_->anisotropicWeight().isDefault())
    {
      if (scalarWeight_ == 1)
      {
        if (flip)
          return Wrapper::template make<COPY, MINUS, IDENTITY, PRE>(from, to);
        else
          return Wrapper::template make<COPY, NONE, IDENTITY, PRE>(from, to);
      }
      else
      {
        if (flip)
          return Wrapper::template make<COPY, SCALAR, IDENTITY, PRE>(from, to, -scalarWeight_);
        else
          return Wrapper::template make<COPY, SCALAR, IDENTITY, PRE>(from, to, scalarWeight_);
      }
    }
    else
    {
      if (flip)
        return Wrapper::template make<COPY, NONE, DIAGONAL, PRE>(from, to, 1, &minusAnisotropicWeight_);
      else
        return Wrapper::template make<COPY, NONE, DIAGONAL, PRE>(from, to, 1, &anisotropicWeight_);
    }
  }
}
