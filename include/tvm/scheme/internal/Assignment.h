#pragma once

/* Copyright 2017 CNRS-UM LIRMM, CNRS-AIST JRL
 *
 * This file is part of TVM.
 *
 * TVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with TVM.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/Variable.h> // Range
#include <tvm/constraint/abstract/LinearConstraint.h>
#include <tvm/hint/internal/Substitutions.h>
#include <tvm/requirements/SolvingRequirements.h>
#include <tvm/scheme/internal/AssignmentTarget.h>
#include <tvm/scheme/internal/CompiledAssignmentWrapper.h>

#include <Eigen/Core>

#include <memory>
#include <type_traits>
#include <vector>

namespace tvm
{

namespace scheme
{

namespace internal
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
    using RHSFunction = const Eigen::VectorXd& (constraint::abstract::LinearConstraint::*)() const;
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
    Assignment(LinearConstraintPtr source, 
               std::shared_ptr<requirements::SolvingRequirements> req,
               const AssignmentTarget& target, 
               const VariableVector& variables, 
               const hint::internal::Substitutions& substitutions = {},
               double scalarizationWeight = 1);

    /** Version for bounds
      * \param first wether this is the first assignement of bounds for this
      * variable (first assignment just copy vectors while the following ones
      * need to perform min/max operations).
      */
    Assignment(LinearConstraintPtr source, const AssignmentTarget& target, const VariablePtr& variables, bool first);

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

    static double big_;

  private:
    struct MatrixAssignment
    {
      CompiledAssignmentWrapper<Eigen::MatrixXd> assignment;
      Variable* x;
      Range colRange;
      MatrixFunction getTargetMatrix;
    };

    struct VectorAssignment
    {
      CompiledAssignmentWrapper<Eigen::VectorXd> assignment;
      bool useSource;
      RHSFunction getSourceVector;
      VectorFunction getTargetVector;
    };

    /** Check that the convention and size of the target are compatible with the
      * convention and size of the source.
      */
    void checkTarget(bool bound = false);

    void checkVariables(bool bound);

    /** Where the magic happens*/
    void build(const VariableVector& variables);
    void build(const VariablePtr& variable, bool first);
    void processRequirements();
    void addMatrixAssignment(Variable& x, MatrixFunction M, const Range& range, bool flip);
    void addMatrixSubstitutionAssignments(const VariableVector& variables, Variable& x, MatrixFunction M, 
                                          const function::BasicLinearFunction& sub, bool flip);
    template<AssignType A = AssignType::COPY>
    void addVectorAssignment(RHSFunction f, VectorFunction v, bool flip);
    template<AssignType A = AssignType::COPY>
    void addConstantAssignment(double d, VectorFunction v);
    void addZeroAssignment(Variable& x, MatrixFunction M, const Range& range);
    void addAssignments(const VariableVector& variables, MatrixFunction M,
                        RHSFunction f, VectorFunction v, bool flip);
    void addAssignments(const VariableVector& variables, MatrixFunction M,
                        RHSFunction f1, VectorFunction v1, 
                        RHSFunction f2, VectorFunction v2);

    template<typename T, AssignType A, typename U>
    CompiledAssignmentWrapper<T> createAssignment(const U& from, const Eigen::Ref<T>& to, bool flip = false);

    template<typename T, AssignType A, typename U, typename V>
    CompiledAssignmentWrapper<T> createSubstitutionAssignment(const U& from, const Eigen::Ref<T>& to, const V& Mult, bool flip = false);

    LinearConstraintPtr source_;
    AssignmentTarget target_;
    double scalarizationWeight_;
    std::shared_ptr<requirements::SolvingRequirements> requirements_;
    /** All the assignements that are setting the initial values of the targeted blocks*/
    std::vector<MatrixAssignment> matrixAssignments_;
    /** All assignments due to substitution. We separe them from matrixAssignments_
      * because these assignements add to existing values, and we need to be sure
      * that the assignements in matrixAssignments_ have been carried out before.
      */
    std::vector<MatrixAssignment> matrixSubstitutionAssignments_;
    /** All the initial rhs assignments*/
    std::vector<VectorAssignment> vectorAssignments_;
    /** The additional rhs assignments due to substitutions. As for matrix
      * assignments, they need to be carried out after those of vectorAssignments_.
      */
    std::vector<VectorAssignment> vectorSubstitutionAssignments_;

    /** Processed requirements*/
    double scalarWeight_;
    Eigen::VectorXd anisotropicWeight_;
    Eigen::VectorXd minusAnisotropicWeight_;

    /** Data for substitutions */
    VariableVector substitutedVariables_;
    std::vector<std::shared_ptr<function::BasicLinearFunction>> variableSubstitutions_;
  };

  template<typename T, AssignType A, typename U>
  inline CompiledAssignmentWrapper<T> Assignment::createAssignment(const U& from, const Eigen::Ref<T>& to, bool flip)
  {
    using Wrapper = CompiledAssignmentWrapper<typename std::conditional<std::is_arithmetic<U>::value, Eigen::VectorXd, T>::type>;
    const Source F = std::is_arithmetic<U>::value ? CONSTANT : EXTERNAL;

    if (requirements_->anisotropicWeight().isDefault())
    {
      if (scalarWeight_ == 1)
      {
        if (flip)
          return Wrapper::template make<A, MINUS, IDENTITY, F>(to, from);
        else
          return Wrapper::template make<A, NONE, IDENTITY, F>(to, from);
      }
      else
      {
        if (flip)
          return Wrapper::template make<A, SCALAR, IDENTITY, F>(to, from, -scalarWeight_);
        else
          return Wrapper::template make<A, SCALAR, IDENTITY, F>(to, from, scalarWeight_);
      }
    }
    else
    {
      if (flip)
        return Wrapper::template make<A, DIAGONAL, IDENTITY, F>(to, from, minusAnisotropicWeight_);
      else
        return Wrapper::template make<A, DIAGONAL, IDENTITY, F>(to, from, anisotropicWeight_);
    }
  }

  template<typename T, AssignType A, typename U, typename V>
  inline CompiledAssignmentWrapper<T> Assignment::createSubstitutionAssignment(const U& from, const Eigen::Ref<T>& to, const V& Mult, bool flip)
  {
    using Wrapper = CompiledAssignmentWrapper<typename std::conditional<std::is_arithmetic<U>::value, Eigen::VectorXd, T>::type>;
    const Source F = std::is_arithmetic<U>::value ? CONSTANT : EXTERNAL;
    const MatrixMult M = GENERAL; //FIXME have a swith on Mult for detecting CUSTOM case

    if (requirements_->anisotropicWeight().isDefault())
    {
      if (scalarWeight_ == 1)
      {
        if (flip)
          return Wrapper::template make<A, MINUS, M, F>(to, from, Mult);
        else
          return Wrapper::template make<A, NONE, M, F>(to, from, Mult);
      }
      else
      {
        if (flip)
          return Wrapper::template make<A, SCALAR, M, F>(to, from, -scalarWeight_, Mult);
        else
          return Wrapper::template make<A, SCALAR, M, F>(to, from, scalarWeight_, Mult);
      }
    }
    else
    {
      if (flip)
        return Wrapper::template make<A, DIAGONAL, M, F>(to, from, minusAnisotropicWeight_, Mult);
      else
        return Wrapper::template make<A, DIAGONAL, M, F>(to, from, anisotropicWeight_, Mult);
    }
  }

  template<AssignType A>
  inline void Assignment::addVectorAssignment(RHSFunction f, VectorFunction v, bool flip)
  {
    bool useSource = source_->rhs() != constraint::RHS::ZERO;
    if (useSource)
    {
      // So far, the sign flip has been deduced only from the ConstraintType of the source
      // and the target. Now we need to take into account the constraint::RHS as well.
      if (source_->rhs() == constraint::RHS::OPPOSITE)
        flip = !flip;
      if (target_.constraintRhs() == constraint::RHS::OPPOSITE)
        flip = !flip;

      const VectorRef& to = (target_.*v)();
      const VectorConstRef& from = (source_.get()->*f)();
      auto w = createAssignment<Eigen::VectorXd, A>(from, to, flip);
      vectorAssignments_.push_back({ w, true, f, v });
    }
    else
    {
      if (target_.constraintRhs() != constraint::RHS::ZERO)
      {
        const VectorRef& to = (target_.*v)();
        auto w = CompiledAssignmentWrapper<Eigen::VectorXd>::make<A, NONE, IDENTITY, ZERO>(to);
        vectorAssignments_.push_back({ w, false, nullptr, v });
      }
    }
  }

  template<AssignType A>
  inline void Assignment::addConstantAssignment(double d, VectorFunction v)
  {
    const VectorRef& to = (target_.*v)();
    auto w = createAssignment<Eigen::VectorXd, A>(d, to, false);
    vectorAssignments_.push_back({ w, false, nullptr, v });
  }

}  // namespace internal

}  // namespace scheme

}  // namespace tvm
