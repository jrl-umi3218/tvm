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
    /** Pointer type to a method of LinearConstraint returning a vector.
      * It is used to make a selection between e(), l() and u().
      */
    using RHSFunction = const Eigen::VectorXd& (constraint::abstract::LinearConstraint::*)() const;
   
    /** Pointer type to a method of AssignmentTarget returning a matrix block.
      * It is used to make a selection between A(), AFirstHalf() and ASecondHalf().
      */
    using MatrixFunction = MatrixRef(AssignmentTarget::*)(int, int) const;

    /** Pointer type to a method of AssignementTarget returning a vector segment.
      * It is used to make a selection between b(), bFirstHalf(), bSecondHalf(),
      * l() and u().
      */
    using VectorFunction = VectorRef(AssignmentTarget::*)() const;

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
               const hint::internal::Substitutions* const subs = nullptr,
               double scalarizationWeight = 1);

    /** Version for bounds
      * \param first wether this is the first assignement of bounds for this
      * variable (first assignment just copy vectors while the following ones
      * need to perform min/max operations).
      */
    Assignment(LinearConstraintPtr source, const AssignmentTarget& target, const VariablePtr& variables, bool first);

    Assignment(const Assignment&) = delete;
    Assignment(Assignment&&) = default;

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
    /** A structure grouping a matrix assignment and some of the elements that
      * defined it.
      */
    struct MatrixAssignment
    {
      CompiledAssignmentWrapper<Eigen::MatrixXd> assignment;
      Variable* x;
      Range colRange;
      MatrixFunction getTargetMatrix;
    };

    /** A structure grouping a vector assignment and some of the elements that
    * defined it.
    */
    struct VectorAssignment
    {
      CompiledAssignmentWrapper<Eigen::VectorXd> assignment;
      bool useSource;
      RHSFunction getSourceVector;
      VectorFunction getTargetVector;
    };

    struct VectorSubstitutionAssignement
    {
      CompiledAssignmentWrapper<Eigen::VectorXd> assignment;
      VectorFunction getTargetVector;
    };

    /** Check that the convention and size of the target are compatible with the
      * convention and size of the source.
      */
    void checkTarget();

    void checkBounds();

    /** Generates the assignments for the general case.
      * \param variables the set of variables for the problem.
      */
    void build(const VariableVector& variables);
    
    /** Generates the assignments for the bound case.
      * \param variables the set of variables for the problem.
      * \param first true if this is the first assignment for the bounds (first
      * assignment makes copy, the following perform min/max
      */
    void build(const VariablePtr& variable, bool first);
    
    /** Build internal data from the requirements*/
    void processRequirements();
    
    /** Creates a matrix assigment from the jacobian of \p source_ corresponding
      * to variable \p x to the block of matrix described by \p M and \p range.
      * \p flip indicates a sign change if \p true.
      */
    void addMatrixAssignment(Variable& x, MatrixFunction M, const Range& range, bool flip);
    
    /** Creates the assignements due to susbtituing the variable \p x by the
      * linear expression given by \p sub. The target is given by \p M.
      * \p flip indicates a sign change if \p true.
      */
    void addMatrixSubstitutionAssignments(const VariableVector& variables, Variable& x, MatrixFunction M, 
                                          const function::BasicLinearFunction& sub, bool flip);
    
    /** Creates and assigment between the vector given by \p f and the one given
      * by \p v, taking care of the RHS conventions for the source and the
      * target. The assignement type is given by the template parameter \p A.
      * \p flip indicates a sign change if \p true.
      */
    template<AssignType A = AssignType::COPY, typename From, typename To>
    void addVectorAssignment(const From& f, To v, bool flip, bool useFRHS = true, bool useTRHS = true);

    /** Creates and assigment between the vector given by \p f and the one given
    * by \p v, taking care of the RHS conventions for the source and the
    * target. The source is premultiplied by the inverse of the diagonal matrix
    * \p D. The assignement type is given by the template parameter \p A.
    * \p flip indicates a sign change if \p true.
    */
    template<AssignType A = AssignType::COPY, typename From, typename To>
    void addVectorAssignment(const From& f, To v, const MatrixConstRef& D, bool flip,
                             bool useFRHS = true, bool useTRHS = true);
    
    /** Creates the assignements due to the substitution of variable \p x by the
      * linear expression \p sub. The target is given by \p v.
      */
    void addVectorSubstitutionAssignments(const function::BasicLinearFunction& sub, 
                                          VectorFunction v, Variable& x, bool flip);
    
    /** Create a vector assignement where the source is a constant. The target
      * is given by \p v and the type of assignement by \p A.
      */
    template<AssignType A = AssignType::COPY, typename To>
    void addVectorAssignment(double d, To v, bool flip = false,
                             bool useFRHS = true, bool useTRHS = true);

    /** Create a vector assignement where the source is a constant that is
      * premultiplied by the inverse of a diagonal matrix \p D.
      * The target is given by \p v and the type of assignement by \p A.
      */
    template<AssignType A = AssignType::COPY, typename To>
    void addVectorAssignment(double d, To v, const MatrixConstRef& D, bool flip = false,
                             bool useFRHS = true, bool useTRHS = true);
    
    /** Creates an assignement setting to zero the matrix block given by \p M
      * and \p range. The variable \p x is simply stored in the corresponding
      * \p MatrixAssignment.
      */
    void addZeroAssignment(Variable& x, MatrixFunction M, const Range& range);
    
    /** Calls addAssignments(const VariableVector& variables, MatrixFunction M,
      * RHSFunction f1, VectorFunction v1, RHSFunction f2, VectorFunction v2, 
      * bool flip) for a single-sided case.
      */
    void addAssignments(const VariableVector& variables, MatrixFunction M,
                        RHSFunction f, VectorFunction v, bool flip);

    /** Calls addAssignments(const VariableVector& variables, MatrixFunction M,
      * RHSFunction f1, VectorFunction v1, RHSFunction f2, VectorFunction v2, 
      * bool flip) for a double-sided case.
      */
    void addAssignments(const VariableVector& variables, MatrixFunction M,
                        RHSFunction f1, VectorFunction v1, 
                        RHSFunction f2, VectorFunction v2);

    /** Creates all the matrix assignements between the source and the target,
      * as well as the vector assignements described by \p f1 and \p v1 and
      * optionnally by \p f2 and \p v2 if those are not \p nullptr.
      * This method is called after the constraint::Type conventions of the
      * source and the target have been processed (resulting in the choice of
      * \p M, \p f1, \p v1, \p f2, \p v2 and \p flip). It handles internally the
      * substitutions.
      */
    void addAssignments(const VariableVector& variables, MatrixFunction M,
                        RHSFunction f1, VectorFunction v1, 
                        RHSFunction f2, VectorFunction v2, bool flip);

    void addBound(const VariablePtr& variable, RHSFunction f, bool first);

    template<typename L, typename U>
    void addBounds(const VariablePtr& variable, L l, U u, bool first);

    template<typename L, typename U, typename TL, typename TU>
    void addBounds(const VariablePtr& variable, L l, U u, TL tl, TU tu, bool first);

    /** Create the compiled assignment between \p from and \p to, taking into
      * account the requirements and the possible sign flip indicated by 
      * \p flip.
      */
    template<typename T, AssignType A, typename U>
    CompiledAssignmentWrapper<T> createAssignment(const U& from, const Eigen::Ref<T>& to, bool flip = false);

    /** Create the compiled substitution assignement to = Mult * from (vector 
      * case) or to = from * mult (matrix case) taking into account the 
      * requirements and \p flip
      */
    template<typename T, AssignType A, MatrixMult M = GENERAL, typename U, typename V>
    CompiledAssignmentWrapper<T> createMultiplicationAssignment(const U& from, const Eigen::Ref<T>& to, const V& Mult, bool flip = false);

    /** The source of the assignment.*/
    LinearConstraintPtr source_;
    /** The target of the assignment.*/
    AssignmentTarget target_;
    /** The weight used to emulate hierarchy in a weight scheme.*/
    double scalarizationWeight_;
    /** The requirements attached to the source.*/
    std::shared_ptr<requirements::SolvingRequirements> requirements_;
    /** Indicates if the requirements use a default weight AND the scalarizationWeight is 1.*/
    bool useDefaultScalarWeight_;
    /** Indicates if the requirements use a defautl anisotropic weight.*/
    bool useDefaultAnisotropicWeight_;
    /** All the assignements that are setting the initial values of the targeted blocks*/
    std::vector<MatrixAssignment> matrixAssignments_;
    /** All assignments due to substitutions. We separe them from matrixAssignments_
      * because these assignements add to existing values, and we need to be sure
      * that the assignements in matrixAssignments_ have been carried out before.
      */
    std::vector<MatrixAssignment> matrixSubstitutionAssignments_;
    /** All the initial rhs assignments*/
    std::vector<VectorAssignment> vectorAssignments_;
    /** The additional rhs assignments due to substitutions. As for matrix
      * assignments, they need to be carried out after those of vectorAssignments_.
      */
    std::vector<VectorSubstitutionAssignement> vectorSubstitutionAssignments_;

    /** Processed requirements*/
    double scalarWeight_;
    double minusScalarWeight_;
    Eigen::VectorXd anisotropicWeight_;
    Eigen::VectorXd minusAnisotropicWeight_;

    /** Data for substitutions */
    VariableVector substitutedVariables_;
    std::vector<std::shared_ptr<function::BasicLinearFunction>> variableSubstitutions_;

    /** Temporary vectors for bound assignements*/
    Eigen::VectorXd tmp1_;
    Eigen::VectorXd tmp2_;
    Eigen::VectorXd tmp3_;
    Eigen::VectorXd tmp4_;
  };

  template<typename L, typename U>
  inline void Assignment::addBounds(const VariablePtr& variable, L l, U u, bool first)
  {
    const auto& J = source_->jacobian(*variable);
    if (substitutedVariables_.contains(*variable))
    {
      addBounds(variable, l, u, VectorRef(tmp1_), VectorRef(tmp2_), first);
    }
    else
    {
      addBounds(variable, l, u, &AssignmentTarget::l, &AssignmentTarget::u, first);
    }
  }

  template<typename L, typename U, typename TL, typename TU>
  inline void Assignment::addBounds(const VariablePtr& variable, L l, U u, TL tl, TU tu, bool first)
  {
    const auto& J = source_->jacobian(*variable);
    if (J.properties().isIdentity())
    {
      if (first)
      {
        addVectorAssignment(l, tl, false);
        addVectorAssignment(u, tu, false);
      }
      else
      {
        addVectorAssignment<MAX>(l, tl, false);
        addVectorAssignment<MIN>(u, tu, false);
      }
    }
    else if (J.properties().isMinusIdentity())
    {
      if (first)
      {
        addVectorAssignment(l, tu, true);
        addVectorAssignment(u, tl, true);
      }
      else
      {
        addVectorAssignment<MAX>(u, tl, true);
        addVectorAssignment<MIN>(l, tu, true);
      }
    }
    else
    {
      assert(J.properties().isDiagonal());
      tmp1_.resize(variable->size());
      tmp2_.resize(variable->size());
      tmp3_.resize(variable->size());
      tmp4_.resize(variable->size());
      addVectorAssignment(l, VectorRef(tmp1_), J, false, true, false);  // tmp1_ = inv(J)*l
      addVectorAssignment(u, VectorRef(tmp2_), J, false, true, false);  // tmp2_ = inv(J)*u
      addVectorAssignment(VectorConstRef(tmp1_), VectorRef(tmp3_), false, false, false); // tmp3_ = inv(J)*l
      addVectorAssignment(VectorConstRef(tmp2_), VectorRef(tmp4_), false, false, false); // tmp4_ = inv(J)*u
      addVectorAssignment<MIN>(VectorConstRef(tmp2_), VectorRef(tmp3_), false, false, false); //tmp3_ = min(inv(J)*l, inv(J)*u)
      addVectorAssignment<MAX>(VectorConstRef(tmp1_), VectorRef(tmp4_), false, false, false); //tmp4_ = max(inv(J)*l, inv(J)*u)
      if (first)
      {
        addVectorAssignment(VectorConstRef(tmp3_), tl, false, false, true);
        addVectorAssignment(VectorConstRef(tmp4_), tu, false, false, true);
      }
      else
      {
        addVectorAssignment<MAX>(VectorConstRef(tmp3_), tl, false, false, true);
        addVectorAssignment<MIN>(VectorConstRef(tmp4_), tu, false, false, true);
      }
    }
  }

  template<typename T, AssignType A, typename U>
  inline CompiledAssignmentWrapper<T> Assignment::createAssignment(const U& from, const Eigen::Ref<T>& to, bool flip)
  {
    using Wrapper = CompiledAssignmentWrapper<typename std::conditional<std::is_arithmetic<U>::value, Eigen::VectorXd, T>::type>;
    const Source F = std::is_arithmetic<U>::value ? CONSTANT : EXTERNAL;

    if (useDefaultAnisotropicWeight_)
    {
      if (useDefaultScalarWeight_)
      {
        if (flip)
          return Wrapper::template make<A, MINUS, IDENTITY, F>(to, from);
        else
          return Wrapper::template make<A, NONE, IDENTITY, F>(to, from);
      }
      else
      {
        if (flip)
          return Wrapper::template make<A, SCALAR, IDENTITY, F>(to, from, minusScalarWeight_);
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

  template<typename T, AssignType A, MatrixMult M, typename U, typename V>
  inline CompiledAssignmentWrapper<T> Assignment::createMultiplicationAssignment(const U& from, const Eigen::Ref<T>& to, const V& Mult, bool flip)
  {
    using Wrapper = CompiledAssignmentWrapper<typename std::conditional<std::is_arithmetic<U>::value, Eigen::VectorXd, T>::type>;
    const Source F = std::is_arithmetic<U>::value ? CONSTANT : EXTERNAL;

    if (useDefaultAnisotropicWeight_)
    {
      if (useDefaultScalarWeight_)
      {
        if (flip)
          return Wrapper::template make<A, MINUS, M, F>(to, from, Mult);
        else
          return Wrapper::template make<A, NONE, M, F>(to, from, Mult);
      }
      else
      {
        if (flip)
          return Wrapper::template make<A, SCALAR, M, F>(to, from, minusScalarWeight_, Mult);
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

  /** Helper function for addVectorAssignment returning the Eigen::Ref to the
    * source vector, where the argument is the vector in question.
    */
  inline VectorConstRef retrieveSource(LinearConstraintPtr, const VectorConstRef& f)
  {
    return f;
  }

  /** Helper function for addVectorAssignment returning the Eigen::Ref to the
    * source vector, where the argument is a method returning the vector.
    */
  inline VectorConstRef retrieveSource(LinearConstraintPtr s, const Assignment::RHSFunction& f)
  {
    return (s.get()->*f)();
  }

  /** Helper function for addVectorAssignment and addConstantAssignment
    * returning the Eigen::Ref to the target vector, where the argument is the
    * vector in question.
    */
  inline VectorRef retrieveTarget(AssignmentTarget&, VectorRef v)
  {
    return v;
  }

  /** Helper function for addVectorAssignment and addConstantAssignment
    * returning the Eigen::Ref to the target vector, where the argument is a
    * method returning the vector.
    */
  inline VectorRef retrieveTarget(AssignmentTarget& t, Assignment::VectorFunction v)
  {
    return (t.*v)();
  }

  /** Helper function for addVectorAssignment returning \p nullptr.*/
  inline std::nullptr_t nullifyIfVecRef(const VectorConstRef&) { return nullptr; }
  /** Helper function for addVectorAssignment returning \p nullptr.*/
  inline std::nullptr_t nullifyIfVecRef(VectorRef) { return nullptr; }
  /** Helper function for addVectorAssignment returning its argument.*/
  inline Assignment::RHSFunction nullifyIfVecRef(Assignment::RHSFunction f) { return f; }
  /** Helper function for addVectorAssignment returning its argument.*/
  inline Assignment::VectorFunction nullifyIfVecRef(Assignment::VectorFunction v) { return v; }

  template<AssignType A, typename From, typename To>
  inline void Assignment::addVectorAssignment(const From& f, To v, bool flip, bool useFRHS, bool useTRHS)
  {
    bool useSource = source_->rhs() != constraint::RHS::ZERO
                  || std::is_same<From, VectorConstRef>::value;
    if (useSource)
    {
      // So far, the sign flip has been deduced only from the ConstraintType of the source
      // and the target. Now we need to take into account the constraint::RHS as well.
      if (source_->rhs() == constraint::RHS::OPPOSITE && useFRHS)
        flip = !flip;
      if (target_.constraintRhs() == constraint::RHS::OPPOSITE && useTRHS)
        flip = !flip;

      const VectorRef& to = retrieveTarget(target_, v);
      const auto& from = retrieveSource(source_, f);
      auto w = createAssignment<Eigen::VectorXd, A>(from, to, flip);
      bool b = nullifyIfVecRef(f);
      vectorAssignments_.push_back({ w, b, nullifyIfVecRef(f), nullifyIfVecRef(v) });
    }
    else
    {
      if (target_.constraintRhs() != constraint::RHS::ZERO)
      {
        const VectorRef& to = retrieveTarget(target_, v);
        auto w = CompiledAssignmentWrapper<Eigen::VectorXd>::make<A, NONE, IDENTITY, ZERO>(to);
        vectorAssignments_.push_back({ w, false, nullptr, nullifyIfVecRef(v) });
      }
    }
  }

  template<AssignType A, typename From, typename To>
  inline void Assignment::addVectorAssignment(const From& f, To v, const MatrixConstRef& D, bool flip, bool useFRHS, bool useTRHS)
  {
    bool useSource = source_->rhs() != constraint::RHS::ZERO
                  || std::is_same<From, VectorConstRef>::value;
    if (useSource)
    {
      // So far, the sign flip has been deduced only from the ConstraintType of the source
      // and the target. Now we need to take into account the constraint::RHS as well.
      if (source_->rhs() == constraint::RHS::OPPOSITE && useFRHS)
        flip = !flip;
      if (target_.constraintRhs() == constraint::RHS::OPPOSITE && useTRHS)
        flip = !flip;

      const VectorRef& to = retrieveTarget(target_, v);
      const auto& from = retrieveSource(source_, f);
      auto w = createMultiplicationAssignment<Eigen::VectorXd, A, INVERSE_DIAGONAL>(from, to, D, flip);
      bool b = nullifyIfVecRef(f);
      vectorAssignments_.push_back({ w, b, nullifyIfVecRef(f), nullifyIfVecRef(v) });
    }
    else
    {
      if (target_.constraintRhs() != constraint::RHS::ZERO)
      {
        const VectorRef& to = retrieveTarget(target_, v);
        auto w = CompiledAssignmentWrapper<Eigen::VectorXd>::make<A, NONE, IDENTITY, ZERO>(to);
        vectorAssignments_.push_back({ w, false, nullptr, nullifyIfVecRef(v) });
      }
    }
  }

  template<AssignType A, typename To>
  inline void Assignment::addVectorAssignment(double d, To v, bool flip, bool, bool)
  {
    const VectorRef& to = retrieveTarget(target_, v);
    auto w = createAssignment<Eigen::VectorXd, A>(d, to, flip);
    vectorAssignments_.push_back({ w, false, nullptr, nullifyIfVecRef(v) });
  }

  template<AssignType A, typename To>
  inline void Assignment::addVectorAssignment(double d, To v, const MatrixConstRef& D, bool flip, bool, bool)
  {
    const VectorRef& to = retrieveTarget(target_, v);
    auto w = createMultiplicationAssignment<Eigen::VectorXd, A>(d, to, D, flip);
    vectorAssignments_.push_back({ w, false, nullptr, nullifyIfVecRef(v) });
  }



}  // namespace internal

}  // namespace scheme

}  // namespace tvm
