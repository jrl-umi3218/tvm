/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

namespace tvm::scheme::internal
{
template<typename L, typename U>
inline void Assignment::addBounds(const VariablePtr & variable, L l, U u, bool first)
{
  if(substitutedVariables_.contains(*variable))
  {
    addBounds(variable, l, u, VectorRef(data_->tmp1_), VectorRef(data_->tmp2_), first);
  }
  else
  {
    addBounds(variable, l, u, &AssignmentTarget::l, &AssignmentTarget::u, first);
  }
}

template<typename L, typename U, typename TL, typename TU>
inline void Assignment::addBounds(const VariablePtr & variable, L l, U u, TL tl, TU tu, bool first)
{
  const auto & J = source_->jacobian(*variable);
  if(J.properties().isIdentity())
  {
    if(first)
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
  else if(J.properties().isMinusIdentity())
  {
    if(first)
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
    data_->tmp1_.resize(variable->size());
    data_->tmp2_.resize(variable->size());
    data_->tmp3_.resize(variable->size());
    data_->tmp4_.resize(variable->size());
    addVectorAssignment(l, VectorRef(data_->tmp1_), J, false, true, false);                          // tmp1_ = inv(J)*l
    addVectorAssignment(u, VectorRef(data_->tmp2_), J, false, true, false);                          // tmp2_ = inv(J)*u
    addVectorAssignment(VectorConstRef(data_->tmp1_), VectorRef(data_->tmp3_), false, false, false); // tmp3_ = inv(J)*l
    addVectorAssignment(VectorConstRef(data_->tmp2_), VectorRef(data_->tmp4_), false, false, false); // tmp4_ = inv(J)*u
    addVectorAssignment<MIN>(VectorConstRef(data_->tmp2_), VectorRef(data_->tmp3_), false, false,
                             false); // tmp3_ = min(inv(J)*l, inv(J)*u)
    addVectorAssignment<MAX>(VectorConstRef(data_->tmp1_), VectorRef(data_->tmp4_), false, false,
                             false); // tmp4_ = max(inv(J)*l, inv(J)*u)
    if(first)
    {
      addVectorAssignment(VectorConstRef(data_->tmp3_), tl, false, false, true);
      addVectorAssignment(VectorConstRef(data_->tmp4_), tu, false, false, true);
    }
    else
    {
      addVectorAssignment<MAX>(VectorConstRef(data_->tmp3_), tl, false, false, true);
      addVectorAssignment<MIN>(VectorConstRef(data_->tmp4_), tu, false, false, true);
    }
  }
}

template<typename T, AssignType A, typename U>
inline CompiledAssignmentWrapper<T> Assignment::createAssignment(const U & from, const Eigen::Ref<T> & to, bool flip)
{
  using Wrapper =
      CompiledAssignmentWrapper<typename std::conditional<std::is_arithmetic<U>::value, Eigen::VectorXd, T>::type>;
  const Source F = std::is_arithmetic<U>::value ? CONSTANT : EXTERNAL;

  if(useDefaultAnisotropicWeight_)
  {
    if(useDefaultScalarWeight_)
    {
      if(flip)
        return Wrapper::template make<A, MINUS, IDENTITY, F>(to, from);
      else
        return Wrapper::template make<A, NONE, IDENTITY, F>(to, from);
    }
    else
    {
      if(flip)
        return Wrapper::template make<A, SCALAR, IDENTITY, F>(to, from, data_->minusScalarWeight_);
      else
        return Wrapper::template make<A, SCALAR, IDENTITY, F>(to, from, data_->scalarWeight_);
    }
  }
  else
  {
    if(flip)
      return Wrapper::template make<A, DIAGONAL, IDENTITY, F>(to, from, data_->minusAnisotropicWeight_);
    else
      return Wrapper::template make<A, DIAGONAL, IDENTITY, F>(to, from, data_->anisotropicWeight_);
  }
}

template<typename T, AssignType A, MatrixMult M, typename U, typename V>
inline CompiledAssignmentWrapper<T> Assignment::createMultiplicationAssignment(const U & from,
                                                                               const Eigen::Ref<T> & to,
                                                                               const V & Mult,
                                                                               bool flip)
{
  using Wrapper =
      CompiledAssignmentWrapper<typename std::conditional<std::is_arithmetic<U>::value, Eigen::VectorXd, T>::type>;
  const Source F = std::is_arithmetic<U>::value ? CONSTANT : EXTERNAL;

  if(useDefaultAnisotropicWeight_)
  {
    if(useDefaultScalarWeight_)
    {
      if(flip)
        return Wrapper::template make<A, MINUS, M, F>(to, from, Mult);
      else
        return Wrapper::template make<A, NONE, M, F>(to, from, Mult);
    }
    else
    {
      if(flip)
        return Wrapper::template make<A, SCALAR, M, F>(to, from, data_->minusScalarWeight_, Mult);
      else
        return Wrapper::template make<A, SCALAR, M, F>(to, from, data_->scalarWeight_, Mult);
    }
  }
  else
  {
    if(flip)
      return Wrapper::template make<A, DIAGONAL, M, F>(to, from, data_->minusAnisotropicWeight_, Mult);
    else
      return Wrapper::template make<A, DIAGONAL, M, F>(to, from, data_->anisotropicWeight_, Mult);
  }
}

/** Helper function for addVectorAssignment returning the Eigen::Ref to the
 * source vector, where the argument is the vector in question.
 */
inline VectorConstRef retrieveSource(LinearConstraintPtr, const VectorConstRef & f) { return f; }

/** Helper function for addVectorAssignment returning the Eigen::Ref to the
 * source vector, where the argument is a method returning the vector.
 */
inline VectorConstRef retrieveSource(LinearConstraintPtr s, const Assignment::RHSFunction & f)
{
  return (s.get()->*f)();
}

/** Helper function for addVectorAssignment and addConstantAssignment
 * returning the Eigen::Ref to the target vector, where the argument is the
 * vector in question.
 */
inline VectorRef retrieveTarget(AssignmentTarget &, VectorRef v) { return v; }

/** Helper function for addVectorAssignment and addConstantAssignment
 * returning the Eigen::Ref to the target vector, where the argument is a
 * method returning the vector.
 */
inline VectorRef retrieveTarget(AssignmentTarget & t, Assignment::VectorFunction v) { return (t.*v)(); }

/** Helper function for addVectorAssignment returning \p nullptr.*/
inline std::nullptr_t nullifyIfVecRef(const VectorConstRef &) { return nullptr; }
/** Helper function for addVectorAssignment returning \p nullptr.*/
inline std::nullptr_t nullifyIfVecRef(VectorRef) { return nullptr; }
/** Helper function for addVectorAssignment returning its argument.*/
inline Assignment::RHSFunction nullifyIfVecRef(Assignment::RHSFunction f) { return f; }
/** Helper function for addVectorAssignment returning its argument.*/
inline Assignment::VectorFunction nullifyIfVecRef(Assignment::VectorFunction v) { return v; }

template<AssignType A, typename From, typename To>
inline void Assignment::addVectorAssignment(const From & f, To v, bool flip, bool useFRHS, bool useTRHS)
{
  bool useSource = source_->rhs() != constraint::RHS::ZERO || std::is_same<From, VectorConstRef>::value;
  if(useSource)
  {
    // So far, the sign flip has been deduced only from the ConstraintType of the source
    // and the target. Now we need to take into account the constraint::RHS as well.
    if(source_->rhs() == constraint::RHS::OPPOSITE && useFRHS)
      flip = !flip;
    if(target_.constraintRhs() == constraint::RHS::OPPOSITE && useTRHS)
      flip = !flip;

    const VectorRef & to = retrieveTarget(target_, v);
    const auto & from = retrieveSource(source_, f);
    auto w = createAssignment<Eigen::VectorXd, A>(from, to, flip);
    bool b = nullifyIfVecRef(f);
    vectorAssignments_.push_back({w, b, nullifyIfVecRef(f), nullifyIfVecRef(v)});
  }
  else
  {
    if(target_.constraintRhs() != constraint::RHS::ZERO)
    {
      const VectorRef & to = retrieveTarget(target_, v);
      auto w = CompiledAssignmentWrapper<Eigen::VectorXd>::make<A, NONE, IDENTITY, ZERO>(to);
      vectorAssignments_.push_back({w, false, nullptr, nullifyIfVecRef(v)});
    }
  }
}

template<AssignType A, typename From, typename To>
inline void Assignment::addVectorAssignment(const From & f,
                                            To v,
                                            const MatrixConstRef & D,
                                            bool flip,
                                            bool useFRHS,
                                            bool useTRHS)
{
  bool useSource = source_->rhs() != constraint::RHS::ZERO || std::is_same<From, VectorConstRef>::value;
  if(useSource)
  {
    // So far, the sign flip has been deduced only from the ConstraintType of the source
    // and the target. Now we need to take into account the constraint::RHS as well.
    if(source_->rhs() == constraint::RHS::OPPOSITE && useFRHS)
      flip = !flip;
    if(target_.constraintRhs() == constraint::RHS::OPPOSITE && useTRHS)
      flip = !flip;

    const VectorRef & to = retrieveTarget(target_, v);
    const auto & from = retrieveSource(source_, f);
    auto w = createMultiplicationAssignment<Eigen::VectorXd, A, INVERSE_DIAGONAL>(from, to, D, flip);
    bool b = nullifyIfVecRef(f);
    vectorAssignments_.emplace_back(w, b, nullifyIfVecRef(f), nullifyIfVecRef(v));
  }
  else
  {
    if(target_.constraintRhs() != constraint::RHS::ZERO)
    {
      const VectorRef & to = retrieveTarget(target_, v);
      auto w = CompiledAssignmentWrapper<Eigen::VectorXd>::make<A, NONE, IDENTITY, ZERO>(to);
      vectorAssignments_.emplace_back(w, false, nullptr, nullifyIfVecRef(v));
    }
  }
}

template<AssignType A, typename To>
inline void Assignment::addVectorAssignment(double d, To v, bool flip, bool, bool)
{
  const VectorRef & to = retrieveTarget(target_, v);
  auto w = createAssignment<Eigen::VectorXd, A>(d, to, flip);
  vectorAssignments_.emplace_back(w, false, nullptr, nullifyIfVecRef(v));
}

template<AssignType A, typename To>
inline void Assignment::addVectorAssignment(double d, To v, const MatrixConstRef & D, bool flip, bool, bool)
{
  const VectorRef & to = retrieveTarget(target_, v);
  auto w = createMultiplicationAssignment<Eigen::VectorXd, A>(d, to, D, flip);
  vectorAssignments_.emplace_back(w, false, nullptr, nullifyIfVecRef(v));
}
} // namespace tvm::scheme::internal