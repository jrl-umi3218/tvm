#include <tvm/constraint/abstract/Constraint.h>

#include <tvm/exception/exceptions.h>

namespace tvm
{

namespace constraint
{

namespace abstract
{

  Constraint::Constraint(Type ct, RHS cr, int m)
    : graph::abstract::OutputSelector<internal::ConstraintBase>(m)
    , cstrType_(ct)
    , constraintRhs_(cr)
    , usel_((ct == Type::GREATER_THAN || ct == Type::DOUBLE_SIDED) && cr != RHS::ZERO)
    , useu_((ct == Type::LOWER_THAN || ct == Type::DOUBLE_SIDED) && cr != RHS::ZERO)
    , usee_(ct == Type::EQUAL && cr != RHS::ZERO)
  {
    if (ct == Type::DOUBLE_SIDED && cr == RHS::ZERO)
      throw std::runtime_error("The combination (ConstraintType::DOUBLE_SIDED, ConstraintRHS::ZERO) is forbidden. Please use (ConstraintType::EQUAL, ConstraintRHS::ZERO) instead.");
    //FIXME: we make the choice here to have no "rhs" output when the ConstraintRHS is zero.
    //An alternative is to use and set to zero the relevant vectors, but then we need
    //to prevent a derived class to change their value.
    resizeCache();
    if (!usel_)
      disableOutput(Output::L);
    if (!useu_)
      disableOutput(Output::U);
    if (!usee_)
      disableOutput(Output::E);
  }

  void Constraint::resizeCache()
  {
    tvm::internal::FirstOrderProvider::resizeCache();
    resizeRHS();
  }

  void Constraint::resizeRHS()
  {
    if (usel_)
      l_.resize(size());

    if (useu_)
      u_.resize(size());

    if (usee_)
      e_.resize(size());
  }

  Type Constraint::type() const
  {
    return cstrType_;
  }

  RHS Constraint::rhs() const
  {
    return constraintRhs_;
  }

}  // namespace abstract

}  // namespace constraint

}  // namespace tvm
