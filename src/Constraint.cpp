#include "Constraint.h"
#include "exceptions.h"

namespace tvm
{
  ConstraintBase::ConstraintBase(int m)
    : FirstOrderProvider(m)
  {
  }

  Constraint::Constraint(ConstraintType ct, ConstraintRHS cr, int m)
    : data::OutputSelector<ConstraintBase>(m)
    , cstrType_(ct)
    , constraintRhs_(cr)
    , usel_((ct == ConstraintType::GREATER_THAN || ct == ConstraintType::DOUBLE_SIDED) && cr != ConstraintRHS::ZERO)
    , useu_((ct == ConstraintType::LOWER_THAN || ct == ConstraintType::DOUBLE_SIDED) && cr != ConstraintRHS::ZERO)
    , usee_(ct == ConstraintType::EQUAL && cr != ConstraintRHS::ZERO)
  {
    if (ct == ConstraintType::DOUBLE_SIDED && cr == ConstraintRHS::ZERO)
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
    if (usel_)
      l_.resize(size());

    if (useu_)
      u_.resize(size());

    if (usee_)
      e_.resize(size());
  }

  ConstraintType Constraint::constraintType() const
  {
    return cstrType_;
  }

  ConstraintRHS Constraint::constraintRhs() const
  {
    return constraintRhs_;
  }
}
