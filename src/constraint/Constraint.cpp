/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/constraint/abstract/Constraint.h>

#include <tvm/exception/exceptions.h>

namespace tvm
{

namespace constraint
{

namespace abstract
{

Constraint::Constraint(Type ct, RHS cr, int m)
: graph::abstract::OutputSelector<Constraint, tvm::internal::FirstOrderProvider>(m), vectors_(ct, cr), cstrType_(ct),
  constraintRhs_(cr)
{
  if(ct == Type::DOUBLE_SIDED && cr == RHS::ZERO)
    throw std::runtime_error("The combination (ConstraintType::DOUBLE_SIDED, ConstraintRHS::ZERO) is forbidden. Please "
                             "use (ConstraintType::EQUAL, ConstraintRHS::ZERO) instead.");
  // FIXME: we make the choice here to have no "rhs" output when the ConstraintRHS is zero.
  // An alternative is to use and set to zero the relevant vectors, but then we need
  // to prevent a derived class to change their value.
  resizeCache();
  if(!vectors_.use_l())
    disableOutput(Output::L);
  if(!vectors_.use_u())
    disableOutput(Output::U);
  if(!vectors_.use_e())
    disableOutput(Output::E);
}

void Constraint::resizeCache()
{
  tvm::internal::FirstOrderProvider::resizeCache();
  vectors_.resize(size());
}

Type Constraint::type() const { return cstrType_; }

bool Constraint::isEquality() const { return cstrType_ == Type::EQUAL; }

RHS Constraint::rhs() const { return constraintRhs_; }

} // namespace abstract

} // namespace constraint

} // namespace tvm
