/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/constraint/abstract/Constraint.h>
#include <tvm/hint/internal/Substitutions.h>
#include <tvm/scheme/internal/helpers.h>

#include <numeric>

namespace tvm
{

namespace scheme
{

namespace internal
{

bool isBound(const ConstraintPtr & c)
{
  const auto & vars = c->variables();
  if(vars.numberOfVariables() == 0)
    return false;
  const auto & jac = c->jacobian(vars[0]);
  const auto & p = jac.properties();
  return (c->linearIn(*vars[0]) && vars.numberOfVariables() == 1 && p.isDiagonal() && p.isInvertible());
}

bool isBound(const ConstraintPtr & c, const hint::internal::Substitutions & subs)
{
  return isBound(c, subs.variables(), subs.variableSubstitutions());
}

bool isBound(const ConstraintPtr & c,
             const std::vector<VariablePtr> & x,
             const std::vector<std::shared_ptr<function::BasicLinearFunction>> & xsub)
{
  if(isBound(c))
  {
    const auto & cx = c->variables()[0];
    auto it = std::find(x.begin(), x.end(), cx);
    if(it == x.end())
    {
      // cx is not part of x, but part of it could be substituted. We check for that case.
      if(std::find_if(x.begin(), x.end(), [&cx](const auto & v) { return cx->intersects(*v); }) != x.end())
      {
        // For now, we don't deal with subvariable substitution in bounds and demote c to a general constraint
        // TODO proper variable substitution in bounds.
        return false;
      }
      // There is no substitution, this is a bound
      return true;
    }
    else
    {
      const auto & sub = xsub[static_cast<size_t>(it - x.begin())];
      const auto & v = sub->variables();
      if(v.numberOfVariables() != 1)
        return false;
      const auto & jac = sub->jacobian(*v[0]);
      const auto & p = jac.properties();
      // There could be 0 variables in sub. In that case we have a trivial
      // constraint that we do not consider as a bound.
      return (p.isDiagonal() && p.isInvertible());
    }
  }
  else
  {
    // We do not consider substitutions that would make a non-bound constraint
    // become a bound. So if the original constraint is not a bound we return
    // false.
    return false;
  }
}

bool canBeUsedAsBound(const ConstraintPtr & c,
                      const hint::internal::Substitutions & subs,
                      constraint::Type targetConvention)
{
  return canBeUsedAsBound(c, subs.variables(), subs.variableSubstitutions(), targetConvention);
}

bool canBeUsedAsBound(const ConstraintPtr & c,
                      const std::vector<VariablePtr> & x,
                      const std::vector<std::shared_ptr<function::BasicLinearFunction>> & xsub,
                      constraint::Type targetConvention)
{
  using constraint::Type;
  if(isBound(c, x, xsub))
  {
    // We have the following cases (row: constraint convention, col: target
    // convention):
    //     |  =  |  >= |  <= |  <<
    // ----+-----+-----+-----+-----
    //  =  |  NO |  NO |  NO |  OK
    //  >= |  NO |  1  |  2  |  OK
    //  <= |  NO |  2  |  1  |  OK
    //  << |  NO |  NO |  NO |  OK
    // with = : EQUAL, >= : GREATER_THAN, <= : LOWER_THAN, << : DOUBLE_SIDED
    // NO: case note possible, OK: case possible without particular
    // positiveness properties of the matrix/matrices
    // 1: possible if diagonal matrix is positive definite or both the
    // the constraint matrix and the matrix is in the substitutions are
    // negative definite,
    // 2: possible if diagonal matrix is negative definite or both the
    // the constraint matrix and the matrix is in the substitutions are
    // positive definite.
    // We do not consider the case where both matrices are indefinite but
    // their product would be positive or negative definite.

    bool case1 = false;
    switch(targetConvention)
    {
      case Type::EQUAL:
        return false;
      case Type::GREATER_THAN:
        if(c->type() == Type::GREATER_THAN)
          case1 = true;
        else if(c->type() == Type::LOWER_THAN)
          case1 = false;
        else
          return false;
        break;
      case Type::LOWER_THAN:
        if(c->type() == Type::GREATER_THAN)
          case1 = false;
        else if(c->type() == Type::LOWER_THAN)
          case1 = true;
        else
          return false;
        break;
      case Type::DOUBLE_SIDED:
        return true;
      default:
        assert(false);
    }

    const auto & p = c->jacobian(*c->variables()[0]).properties();
    auto it = std::find(x.begin(), x.end(), c->variables()[0]);
    if(it == x.end())
    {
      if(case1)
        return p.isPositiveDefinite();
      else
        return p.isNegativeDefinite();
    }
    else
    {
      const auto & sub = xsub[static_cast<size_t>(it - x.begin())];
      auto ps = sub->jacobian(*x[0]).properties();
      if(case1)
        return (p * ps).isPositiveDefinite();
      else
        return (p * ps).isNegativeDefinite();
    }
  }
  else
  {
    return false;
  }
}
} // namespace internal

} // namespace scheme

} // namespace tvm
