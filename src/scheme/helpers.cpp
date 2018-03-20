#include <tvm/scheme/internal/helpers.h>
#include <tvm/constraint/abstract/Constraint.h>
#include <tvm/hint/internal/Substitutions.h>

namespace tvm
{

  namespace scheme
  {

    namespace internal
    {

      bool isBound(const ConstraintPtr& c)
      {
        const auto& vars = c->variables();
        const auto& p = c->jacobian(*vars[0]).properties();
        return (c->type() != constraint::Type::EQUAL && vars.numberOfVariables() == 1 && p.isDiagonal() && p.isInvertible());
      }

      bool isBound(const ConstraintPtr& c, const hint::internal::Substitutions& subs)
      {
        //FIXME we can do better here in particular if substituting a variable with
        // diag * another one
        const auto& vars = subs.variables();
        //for now the only test is wether or not the variable is substituted
        //if it is, we return false
        bool b = std::find(vars.begin(), vars.end(), c->variables()[0]) == vars.end();
        return isBound(c) && b;
      }
    }

  }

}