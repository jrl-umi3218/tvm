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
        if (vars.numberOfVariables() == 0) return false;
        const auto& p = c->jacobian(*vars[0]).properties();
        return (c->linearIn(*vars[0]) && vars.numberOfVariables() == 1
                && p.isDiagonal() && p.isInvertible());
      }

      bool isBound(const ConstraintPtr& c, const hint::internal::Substitutions& subs)
      {
        return isBound(c, subs.variables(), subs.variableSubstitutions());
      }

      bool TVM_DLLAPI isBound(const ConstraintPtr & c, const std::vector<VariablePtr>& x, const std::vector<std::shared_ptr<function::BasicLinearFunction>>& xsub)
      {
        if (isBound(c))
        {
          auto it = std::find(x.begin(), x.end(), c->variables()[0]);
          if (it == x.end())
          {
            // There is no susbtitution, this is a bound
            return true;
          }
          else
          {
            const auto& sub = xsub[static_cast<size_t>(it - x.begin())];
            const auto& v = sub->variables();
            if (v.numberOfVariables() != 1) return false;
            const auto& p = sub->jacobian(*v[0]).properties();
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

      bool TVM_DLLAPI canBeUsedAsBound(const ConstraintPtr& c, const hint::internal::Substitutions& subs,
                                       constraint::Type targetConvention)
      {
        return canBeUsedAsBound(c, subs.variables(), subs.variableSubstitutions(), targetConvention);
      }

      bool TVM_DLLAPI canBeUsedAsBound(const ConstraintPtr& c, const std::vector<VariablePtr>& x,
                                       const std::vector<std::shared_ptr<function::BasicLinearFunction>>& xsub,
                                       constraint::Type targetConvention)
      {
        using constraint::Type;
        if (isBound(c, x, xsub))
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
          // We do not consider the case where both matrices are undefinite but
          // their product would be positive or negative definite.

          bool case1 = false;
          switch (targetConvention)
          {
          case Type::EQUAL:         return false;
          case Type::GREATER_THAN:  
            if (c->type() == Type::GREATER_THAN) case1 = true;
            else if (c->type() == Type::LOWER_THAN) case1 = false;
            else return false;
          case Type::LOWER_THAN:
            if (c->type() == Type::GREATER_THAN) case1 = false;
            else if (c->type() == Type::LOWER_THAN) case1 = true;
            else return false;
          case Type::DOUBLE_SIDED: return true;
          default: assert(false);
          }

          const auto& p = c->jacobian(*c->variables()[0]).properties();
          auto it = std::find(x.begin(), x.end(), c->variables()[0]);
          if (it == x.end())
          {
            if (case1) return p.isPositiveDefinite();
            else       return p.isNegativeDefinite();
          }
          else
          {
            const auto& sub = xsub[static_cast<size_t>(it - x.begin())];
            const auto& v = sub->variables();
            const auto& ps = sub->jacobian(*x[0]).properties();
            if (case1) return (p*ps).isPositiveDefinite();
            else       return (p*ps).isNegativeDefinite();
          }
        }
        else
        {
          return false;
        }
      }
    }

  }

}