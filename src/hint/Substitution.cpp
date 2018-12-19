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

#include <tvm/hint/Substitution.h>

#include <tvm/Variable.h>
#include <tvm/constraint/abstract/LinearConstraint.h>

#include <sstream>

namespace tvm
{

namespace hint
{

  Substitution::Substitution(LinearConstraintPtr cstr, VariablePtr x, int rank,
                             const abstract::SubstitutionCalculator& calc)
    : Substitution(std::vector<LinearConstraintPtr>{ cstr }, std::vector<VariablePtr>{ x }, rank, calc)
  {
  }

  Substitution::Substitution(const std::vector<LinearConstraintPtr>& cstr, VariablePtr x, int rank,
                             const abstract::SubstitutionCalculator& calc)
    : Substitution(cstr, std::vector<VariablePtr>{ x }, rank, calc)
  {
  }

  Substitution::Substitution(LinearConstraintPtr cstr, std::vector<VariablePtr>& x, int rank,
                             const abstract::SubstitutionCalculator& calc)
    : Substitution(std::vector<LinearConstraintPtr>{ cstr }, x, rank, calc)
  {
  }

  Substitution::Substitution(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank,
                             const abstract::SubstitutionCalculator& calc)
    : rank_(rank), m_(0), constraints_(cstr), x_(x)
  {
    if (rank == constant::fullRank)
    {
      int r = 0;
      for (const auto& c : cstr)
      {
        r += c->size();
      }
      rank_ = r;
    }
    for (const auto& c : constraints_)
    {
      m_ += c->size();
    }
    check();
    calculator_ = calc.impl(cstr, x, rank_);
  }

  int Substitution::rank() const
  {
    return rank_;
  }

  int Substitution::m() const
  {
    return m_;
  }

  const std::vector<LinearConstraintPtr>& Substitution::constraints() const
  {
    return constraints_;
  }

  const std::vector<VariablePtr>& Substitution::variables() const
  {
    return x_;
  }

  bool Substitution::isSimple() const
  {
    return constraints_.size() == 1 && x_.size() == 1;
  }

  std::shared_ptr<abstract::SubstitutionCalculatorImpl> Substitution::calculator() const
  {
    return calculator_;
  }

  void Substitution::check() const
  {
    //all constraints need to be equality
    for (const auto& c : constraints_)
    {
      // It could be possible to accept double sided inequality constraints with both bounds equal
      // but in this case we would need to assert that the bounds are and stay equals (the latter
      // requiring some form of annotation).
      // If the bounds are known to be always equal, the user should simply use an equality.
      // If only some of the bounds are equal but not all, we would need an annotation giving the
      // pattern, and handling it would add a lot of complexity for a not so common case.
      if (c->type() != constraint::Type::EQUAL)
      {
        throw std::runtime_error("Substitution can only be done using equality constraints.");
      }
    }

    std::vector<bool> varIsInOneConstr(x_.size(), false);
    std::vector<bool> constrHasOneVar(constraints_.size(), false);

    for (size_t i = 0; i < constraints_.size(); ++i)
    {
      for (size_t j = 0; j < x_.size(); ++j)
      {
        if (constraints_[i]->variables().contains(*x_[j]))
        {
          varIsInOneConstr[j] = true;
          constrHasOneVar[i] = true;
        }
      }
    }

    //all variables need to appear in at least one constraints
    for (size_t j = 0; j < x_.size(); ++j)
    {
      if (!varIsInOneConstr[j])
      {
        std::stringstream ss;
        ss << "The variable " << x_[j] << " cannot be used for substitution: it does not appear in any of the given constraints.";
        throw std::runtime_error(ss.str());
      }
    }

    //all constraints need to contain at least one of the listed variables
    for (size_t i = 0; i < constraints_.size(); ++i)
    {
      if (!constrHasOneVar[i])
      {
        std::stringstream ss;
        ss << "The " << (i + 1) << "-th constraint (index " << i << ") does not contains any of the specified variables.";
        throw std::runtime_error(ss.str());
      }
    }

    //rank of the substitution cannot be greater than the size of the variables
    int s = 0;
    for (const auto& xi : x_)
    {
      s += xi->size();
    }
    if (rank_ > s)
    {
      throw std::runtime_error("Substitution is not feasible: too many equations.");
    }
  }

}

}
