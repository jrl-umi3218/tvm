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

#include <tvm/VariableVector.h>
#include <tvm/scheme/internal/SchemeAbilities.h>

namespace tvm
{

class LinearizedControlProblem;

namespace scheme
{

namespace abstract
{

  /** Base class for solving a ControlProblem*/
  class TVM_DLLAPI ResolutionScheme
  {
  public:
    double big_number() const;
    void big_number(double big);

  protected:
    /** Constructor, meant only for derived classes
      *
      * \param abilities The set of abilities for this scheme.
      * \param big A big number use to represent infinity, in particular when
      * specifying non-existing bounds (e.g. x <= Inf is given as x <= big).
      */
    ResolutionScheme(const internal::SchemeAbilities& abilities, double big = std::numeric_limits<double>::max()/2);

    void addVariable(VariablePtr var);
    void addVariable(const std::vector<VariablePtr>& vars);
    void removeVariable(Variable* v);
    void removeVariable(const std::vector<VariablePtr>& vars);

    /** The problem variable*/
    VariableVector x_;
    internal::SchemeAbilities abilities_;

    /** A number to use for infinite bounds*/
    double big_number_;
  };

  /** Base class for scheme solving linear problems*/
  class TVM_DLLAPI LinearResolutionScheme : public ResolutionScheme
  {
  public:
    void solve();

  protected:
    LinearResolutionScheme(const internal::SchemeAbilities& abilities, std::shared_ptr<LinearizedControlProblem> pb, double big = std::numeric_limits<double>::max() / 2);

    virtual void solve_() = 0;

    std::shared_ptr<LinearizedControlProblem> problem_;
  };

}  // namespace abstract

}  // namespace scheme

}  // namespace tvm
