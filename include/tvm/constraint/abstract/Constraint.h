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

#include <tvm/constraint/enums.h>
#include <tvm/internal/FirstOrderProvider.h>
#include <tvm/graph/abstract/OutputSelector.h>

#include <Eigen/Core>

#include <memory>

namespace tvm
{

namespace constraint
{

namespace abstract
{

  /** Base class for representing a constraint.
    *
    * It manages the enabling/disabling of the outputs L, U and E (depending
    * on its type), and the memory of the associated cache.
    *
    * FIXME: have the updateValue here and add an output check()
    */
  class TVM_DLLAPI Constraint : public graph::abstract::OutputSelector<Constraint, tvm::internal::FirstOrderProvider>
  {
  public:
    SET_OUTPUTS(Constraint, L, U, E)

    /** Note: by default, these methods return the cached value.
    * However, they are virtual in case the user might want to bypass the cache.
    * This would be typically the case if he/she wants to directly return the
    * output of another method.
    */
    virtual const Eigen::VectorXd& l() const;
    virtual const Eigen::VectorXd& u() const;
    virtual const Eigen::VectorXd& e() const;

    Type type() const;
    RHS rhs() const;

  protected:
    Constraint(Type ct, RHS cr, int m=0);
    void resizeCache() override;
    void resizeRHS();

    Eigen::VectorXd l_;
    Eigen::VectorXd u_;
    Eigen::VectorXd e_;

  private:
    Type  cstrType_;
    RHS   constraintRhs_;

    bool usel_;
    bool useu_;
    bool usee_;
  };


  inline const Eigen::VectorXd& Constraint::l() const
  {
    return l_;
  }

  inline const Eigen::VectorXd& Constraint::u() const
  {
    return u_;
  }

  inline const Eigen::VectorXd& Constraint::e() const
  {
    return e_;
  }

}  // namespace abstract

}  // namespace constraint

}  // namespace tvm
