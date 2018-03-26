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
#include <tvm/constraint/internal/RHSVectors.h>
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
    * on its type and rhs convention).
    *
    * FIXME: have the updateValue here and add an output check()
    *
    * \dot
    * digraph "update graph" {
    *   rankdir="LR";
    *   {
    *     rank = same; node [shape=hexagon];
    *     Value; Jacobian; L; U; E;
    *   }
    *   {
    *     rank = same; node [style=invis, label=""];
    *     outValue; outJacobian; outL; outU; outE;
    *   }
    *   Value -> outValue [label="value()"];
    *   Jacobian -> outJacobian [label="jacobian(x_i)"];
    *   L -> outL [label="l()"];
    *   U -> outU [label="u()"];
    *   E -> outE [label="e()"];
    * }
    * \enddot
    */
  class TVM_DLLAPI Constraint : public graph::abstract::OutputSelector<Constraint, tvm::internal::FirstOrderProvider>
  {
  public:
    SET_OUTPUTS(Constraint, L, U, E)

    /** \internal by default, these methods return the cached value.
      * However, they are virtual in case the user might want to bypass the cache.
      * This would be typically the case if he/she wants to directly return the
      * output of another method.
      */
    /** Return the vector \p l
      * \warning this does not if \p l exists for the given constraint conventions.
      */
    virtual const Eigen::VectorXd& l() const;
    /** Return the vector \p u
      * \warning this does not if \p u exists for the given constraint conventions.
      */
    virtual const Eigen::VectorXd& u() const;
    /** Return the vector \p e
      * \warning this does not if \p e exists for the given constraint conventions.
      */
    virtual const Eigen::VectorXd& e() const;

    /** Return the type of the constraint.*/
    Type type() const;
    /** Return the convention for the right-hand side \p e, \p l, \p u or both
      * \p l and \p u of the constraint.
      */
    RHS rhs() const;

  protected:
    /** Constructor. Only available to derived classes.
      * \param ct The constraint type
      * \param cr The rhs convention
      * \param The (output) size of the constraint
      */
    Constraint(Type ct, RHS cr, int m=0);

    /** Resize the cache (rhs vector(s), jacobian matrices,...) for the current
      * size of the constraint.
      */
    void resizeCache() override;

    /** Direct (non-const) access to \p l for derived classes */
    Eigen::VectorXd& lRef();
    /** Direct (non-const) access to \p u for derived classes */
    Eigen::VectorXd& uRef();
    /** Direct (non-const) access to \p e for derived classes */
    Eigen::VectorXd& eRef();

    /** Cache for l, u and e */
    internal::RHSVectors vectors_;

  private:
    Type  cstrType_;      // The constraint type
    RHS   constraintRhs_; // The rhs convention
  };


  inline const Eigen::VectorXd& Constraint::l() const
  {
    return vectors_.l();
  }

  inline const Eigen::VectorXd& Constraint::u() const
  {
    return vectors_.u();
  }

  inline const Eigen::VectorXd& Constraint::e() const
  {
    return vectors_.e();
  }

  inline Eigen::VectorXd& Constraint::lRef()
  {
    return vectors_.l();
  }

  inline Eigen::VectorXd& Constraint::uRef()
  {
    return vectors_.u();
  }

  inline Eigen::VectorXd& Constraint::eRef()
  {
    return vectors_.e();
  }

}  // namespace abstract

}  // namespace constraint

}  // namespace tvm
