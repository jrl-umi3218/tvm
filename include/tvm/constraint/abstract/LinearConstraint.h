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

#include <tvm/defs.h>
#include <tvm/constraint/abstract/Constraint.h>

#include <Eigen/Core>

#include <initializer_list>
#include <memory>

namespace tvm
{

namespace constraint
{

namespace abstract
{

  /** Note: if you intend to use the Value output and your jacobian matrices
    * are not constant, you need to tie updateValue to the update of those
    * matrices.
    *
    * \dot
    * digraph "update graph" {
    *   rankdir="LR";
    *   {
    *     rank=same; node [shape=circle];
    *     x_i;
    *   }
    *   {
    *     uValue [label=Value];
    *   }
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
    *   x_i -> uValue [label="value()"];
    *   uValue -> Value;
    * }
    * \enddot
    */
  class TVM_DLLAPI LinearConstraint : public Constraint
  {
  public:
    SET_UPDATES(LinearConstraint, Value)

    /** Update the value of the constraint (i.e.) f(x) for a constraint
      * f(x) op rhs.
      */
    void updateValue();

  protected:
    /** Constructor. Only available to derived classes.
      * \param ct The constraint type
      * \param cr The rhs convention
      * \param The (output) size of the constraint
      */
    LinearConstraint(Type ct, RHS cr, int m);
  };

}  // namespace abstract

}  // namespace constraint

}  // namespace tvm
