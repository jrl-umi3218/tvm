/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/constraint/abstract/Constraint.h>
#include <tvm/defs.h>

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

} // namespace abstract

} // namespace constraint

} // namespace tvm
