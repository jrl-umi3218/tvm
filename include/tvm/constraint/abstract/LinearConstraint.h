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
