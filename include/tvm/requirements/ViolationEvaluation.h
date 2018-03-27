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

#include <tvm/api.h>
#include <tvm/requirements/abstract/SingleSolvingRequirement.h>

#include <Eigen/Core>

namespace tvm
{

namespace requirements
{

  /** Given a constraint, let the vector v(x) be its componentwise
    * violation.
    *
    * For example, for the constraint c(x) = 0, we simply have v(x) = c(x),
    * for c(x) >= b, we have v(x) = max(b-c(x),0).
    *
    * This enumeration specifies how v(x) is made into a scalar measure
    * f(x) of this violation.
    */
  enum class ViolationEvaluationType
  {
    /** f(x) = sum(abs(v_i(x))) */
    L1,
    /** f(x) = v(x)^T*v(x) */
    L2,
    /** f(x) = max(abs(v_i(x))) */
    LINF
  };

  /** A class specifying how a constraint violation should be handled.
    * By default the L2 norm of the violation is used.
    * \sa ViolationEvaluationType
    */
  class TVM_DLLAPI ViolationEvaluation : public abstract::SingleSolvingRequirement<ViolationEvaluationType>
  {
  public:
    ViolationEvaluation(ViolationEvaluationType t = ViolationEvaluationType::L2);
  };

}  // namespace requirements

}  // namespace tvm
