
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

namespace tvm
{

namespace requirements
{

  /** This class represents the scalar weight alpha of a constraint,
    * within its priority level. It is meant to ajust the influence of
    * several constraints at the same level.
    *
    * Given a scalar weight \p alpha, and a constraint violation measurement
    * f(x), the product alpha*f(x) will be minimized.
    *
    * By default the weight is 1.
    */
  class TVM_DLLAPI Weight : public abstract::SingleSolvingRequirement<double>
  {
  public:
    Weight(double alpha=1);
  };

}  // namespace requirements

}  // namespace tvm
