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
#include <tvm/defs.h>

namespace tvm
{

namespace utils
{
  class CheckOptions
  {
  public:
    double step = 1e-7;
    double prec = 1e-6;
    bool verbose = true;
  };

  /** Check the jacobian matrices of function f by forward finite differences.*/
  bool TVM_DLLAPI checkJacobian(FunctionPtr f, CheckOptions opt = CheckOptions());

  /** Check the velocity of the function by comparing it to J*x. Assume that the
    * jacobian matrices are correct.
    */
  bool TVM_DLLAPI checkVelocity(FunctionPtr f, CheckOptions opt = CheckOptions());
  bool TVM_DLLAPI checkNormalAcceleration(FunctionPtr f, CheckOptions opt = CheckOptions());
}

}