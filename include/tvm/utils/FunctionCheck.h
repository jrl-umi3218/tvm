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
  /** A small structure to specify options for the following functions.
    *  - \a step is the increment that will be taken for finite difference schemes
    *  - \a prec is the precision with which the equality of two vectors is tested.
    *    It corresponds to the \a prec parameter of Eigen's \a isApprox method.
    *  - if \a verbose is true, the functions will display some indications when an
    *    mismatch is detected.
    */
  class CheckOptions
  {
  public:
    CheckOptions() : step(1e-7), prec(1e-6), verbose(true) {}
    CheckOptions(double s, double p, bool v) : step(s), prec(p), verbose(v) {}
    double step;
    double prec;
    bool verbose;
  };

  /** Check the jacobian matrices of function \a f by forward finite differences.*/
  bool TVM_DLLAPI checkJacobian(FunctionPtr f, CheckOptions opt = CheckOptions());

  /** Check the velocity of the function \a f by comparing it to J*x.
    * Assume that the jacobian matrices are correct.
    */
  bool TVM_DLLAPI checkVelocity(FunctionPtr f, CheckOptions opt = CheckOptions());

  /** Check the normal acceleration of the function \a f.
    * Noting v=f(x), this is done by comparing it to ddot{v}-J\ddot{x}, where
    * \ddot{x} is taken constant over the interval opt.step, and ddot{v} is obtained
    * by finite differences.
    * Assume that the jacobian matrices and the velocity are correct.
    */
  bool TVM_DLLAPI checkNormalAcceleration(FunctionPtr f, CheckOptions opt = CheckOptions());
}

}