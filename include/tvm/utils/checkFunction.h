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

#include <tvm/api.h>
#include <tvm/defs.h>

namespace tvm
{

namespace utils
{
  /** A small structure to specify options for the checks in \ref checkGroup.
    *  - \a step is the increment that will be taken for finite difference schemes
    *  - \a prec is the precision with which the equality of two vectors is tested.
    *    It corresponds to the \a prec parameter of Eigen's \a isApprox method.
    *  - if \a verbose is true, the functions will display some indications when a
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

  /** \defgroup checkGroup */
  /**@{*/
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

  /** Check the jacobian matrics, velocity and normal acceleration of the
    * function \a f
    */
  bool TVM_DLLAPI checkFunction(FunctionPtr f, CheckOptions opt = CheckOptions());
  /**@}*/
}

}