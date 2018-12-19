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
#include <tvm/hint/abstract/SubstitutionCalculator.h>
#include <tvm/hint/abstract/SubstitutionCalculatorImpl.h>
#include <tvm/utils/internal/BufferedMatrix.h>

#include <Eigen/QR>

#include <vector>

namespace tvm
{

namespace hint
{

namespace internal
{
  /** The default substitution calculator for a set of constraints.
    * A^#, N and S are deduced from a single rank-revealing QR:
    * A | P1  P2 | = | Q1  Q2 | | R1  R2 |
    *                           |  0   0 |
    * A^# = P1 R1^-1 Q1^T
    * N = P2 - P1 R1^-1 R2
    * S = Q2
    */
  class TVM_DLLAPI GenericCalculator: public abstract::SubstitutionCalculator
  {
  public:
    class TVM_DLLAPI Impl : public abstract::SubstitutionCalculatorImpl
    {
    public:
      Impl(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank);

      virtual void update_() override;
      virtual void premultiplyByASharpAndSTranspose_(MatrixRef outA, MatrixRef outS, const MatrixConstRef& in, bool minus) const override;

    private:
      Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_;
      Eigen::MatrixXd invR1R2_;                         //inv(R1)*R2
      mutable utils::internal::BufferedMatrix tmp_;     //temporary for the premultiplication by Asharp and S^T
    };

  protected:
    std::unique_ptr<abstract::SubstitutionCalculatorImpl> impl_(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank) const override;

  };

} // internal

} // hint

} // tvm