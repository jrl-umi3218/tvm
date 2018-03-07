#pragma once

/* Copyright 2017-2018 CNRS-UM LIRMM, CNRS-AIST JRL
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
    * A = | P1  P2 | = | Q1  Q2 | | R1  R2 |
    *                             |  0   0 |
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
      virtual void postMultiplyByN_(MatrixRef out, const MatrixConstRef& in, bool add) const override;

    private:
      Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_;
      Eigen::MatrixXd invR1R2_;                         //inv(R1)*R2
      mutable utils::internal::BufferedMatrix tmp_;     //temporary for the premultiplication by Asharp and S^T
    };

  protected:
    std::unique_ptr<abstract::SubstitutionCalculatorImpl> impl_(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank) const;

  };

} // internal

} // hint

} // tvm