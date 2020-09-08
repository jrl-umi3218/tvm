/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

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
class TVM_DLLAPI GenericCalculator : public abstract::SubstitutionCalculator
{
public:
  class TVM_DLLAPI Impl : public abstract::SubstitutionCalculatorImpl
  {
  public:
    Impl(const std::vector<LinearConstraintPtr> & cstr, const std::vector<VariablePtr> & x, int rank);

    virtual void update_() override;
    virtual void premultiplyByASharpAndSTranspose_(MatrixRef outA,
                                                   MatrixRef outS,
                                                   const MatrixConstRef & in,
                                                   bool minus) const override;

  private:
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_;
    Eigen::MatrixXd invR1R2_;                     // inv(R1)*R2
    mutable utils::internal::BufferedMatrix tmp_; // temporary for the premultiplication by Asharp and S^T
  };

protected:
  std::unique_ptr<abstract::SubstitutionCalculatorImpl> impl_(const std::vector<LinearConstraintPtr> & cstr,
                                                              const std::vector<VariablePtr> & x,
                                                              int rank) const override;
};

} // namespace internal

} // namespace hint

} // namespace tvm
