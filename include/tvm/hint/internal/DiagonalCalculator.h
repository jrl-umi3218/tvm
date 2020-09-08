/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/hint/abstract/SubstitutionCalculator.h>
#include <tvm/hint/abstract/SubstitutionCalculatorImpl.h>

namespace tvm
{

namespace hint
{

namespace internal
{
/** A calculator for all matrices that are the row selection of a diagonal
 * matrix.
 * The structure of the matrix (position of non-zero elements) is assumed to
 * be constant.
 */
class TVM_DLLAPI DiagonalCalculator : public abstract::SubstitutionCalculator
{
public:
  class TVM_DLLAPI Impl : public abstract::SubstitutionCalculatorImpl
  {
  public:
    Impl(const std::vector<LinearConstraintPtr> & cstr,
         const std::vector<VariablePtr> & x,
         int rank,
         Eigen::DenseIndex first,
         Eigen::DenseIndex size);
    Impl(const std::vector<LinearConstraintPtr> & cstr,
         const std::vector<VariablePtr> & x,
         int rank,
         const std::vector<Eigen::DenseIndex> & nnzRows,
         const std::vector<Eigen::DenseIndex> & zeroRows);

    virtual void update_() override;
    virtual void premultiplyByASharpAndSTranspose_(MatrixRef outA,
                                                   MatrixRef outS,
                                                   const MatrixConstRef & in,
                                                   bool minus) const override;
    virtual void postMultiplyByN_(MatrixRef out, const MatrixConstRef & in, bool add) const override;

  private:
    void build();

    Eigen::DenseIndex first_;
    Eigen::DenseIndex size_;
    std::vector<Eigen::DenseIndex> nnz_;   // indices of the non-zero rows
    std::vector<Eigen::DenseIndex> innz_;  // indices of the non-zero rows in the resulting matrix
    std::vector<Eigen::DenseIndex> cnnz_;  // complementary to nnz in [0:col-1]
    std::vector<Eigen::DenseIndex> zeros_; // indices of the zero rows

    Eigen::VectorXd inverse_;
  };

  /** Continuous rows of a diagonal matrix starting at row \p first of this
   * matrix and finishing at row \p first + \p size (not included).
   * If \p size = -1, finishes at the last row.
   */
  DiagonalCalculator(Eigen::DenseIndex first = 0, Eigen::DenseIndex size = -1);
  /** Non-continuous rows of a diagonal matrix
   * \param nnzRows indices of the rows with a non-zero element.
   * \param zeroRows indices of the rows (if any) with only zero
   *
   * Examples:
   * Knowing that it describes a matrix with 6 columns, nnzRows = {0,2,3},
   * zeroRows = {} corresponds to the shape
   * x 0 0 0 0 0
   * 0 0 x 0 0 0
   * 0 0 0 x 0 0
   *
   * while nnzRows = {0,2,3}, zeroRows = {1,4} corresponds to
   * x 0 0 0 0 0
   * 0 0 0 0 0 0
   * 0 0 x 0 0 0
   * 0 0 0 x 0 0
   * 0 0 0 0 0 0
   */
  DiagonalCalculator(const std::vector<Eigen::DenseIndex> & nnzRows,
                     const std::vector<Eigen::DenseIndex> & zeroRows = {});

protected:
  std::unique_ptr<abstract::SubstitutionCalculatorImpl> impl_(const std::vector<LinearConstraintPtr> & cstr,
                                                              const std::vector<VariablePtr> & x,
                                                              int rank) const;

private:
  Eigen::DenseIndex first_;
  Eigen::DenseIndex size_;
  std::vector<Eigen::DenseIndex> nnz_;   // indices of the non-zero rows
  std::vector<Eigen::DenseIndex> zeros_; // indices of the zero rows
};
} // namespace internal

} // namespace hint

} // namespace tvm
