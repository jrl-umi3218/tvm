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
      Impl(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank, 
           Eigen::DenseIndex first, Eigen::DenseIndex size);
      Impl(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank, 
           const std::vector<Eigen::DenseIndex>& nnzRows, const std::vector<Eigen::DenseIndex>& zeroRows);

      virtual void update_() override;
      virtual void premultiplyByASharpAndSTranspose_(MatrixRef outA, MatrixRef outS, const MatrixConstRef& in, bool minus) const override;
      virtual void postMultiplyByN_(MatrixRef out, const MatrixConstRef& in, bool add) const override;

    private:
      void build();

      Eigen::DenseIndex first_;
      Eigen::DenseIndex size_;
      std::vector<Eigen::DenseIndex> nnz_;      //indices of the non-zero rows
      std::vector<Eigen::DenseIndex> innz_;     //indices of the non-zero rows in the resulting matrix
      std::vector<Eigen::DenseIndex> cnnz_;     //complementary to nnz in [0:col-1]
      std::vector<Eigen::DenseIndex> zeros_;    //indices of the zero rows

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
    DiagonalCalculator(const std::vector<Eigen::DenseIndex>& nnzRows, const std::vector<Eigen::DenseIndex>& zeroRows = {});

  protected:
    std::unique_ptr<abstract::SubstitutionCalculatorImpl> impl_(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank) const;

  private:
    Eigen::DenseIndex first_;
    Eigen::DenseIndex size_;
    std::vector<Eigen::DenseIndex> nnz_;      //indices of the non-zero rows
    std::vector<Eigen::DenseIndex> zeros_;    //indices of the zero rows

  };
} // internal

} // hint

} // tvm