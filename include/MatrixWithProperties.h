#pragma once

#include <Eigen/Core>

#include "MatrixProperties.h"

namespace tvm
{
  class MatrixWithProperties : public Eigen::MatrixXd
  {
  public:
    using Eigen::MatrixXd::MatrixXd;
    template<typename OtherDerived>
    MatrixWithProperties& operator=(const Eigen::MatrixBase<OtherDerived>& other)
    {
      assert(this->rows() == other.rows() && this->cols() == other.cols() 
              && "It is not allowed to assign an expression with a different size. Please explicitely resize the matrix before.")
      this->Eigen::MatrixXd::operator=(other);
      properties_ = MatrixProperties();
      return *this;
    }
    const MatrixProperties& properties() const { return properties_; }
    void properties(MatrixProperties p) { properties_ = p; }

  private:
    MatrixProperties properties_;
  };
}

