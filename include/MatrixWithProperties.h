#pragma once

#include <Eigen/Core>

#include "MatrixProperties.h"

namespace tvm
{
  class MatrixWithProperties : public Eigen::MatrixXd
  {
  public:
    using Eigen::MatrixXd::MatrixXd;
    const MatrixProperties& properties() const { return properties_; }
    void properties(MatrixProperties p) { properties_ = p; }

  private:
    MatrixProperties properties_;
  };
}

