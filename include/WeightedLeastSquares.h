#pragma once

#include "Assignment.h"
#include "ResolutionScheme.h"

namespace tvm
{
  namespace scheme
  {
    /** This class implements the classic weighted least square scheme
      */
    class TVM_DLLAPI WeightedLeastSquares : public LinearResolutionScheme
    {
    public:
      WeightedLeastSquares(std::shared_ptr<LinearizedControlProblem> pb, double scalarizationWeight = 1000);

    protected:
      void solve_();

    private:
      struct Memory
      {
        Memory(int n, int m0, int m1, double big_number);

        Eigen::MatrixXd A;
        Eigen::MatrixXd C;
        Eigen::VectorXd b;
        Eigen::VectorXd l;
        Eigen::VectorXd u;
      };

      void build();

      double scalarizationWeight_;
      std::vector<Assignment> assignments_;
      std::shared_ptr<Memory> memory_;
    };
  }
}