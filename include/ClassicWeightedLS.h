#pragma once

#include "Assignment.h"
#include "ResolutionScheme.h"

namespace tvm
{
  namespace scheme
  {
    class TVM_DLLAPI ClassicWeightedLS : public LinearResolutionScheme
    {
    public:
      ClassicWeightedLS(std::shared_ptr<LinearizedControlProblem> pb, double scalarizationWeight = 1000);

    protected:
      void solve_();

    private:
      struct Memory
      {
        Memory(int n, int m0, int m1);

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