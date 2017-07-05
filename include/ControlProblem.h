#pragma once

#include <vector>

#include "DataGraph.h"
#include "Time.h"


namespace tvm
{
  class Function;

  class ControlProblem
  {
  public:
    void update();

  protected:
    void addFunction(std::shared_ptr<Function> f);
    void removeFunction(Function* f);

    void build();

  private:


  private:
    double dt_;

    UpdateGraph graph_;
    UpdatePlan  updates_;
    std::shared_ptr<Clock> clock_;
    std::vector<std::shared_ptr<ExplicitlyTimeDependent>> timeDependents_;
  };
}