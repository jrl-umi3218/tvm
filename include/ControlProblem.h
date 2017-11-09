#pragma once

#include <memory>
#include <vector>

#include "SolvingRequirements.h"
#include "Task.h"
#include "TaskDynamics.h"
#include "Clock.h"


namespace tvm
{
  class TVM_DLLAPI TaskWithRequirements
  {
  public:
    TaskWithRequirements(const Task& task, SolvingRequirements req);

    Task task;
    SolvingRequirements requirements;
  };

  typedef std::shared_ptr<TaskWithRequirements> TaskWithRequirementsPtr;

  class TVM_DLLAPI ControlProblem
  {
  public:
    TaskWithRequirementsPtr add(const Task& task, const SolvingRequirements& req = {});
    TaskWithRequirementsPtr add(ProtoTask proto, std::shared_ptr<TaskDynamics> td, const SolvingRequirements& req = {});
    void add(TaskWithRequirementsPtr tr);
    void remove(TaskWithRequirements* tr);
    const std::vector<TaskWithRequirementsPtr>& tasks() const;

  private:
    //Note: we want to keep the tasks in the order they were introduced, mostly
    //for human understanding and debugging purposes, so that we take a
    //std::vector.
    //If this induce too much overhead when adding/removing a constraint, then
    //we should consider std::set.
    std::vector<TaskWithRequirementsPtr> tr_;
  };
}