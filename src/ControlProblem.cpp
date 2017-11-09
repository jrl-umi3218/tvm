#include "ControlProblem.h"

namespace tvm
{
  TaskWithRequirements::TaskWithRequirements(const Task& t, SolvingRequirements req)
    : task(t)
    , requirements(req)
  {
  }


  TaskWithRequirementsPtr ControlProblem::add(const Task& task, const SolvingRequirements& req)
  {
    auto tr = std::make_shared<TaskWithRequirements>(task, req);
    add(tr);
    return tr;
  }

  TaskWithRequirementsPtr ControlProblem::add(ProtoTask proto, std::shared_ptr<TaskDynamics> td, const SolvingRequirements& req)
  {
    return add({ proto,td }, req);
  }

  void ControlProblem::add(TaskWithRequirementsPtr tr)
  {
    tr_.push_back(tr);
  }

  void ControlProblem::remove(TaskWithRequirements* tr)
  {
    auto it = std::find_if(tr_.begin(), tr_.end(), [tr](const TaskWithRequirementsPtr& p) {return p.get() == tr; });
    if (it != tr_.end())
      tr_.erase(it);
  }

  const std::vector<TaskWithRequirementsPtr>& ControlProblem::tasks() const
  {
    return tr_;
  }
}