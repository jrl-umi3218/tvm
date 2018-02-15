#include <tvm/ControlProblem.h>

namespace tvm
{

  TaskWithRequirements::TaskWithRequirements(const Task& t, requirements::SolvingRequirements req)
    : task(t)
    , requirements(req)
  {
  }


  TaskWithRequirementsPtr ControlProblem::add(const Task& task, const requirements::SolvingRequirements& req)
  {
    auto tr = std::make_shared<TaskWithRequirements>(task, req);
    add(tr);
    return tr;
  }

  ControlProblem::ControlProblem(double dt)
  : clock_(std::make_shared<Clock>(dt))
  {
  }

  ControlProblem::ControlProblem(std::shared_ptr<Clock> clock)
  : clock_(clock)
  {
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

}  // namespace tvm
