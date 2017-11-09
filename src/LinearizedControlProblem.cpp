#include "LinearizedControlProblem.h"
#include "LinearizedTaskConstraint.h"

namespace tvm
{
  LinearizedControlProblem::LinearizedControlProblem()
  {
  }

  LinearizedControlProblem::LinearizedControlProblem(const ControlProblem& pb)
  {
    for (auto tr : pb.tasks())
      add(tr);
  }

  TaskWithRequirementsPtr LinearizedControlProblem::add(const Task& task, const SolvingRequirements& req)
  {
    auto tr = std::make_shared<TaskWithRequirements>(task, req);
    add(tr);
    return tr;
  }

  TaskWithRequirementsPtr LinearizedControlProblem::add(ProtoTask proto, std::shared_ptr<TaskDynamics> td, const SolvingRequirements& req)
  {
    return add({ proto, td }, req);
  }

  void LinearizedControlProblem::add(TaskWithRequirementsPtr tr)
  {
    ControlProblem::add(tr);

    //FIXME A lot of work can be done here based on the properties of the task's jacobian.
    //In particular, we could detect bounds, pairs of tasks forming a double-sided constraints...
    LinearConstraintWithRequirements lcr;
    lcr.constraint = std::make_shared<LinearizedTaskConstraint>(tr->task);
    //we use the aliasing constructor of std::shared_ptr to ensure that
    //lcr.requirements points to and doesn't outlive tr->requirements.
    lcr.requirements = std::shared_ptr<SolvingRequirements>(tr, &tr->requirements);
    constraints_[tr.get()] = lcr;
    typedef internal::FirstOrderProvider::Output CstrOutput;
    updater_.addInput(lcr.constraint, CstrOutput::Jacobian);
    switch (tr->task.type())
    {
    case ConstraintType::EQUAL: updater_.addInput(lcr.constraint, Constraint::Output::E); break;
    case ConstraintType::GREATER_THAN: updater_.addInput(lcr.constraint, Constraint::Output::L); break;
    case ConstraintType::LOWER_THAN: updater_.addInput(lcr.constraint, Constraint::Output::U); break;
    case ConstraintType::DOUBLE_SIDED: updater_.addInput(lcr.constraint, Constraint::Output::L, Constraint::Output::U); break;
    }
  }

  void LinearizedControlProblem::remove(TaskWithRequirements* tr)
  {
    ControlProblem::remove(tr);
    // if the above line did not throw, tr exists in the problem and in constraints_
    updater_.removeInput(constraints_[tr].constraint.get());
    constraints_.erase(tr);
  }

  std::vector<LinearConstraintWithRequirements> LinearizedControlProblem::constraints() const
  {
    std::vector<LinearConstraintWithRequirements> constraints;
    constraints.reserve(constraints_.size());
    for (auto c: constraints_)
      constraints.push_back(c.second);

    return constraints;
  }

  void LinearizedControlProblem::update()
  {
    updater_.refresh();
    updater_.run();
  }

  LinearizedControlProblem::Updater::Updater()
    : upToDate_(false)
    , inputs_(new data::Inputs)
  {
  }

  void LinearizedControlProblem::Updater::refresh()
  {
    if (!upToDate_)
    {
      updateGraph_.clear();
      updateGraph_.add(inputs_);
      updateGraph_.update();
      upToDate_ = true;
    }
  }

  void LinearizedControlProblem::Updater::run()
  {
    updateGraph_.execute();
  }
}
