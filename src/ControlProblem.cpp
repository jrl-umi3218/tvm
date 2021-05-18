/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/ControlProblem.h>

namespace tvm
{

TaskWithRequirements::TaskWithRequirements(const Task & t, requirements::SolvingRequirements req)
: task(t), requirements(req)
{}

TaskWithRequirementsPtr ControlProblem::add(const Task & task, const requirements::SolvingRequirements & req)
{
  auto tr = std::make_shared<TaskWithRequirements>(task, req);
  add(tr);
  return tr;
}

void ControlProblem::add(TaskWithRequirementsPtr tr)
{
  tr_.push_back(tr);
  addCallBackToTask(tr);
  notify(scheme::internal::ProblemDefinitionEvent(scheme::internal::ProblemDefinitionEvent::Type::TaskAddition, *tr));
  finalized_ = false;
}

void ControlProblem::remove(const TaskWithRequirements & tr)
{
  auto it = std::find_if(tr_.begin(), tr_.end(), [&tr](const TaskWithRequirementsPtr & p) { return p.get() == &tr; });
  if(it == tr_.end())
  {
    return;
  }
  tr_.erase(it);
  notify(scheme::internal::ProblemDefinitionEvent(scheme::internal::ProblemDefinitionEvent::Type::TaskRemoval, tr));
  callbackTokens_.erase(&tr);
  finalized_ = false;
}

const std::vector<TaskWithRequirementsPtr> & ControlProblem::tasks() const { return tr_; }

int ControlProblem::size() const { return static_cast<int>(tr_.size()); }

void ControlProblem::update()
{
  finalize();
  updater_.run();
  update_();
}

void ControlProblem::finalize()
{
  if(!finalized_)
  {
    updater_.refresh();
    finalize_();
    finalized_ = true;
  }
}

const graph::CallGraph & ControlProblem::updateGraph() const { return updater_.updateGraph(); }

void ControlProblem::needFinalize() { finalized_ = false; }

void ControlProblem::notify(const scheme::internal::ProblemDefinitionEvent & e)
{
  for(auto & c : computationData_)
  {
    c.second->addEvent(e);
  }
}

void ControlProblem::addCallBackToTask(TaskWithRequirementsPtr tr)
{
  using EventType = scheme::internal::ProblemDefinitionEvent::Type;
  std::vector<internal::PairElementToken> tokens;
  TaskWithRequirements * t = tr.get();

  auto l1 = [this, t]() { this->notify({EventType::WeightChange, *t}); };
  tokens.emplace_back(tr->requirements.weight().registerCallback(l1));

  auto l2 = [this, t]() { this->notify({EventType::AnisotropicWeightChange, *t}); };
  tokens.emplace_back(tr->requirements.anisotropicWeight().registerCallback(l2));

  callbackTokens_[t] = std::move(tokens);
}

ControlProblem::Updater::Updater() : upToDate_(false), inputs_(new graph::internal::Inputs) {}

void ControlProblem::Updater::refresh()
{
  if(!upToDate_)
  {
    updateGraph_.clear();
    updateGraph_.add(inputs_);
    updateGraph_.update();
    upToDate_ = true;
  }
}

void ControlProblem::Updater::run() { updateGraph_.execute(); }

const graph::CallGraph & ControlProblem::Updater::updateGraph() const { return updateGraph_; }

} // namespace tvm
