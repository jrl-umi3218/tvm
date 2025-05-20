/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/Task.h>
#include <tvm/graph/CallGraph.h>
#include <tvm/requirements/SolvingRequirements.h>
#include <tvm/scheme/abstract/ResolutionScheme.h>
#include <tvm/scheme/internal/ProblemComputationData.h>
#include <tvm/scheme/internal/ProblemDefinitionEvent.h>
#include <tvm/scheme/internal/ResolutionSchemeBase.h>
#include <tvm/scheme/internal/helpers.h>
#include <tvm/task_dynamics/None.h>
#include <tvm/task_dynamics/abstract/TaskDynamics.h>
#include <tvm/utils/ProtoTask.h>

#include <memory>
#include <vector>

namespace tvm
{
class TVM_DLLAPI TaskWithRequirements : public tvm::internal::ObjWithId
{
public:
  TaskWithRequirements(const Task & task, requirements::SolvingRequirements req);

  Task task;
  requirements::SolvingRequirementsWithCallbacks requirements;
};

using TaskWithRequirementsPtr = std::shared_ptr<TaskWithRequirements>;

class TVM_DLLAPI ControlProblem
{
  friend class LinearizedControlProblem;
  // for addEventsToComputationData
  template<typename U>
  friend class scheme::abstract::ResolutionScheme;

public:
  ControlProblem() = default;
  /** \internal We delete these functions because they would require by
   * default a copy of the unique_ptr in the computationData_ map.
   * There is no problem in implementing them as long as there is a real
   * deep copy of the ProblemComputationData. If making this copy, care must
   * be taken that the objects pointed to by the unique_ptr are instances of
   * classes derived from ProblemComputationData.
   */
  ControlProblem(const ControlProblem &) = delete;
  ControlProblem & operator=(const ControlProblem &) = delete;

  TaskWithRequirementsPtr add(const Task & task, const requirements::SolvingRequirements & req = {});
  template<constraint::Type T>
  TaskWithRequirementsPtr add(utils::ProtoTask<T> proto,
                              const task_dynamics::abstract::TaskDynamics & td,
                              const requirements::SolvingRequirements & req = {});
  template<constraint::Type T>
  TaskWithRequirementsPtr add(utils::LinearProtoTask<T> proto,
                              const task_dynamics::abstract::TaskDynamics & td,
                              const requirements::SolvingRequirements & req = {});
  template<constraint::Type T>
  TaskWithRequirementsPtr add(utils::LinearProtoTask<T> proto, const requirements::SolvingRequirements & req = {});
  void add(TaskWithRequirementsPtr tr, bool notify = true);
  void remove(const TaskWithRequirements & tr, bool notify = true);
  const std::vector<TaskWithRequirementsPtr> & tasks() const;

  /** Number of tasks in the problem.*/
  int size() const;

  /** Compute all quantities necessary for solving the problem.*/
  void update();

  /** Finalize the data of the solver*/
  void finalize();

  /** Access to update graph, for diagnostic purpose.*/
  const graph::CallGraph & updateGraph() const;

protected:
  class Updater
  {
  public:
    Updater();

    template<typename T, typename... Args>
    void addInput(std::shared_ptr<T> source, Args... args);
    template<typename T>
    void removeInput(T * source);

    /** Manually calls for an update of the call graph, if needed.*/
    void refresh();
    /** Execute the call graph.*/
    void run();

    const graph::CallGraph & updateGraph() const;

  private:
    bool upToDate_;
    std::shared_ptr<graph::internal::Inputs> inputs_;
    graph::CallGraph updateGraph_;
  };

  virtual void update_() {}
  virtual void finalize_() {}

  void needFinalize();
  /**
   * Request for an event to be handled by the resolution scheme
   *
   * \note The actual addition of events to the problem computation data is delayed until
   * \ref addEventsToComputationData is called.
   */
  void notify(const scheme::internal::ProblemDefinitionEvent & e);

  Updater updater_;

private:
  void addCallBackToTask(TaskWithRequirementsPtr tr);
  /**
   * Add all events added by \ref notify to the resolution scheme's problem computation data
   */
  void addEventsToComputationData();

  // Note: we want to keep the tasks in the order they were introduced, mostly
  // for human understanding and debugging purposes, so that we take a
  // std::vector.
  // If this induces too much overhead when adding/removing a constraint, then
  // we should consider std::set.
  std::vector<TaskWithRequirementsPtr> tr_;

  // Tokens to identify and keep callbacks alive
  tvm::utils::internal::map<TaskWithRequirements const *, std::vector<internal::PairElementToken>> callbackTokens_;
  std::queue<scheme::internal::ProblemDefinitionEvent> eventsToProcess_;

  // Computation data for the resolution schemes
  std::map<scheme::identifier, std::unique_ptr<scheme::internal::ProblemComputationData>> computationData_;

  bool finalized_ = false;

  template<typename Problem, typename Scheme>
  friend scheme::internal::ProblemComputationData * scheme::internal::getComputationData(
      Problem & problem,
      const Scheme & resolutionScheme);
};

template<constraint::Type T>
TaskWithRequirementsPtr ControlProblem::add(utils::ProtoTask<T> proto,
                                            const task_dynamics::abstract::TaskDynamics & td,
                                            const requirements::SolvingRequirements & req)
{
  return add({proto, td}, req);
}

template<constraint::Type T>
TaskWithRequirementsPtr ControlProblem::add(utils::LinearProtoTask<T> proto,
                                            const task_dynamics::abstract::TaskDynamics & td,
                                            const requirements::SolvingRequirements & req)
{
  return add({proto, td}, req);
}

template<constraint::Type T>
TaskWithRequirementsPtr ControlProblem::add(utils::LinearProtoTask<T> proto,
                                            const requirements::SolvingRequirements & req)
{
  return add({proto, task_dynamics::None()}, req);
}

template<typename T, typename... Args>
inline void ControlProblem::Updater::addInput(std::shared_ptr<T> source, Args... args)
{
  inputs_->addInput(source, args...);
  upToDate_ = false;
}

template<typename T>
inline void ControlProblem::Updater::removeInput(T * source)
{
  inputs_->removeInput(source);
  upToDate_ = false;
}
} // namespace tvm
