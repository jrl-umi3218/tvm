#pragma once

#include "defs.h"
#include "ControlProblem.h"
#include "tvm/CallGraph.h"

namespace tvm
{
  class TVM_DLLAPI LinearConstraintWithRequirements
  {
  public:
    LinearConstraintPtr constraint;
    SolvingRequirementsPtr requirements;
  };


  class TVM_DLLAPI  LinearizedControlProblem : public ControlProblem
  {
  public:
    LinearizedControlProblem();
    LinearizedControlProblem(const ControlProblem& pb);

    TaskWithRequirementsPtr add(const Task& task, const SolvingRequirements& req = {});
    TaskWithRequirementsPtr add(ProtoTask proto, std::shared_ptr<TaskDynamics> td, const SolvingRequirements& req = {});
    void add(TaskWithRequirementsPtr tr);
    void remove(TaskWithRequirements* tr);

    /** Access to constraints*/
    std::vector<LinearConstraintWithRequirements> constraints() const;

    /** Compute all quantities necessary for solving the problem.*/
    void update();

  private:
    class Updater
    {
    public:
      Updater();

      template<typename T, typename ... Args>
      void addInput(std::shared_ptr<T> source, Args... args);
      template<typename T>
      void removeInput(T* source);

      /** Manually calls for an update of the call graph, if needed.*/
      void refresh();
      /** Execute the call graph.*/
      void run();

    private:
      bool upToDate_;
      std::shared_ptr<data::Inputs> inputs_;
      CallGraph updateGraph_;
    };

    std::map<TaskWithRequirements*, LinearConstraintWithRequirements> constraints_;
    Updater updater_;
  };

  template<typename T, typename ... Args>
  inline void LinearizedControlProblem::Updater::addInput(std::shared_ptr<T> source, Args... args)
  {
    inputs_->addInput(source, args...);
    upToDate_ = false;
  }

  template<typename T>
  inline void LinearizedControlProblem::Updater::removeInput(T* source)
  {
    inputs_->removeInput(source);
    upToDate_ = false;
  }
}
