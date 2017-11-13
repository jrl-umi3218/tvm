#pragma once

/* Copyright 2017 CNRS-UM LIRMM, CNRS-AIST JRL
 *
 * This file is part of TVM.
 *
 * TVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with TVM.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <tvm/defs.h>
#include <tvm/ControlProblem.h>
#include <tvm/graph/CallGraph.h>

namespace tvm
{
  class TVM_DLLAPI LinearConstraintWithRequirements
  {
  public:
    LinearConstraintPtr constraint;
    SolvingRequirementsPtr requirements;
  };

  class TVM_DLLAPI LinearizedControlProblem : public ControlProblem
  {
  public:
    LinearizedControlProblem();
    LinearizedControlProblem(const ControlProblem& pb);

    TaskWithRequirementsPtr add(const Task& task, const requirements::SolvingRequirements& req = {});
    TaskWithRequirementsPtr add(ProtoTask proto, TaskDynamicsPtr td, const requirements::SolvingRequirements& req = {});
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
      std::shared_ptr<graph::internal::Inputs> inputs_;
      graph::CallGraph updateGraph_;
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
}  // namespace tvm
