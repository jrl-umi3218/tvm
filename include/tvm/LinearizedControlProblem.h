/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <tvm/defs.h>
#include <tvm/ControlProblem.h>
#include <tvm/graph/CallGraph.h>
#include <tvm/hint/Substitution.h>
#include <tvm/hint/internal/Substitutions.h>

namespace tvm
{
  class TVM_DLLAPI LinearConstraintWithRequirements
  {
  public:
    LinearConstraintPtr constraint;
    SolvingRequirementsPtr requirements;
    bool bound;
  };

  class TVM_DLLAPI LinearizedControlProblem : public ControlProblem
  {
  public:
    LinearizedControlProblem();
    LinearizedControlProblem(const ControlProblem& pb);

    TaskWithRequirementsPtr add(const Task& task, const requirements::SolvingRequirements& req = {});
    template<constraint::Type T>
    TaskWithRequirementsPtr add(utils::ProtoTask<T> proto, const task_dynamics::abstract::TaskDynamics& td, const requirements::SolvingRequirements& req = {});
    template<constraint::Type T>
    TaskWithRequirementsPtr add(utils::LinearProtoTask<T> proto, const task_dynamics::abstract::TaskDynamics& td, const requirements::SolvingRequirements& req = {});
    template<constraint::Type T>
    TaskWithRequirementsPtr add(utils::LinearProtoTask<T> proto, const requirements::SolvingRequirements& req = {});
    void add(TaskWithRequirementsPtr tr);
    void remove(TaskWithRequirements* tr);

    void add(const hint::Substitution& s);
    const hint::internal::Substitutions& substitutions() const;

    /** Access to constraints*/
    std::vector<LinearConstraintWithRequirements> constraints() const;

    LinearConstraintPtr constraint(TaskWithRequirements* t) const;

    /** Compute all quantities necessary for solving the problem.*/
    void update();

    /** Finalize the data of the solver*/
    void finalize();

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

    bool finalized_;
    utils::internal::map<TaskWithRequirements*, LinearConstraintWithRequirements> constraints_;
    Updater updater_;
    hint::internal::Substitutions substitutions_;
  };


  template<constraint::Type T>
  TaskWithRequirementsPtr LinearizedControlProblem::add(utils::ProtoTask<T> proto, const task_dynamics::abstract::TaskDynamics& td, const requirements::SolvingRequirements& req)
  {
    return add({ proto, td }, req);
  }

  template<constraint::Type T>
  TaskWithRequirementsPtr LinearizedControlProblem::add(utils::LinearProtoTask<T> proto, const task_dynamics::abstract::TaskDynamics& td, const requirements::SolvingRequirements& req)
  {
    return add({ proto, td }, req);
  }

  template<constraint::Type T>
  TaskWithRequirementsPtr LinearizedControlProblem::add(utils::LinearProtoTask<T> proto, const requirements::SolvingRequirements& req)
  {
    return add({ proto, task_dynamics::None() }, req);
  }

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
