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

#include <tvm/Task.h>
#include <tvm/requirements/SolvingRequirements.h>
#include <tvm/scheme/internal/helpers.h>
#include <tvm/scheme/internal/ProblemComputationData.h>
#include <tvm/scheme/internal/ResolutionSchemeBase.h>
#include <tvm/task_dynamics/abstract/TaskDynamics.h>
#include <tvm/utils/ProtoTask.h>

#include <memory>
#include <vector>

namespace tvm
{
  class TVM_DLLAPI TaskWithRequirements
  {
  public:
    TaskWithRequirements(const Task& task, requirements::SolvingRequirements req);

    Task task;
    requirements::SolvingRequirements requirements;
  };

  using TaskWithRequirementsPtr = std::shared_ptr<TaskWithRequirements>;

  class TVM_DLLAPI ControlProblem
  {
    friend class LinearizedControlProblem;
  public:
    ControlProblem() = default;
    /** \internal We delete these functions because they would require by
      * default a copy of the unique_ptr in the computationData_ map.
      * There is no problem in implementing them as long as there is a real
      * deep copy of the ProblemComputationData. If making this copy, care must
      * be taken that the objects pointed to by the unique_ptr are instances of
      * classes derived from ProblemComputationData.
      */
    ControlProblem(const ControlProblem&) = delete;
    ControlProblem& operator=(const ControlProblem &) = delete;

    TaskWithRequirementsPtr add(const Task& task, const requirements::SolvingRequirements& req = {});
    template<constraint::Type T>
    TaskWithRequirementsPtr add(utils::ProtoTask<T> proto, const task_dynamics::abstract::TaskDynamics& td, const requirements::SolvingRequirements& req = {});
    void add(TaskWithRequirementsPtr tr);
    void remove(TaskWithRequirements* tr);
    const std::vector<TaskWithRequirementsPtr>& tasks() const;
  private:
    //Note: we want to keep the tasks in the order they were introduced, mostly
    //for human understanding and debugging purposes, so that we take a
    //std::vector.
    //If this induces too much overhead when adding/removing a constraint, then
    //we should consider std::set.
    std::vector<TaskWithRequirementsPtr> tr_;

    //Computation data for the resolution schemes
    std::map<scheme::identifier, std::unique_ptr<scheme::internal::ProblemComputationData>> computationData_;

    template<typename Problem, typename Scheme>
    friend scheme::internal::ProblemComputationData&
      scheme::internal::getComputationData(Problem& problem, const Scheme& resolutionScheme);
  };

  template<constraint::Type T>
  TaskWithRequirementsPtr ControlProblem::add(utils::ProtoTask<T> proto, const task_dynamics::abstract::TaskDynamics& td, const requirements::SolvingRequirements& req)
  {
    return add({ proto,td }, req);
  }
}  // namespace tvm
