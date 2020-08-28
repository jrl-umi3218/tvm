/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/LinearizedControlProblem.h>
#include <tvm/scheme/internal/ProblemComputationData.h>

namespace tvm::scheme::internal
{
  /** An extension of ProblemComputationData containing data specific to
    * linearized problems.
    */
  class LinearizedProblemComputationData : public ProblemComputationData
  {
  public:
    /** Give the copy to the computation data of a pair (TaskWithRequirements*, 
      * LinearConstraintWithRequirements) being deleted from the problem.
      *
      * \note THis is needed because the removal in the computation data is
      * delayed compared to the removal from the problem, but we still need data
      * related to the constraint being removed, so that we need to make sure its
      * life is extended.
      * It is the scheme implementation responsability to clear the list of
      * removed constraints with \c clearRemovedConstraints. Failing to do so
      * will not cause error, but can induce higher memory usage and slightly
      * slower operations in future removal processes.
      */
    void transferRemovedConstraint(std::pair<TaskWithRequirements*, LinearConstraintWithRequirements> c)
    {
      removedConstraints_.emplace_back(std::move(c));
    }

    /** Access the removed constraint corresponding to \p tr.*/
    const LinearConstraintWithRequirements& removedConstraint(TaskWithRequirements* tr)
    {
      for (const auto& c : removedConstraints_)
      {
        if (c.first == tr) return c.second;
      }
      throw std::runtime_error("[LinearConstraintWithRequirements::removedConstraint] invalid task pointer.");
    }

    /** Remove all the registered pait (TaskWithRequirements*,  LinearConstraintWithRequirements)
      *
      * To be called after the scheme has processed all the removed constraints.
      */
    void clearRemovedConstraints() { removedConstraints_.clear(); }

  protected:
    LinearizedProblemComputationData(int solverId) : ProblemComputationData(solverId) {}

  private:
    std::vector<std::pair<TaskWithRequirements*, LinearConstraintWithRequirements>> removedConstraints_;
  };
}