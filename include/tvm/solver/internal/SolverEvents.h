/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>

#include <vector>

namespace tvm::solver::internal
{
  /** An agregation of all the events to be handled by a solver*/
  class SolverEvents
  {
  public:
    struct WeightEvent
    {
      constraint::abstract::LinearConstraint* c;
      bool scalar;
      bool vector;
    };

    struct Objective
    {
      LinearConstraintPtr c;
      SolvingRequirementsPtr req;
      double scalarizationWeight;
    };


    void addScalarWeigthEvent(constraint::abstract::LinearConstraint* c);
    void addVectorWeigthEvent(constraint::abstract::LinearConstraint* c);

    void addConstraint(LinearConstraintPtr c);
    void removeConstraint(LinearConstraintPtr c);
    void addBound(LinearConstraintPtr c);
    void removeBound(LinearConstraintPtr c);
    void addObjective(const Objective& o);
    void removeObjective(LinearConstraintPtr o);

    void addVariable(VariablePtr v);
    void removeVariable(VariablePtr v);

    const std::vector<WeightEvent>& weightEvents() const { return weightEvents_; }
    const std::vector<LinearConstraintPtr>& addedConstraints() const { return addedConstraints_; }
    const std::vector<LinearConstraintPtr>& addedBounds() const { return addedBounds_; }
    const std::vector<Objective>& addedObjectives() const { return addedObjectives_; }
    const std::vector<LinearConstraintPtr>& removedConstraints() const { return removedConstraints_; }
    const std::vector<LinearConstraintPtr>& removedBounds() const { return removedBounds_; }
    const std::vector<LinearConstraintPtr>& removedObjectives() const { return removedObjectives_; }

    const std::vector<VariablePtr>& addedVariables() const { return addedVariables_; }
    const std::vector<VariablePtr>& removedVariables() const { return removedVariables_; }

  private:
    /** If c is in removeVec, remove it, otherwise, add it to addVec*/
    template<typename T>
    void addIfPair(T& c, std::vector<T>& addVec, std::vector<T>& removeVec);

    /** \internal We don't anticipate to have many events at the same time so 
      * that searching in the vector will be fast. If it was not the case, we can
      * change the data structure, or add one to speed up search.*/
    std::vector<WeightEvent> weightEvents_;

    std::vector<LinearConstraintPtr> addedConstraints_;
    std::vector<LinearConstraintPtr> addedBounds_;
    std::vector<Objective> addedObjectives_;
    std::vector<LinearConstraintPtr> removedConstraints_;
    std::vector<LinearConstraintPtr> removedBounds_;
    std::vector<LinearConstraintPtr> removedObjectives_;

    std::vector<VariablePtr> addedVariables_;
    std::vector<VariablePtr> removedVariables_;
  };

  inline void SolverEvents::addScalarWeigthEvent(constraint::abstract::LinearConstraint* c)
  {
    for (auto& e : weightEvents_)
    {
      if (e.c == c)
      {
        e.scalar = true;
        return;
      }
    }

    weightEvents_.push_back({ c, true, false });
  }

  inline void SolverEvents::addVectorWeigthEvent(constraint::abstract::LinearConstraint* c)
  {
    for (auto& e : weightEvents_)
    {
      if (e.c == c)
      {
        e.vector = true;
        return;
      }
    }

    weightEvents_.push_back({ c, false, true });
  }

  inline void SolverEvents::addConstraint(LinearConstraintPtr c)
  {
    addIfPair(c, addedConstraints_, removedConstraints_);
  }

  inline void SolverEvents::removeConstraint(LinearConstraintPtr c)
  {
    addIfPair(c, removedConstraints_, addedConstraints_);
  }

  inline void SolverEvents::addBound(LinearConstraintPtr b)
  {
    addIfPair(b, addedBounds_, removedBounds_);
  }

  inline void SolverEvents::removeBound(LinearConstraintPtr b)
  {
    addIfPair(b, removedBounds_, addedBounds_);
  }

  inline void SolverEvents::addObjective(const Objective& o)
  {
    auto it = std::find(removedObjectives_.begin(), removedObjectives_.end(), o.c);
    if (it == removedObjectives_.end())
      addedObjectives_.push_back(o);
    else
      removedObjectives_.erase(it);
  }

  inline void SolverEvents::removeObjective(LinearConstraintPtr o)
  {
    auto it = std::find_if(addedObjectives_.begin(), addedObjectives_.end(), [&o](const auto& it) {return it.c == o; });
    if (it == addedObjectives_.end())
      removedObjectives_.push_back(o);
    else
      addedObjectives_.erase(it);
  }

  inline void SolverEvents::addVariable(VariablePtr v)
  {
    addIfPair(v, addedVariables_, removedVariables_);
  }

  inline void SolverEvents::removeVariable(VariablePtr v)
  {
    addIfPair(v, removedVariables_, addedVariables_);
  }

  template<typename T>
  inline void SolverEvents::addIfPair(T& c, std::vector<T>& addVec, std::vector<T>& removeVec)
  {
    auto it = std::find(removeVec.begin(), removeVec.end(), c);
    if (it == removeVec.end())
      addVec.push_back(c);
    else
      removeVec.erase(it);
  }
}