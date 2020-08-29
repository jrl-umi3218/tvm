/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/constraint/abstract/Constraint.h>
#include <tvm/requirements/ViolationEvaluation.h>

#include <iostream>
#include <map>
#include <vector>

namespace tvm
{

namespace scheme
{

namespace internal
{

  inline constexpr int GeneralLevel = -1;
  inline constexpr int NoLimit = GeneralLevel;

  class TVM_DLLAPI LevelAbilities
  {
  public:
    LevelAbilities(bool inequality, const std::vector<requirements::ViolationEvaluationType>& types);

    template<class RequirementsPtr>
    void check(const ConstraintPtr& c, const RequirementsPtr& req, bool emitWarnings = true) const;

  private:
    bool inequalities_;
    std::vector<requirements::ViolationEvaluationType> evaluationTypes_;
  };


  /** A class to describe what type of constraint and solving requirements a
    * resolution scheme can handle
    */
  class TVM_DLLAPI SchemeAbilities
  {
  public:
    SchemeAbilities(int numberOfLevels, const std::map<int, LevelAbilities>& abilities, bool scalarization=false);

    template<class RequirementsPtr>
    void check(const ConstraintPtr& c, const RequirementsPtr& req, bool emitWarnings = true) const;

  private:
    int numberOfLevels_;
    bool scalarization_;
    std::map<int, LevelAbilities> abilities_;
  };


  template<class RequirementsPtr>
  inline void LevelAbilities::check(const ConstraintPtr& c, const RequirementsPtr& req, bool /*emitWarnings*/) const
  {
    //checking the constraint type
    if (c->type() != constraint::Type::EQUAL && !inequalities_)
      throw std::runtime_error("This level does not handle inequality constraints.");

    //checking the evaluation type
    auto it = std::find(evaluationTypes_.begin(), evaluationTypes_.end(), req->violationEvaluation().value());
    if (it == evaluationTypes_.end())
      throw std::runtime_error("This level does not handle the required violation evaluation value.");
  }


  template<class RequirementsPtr>
  inline void SchemeAbilities::check(const ConstraintPtr& c, const RequirementsPtr& req, bool emitWarnings) const
  {
    //check priority level value
    int p = req->priorityLevel().value();
    if (numberOfLevels_ > 0 && p >= numberOfLevels_)
    {
      if (scalarization_)
      {
        if (emitWarnings)
        {
          std::cout << "Warning: required priority level (" << p
            << ") cannot be handled as a strict hierarchy level by the resolution scheme. "
            << "Using scalarization to revert to weighted approach.";
        }
      }
      else
      {
        std::stringstream s;
        s << "This resolution scheme can handle priorities level up to" << numberOfLevels_ - 1
          << ". It cannot process required level (" << p << ")." << std::endl;
        throw std::runtime_error(s.str());
      }
    }

    //check abilities of the required priority level
    auto it = abilities_.find(p);
    if (it != abilities_.end())
      it->second.check(c, req, emitWarnings);
    else
      abilities_.at(GeneralLevel).check(c, req, emitWarnings);
  }

}  // namespace internal

}  // namespace scheme

}  // namespace tvm
