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

/** Used to describe any level that is not explicitly specified.*/
inline constexpr int GeneralLevel = -1;
/** Used to specify that a scheme can handle an unlimited number of levels.*/
inline constexpr int NoLimit = GeneralLevel;

/** A class to describe what type of constraint and solving requirements a
 * level of a resolution scheme can handle.
 */
class TVM_DLLAPI LevelAbilities
{
public:
  /** \param inequality True if the level is able to handle inequality
   * constraints.
   * \param types The types of violation evaluation that can be handled at
   * this level.
   */
  LevelAbilities(bool inequality, const std::vector<requirements::ViolationEvaluationType> & types);

  /** Check that the given constraint \c and requirements \req can be handled
   * by this level.
   *
   * \throws std::runtime_error if the level cannot handled them.
   */
  template<class RequirementsPtr>
  void check(const ConstraintPtr & c, const RequirementsPtr & req, bool emitWarnings = true) const;

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
  /** \param numberOfLevels The number of levels that the scheme can handle.
   * Use \c NoLimit to indicate an unlimited number of levels.
   * \param abilies The association of a level number and the LevelAbilities
   * of that level. Use \c GeneralLevel to specify all levels that do not
   * appear explicitly in the map.
   * \param scalarization Specify is this scheme can use weights to approximate
   * priority.
   *
   * \note \p numberOfLevels should be the number of levels the scheme can
   * handle without scalarization. Scalarization is an artificial way to add
   * levels. For example, a QP-based scheme should declare 2 levels, even if
   * scalarization will make it possible to emulate a few more.
   */
  SchemeAbilities(int numberOfLevels, const std::map<int, LevelAbilities> & abilities, bool scalarization = false);

  /** Check that the given constraint \c and requirements \req can be handled
   * by this scheme.
   *
   * \throws std::runtime_error if the scheme cannot handled them.
   */
  template<class RequirementsPtr>
  void check(const ConstraintPtr & c, const RequirementsPtr & req, bool emitWarnings = true) const;

private:
  int numberOfLevels_;
  bool scalarization_;
  std::map<int, LevelAbilities> abilities_;
};

template<class RequirementsPtr>
inline void LevelAbilities::check(const ConstraintPtr & c, const RequirementsPtr & req, bool /*emitWarnings*/) const
{
  // checking the constraint type
  if(c->type() != constraint::Type::EQUAL && !inequalities_)
    throw std::runtime_error("This level does not handle inequality constraints.");

  // checking the evaluation type
  auto it = std::find(evaluationTypes_.begin(), evaluationTypes_.end(), req->violationEvaluation().value());
  if(it == evaluationTypes_.end())
    throw std::runtime_error("This level does not handle the required violation evaluation value.");
}

template<class RequirementsPtr>
inline void SchemeAbilities::check(const ConstraintPtr & c, const RequirementsPtr & req, bool emitWarnings) const
{
  // check priority level value
  int p = req->priorityLevel().value();
  if(numberOfLevels_ > 0 && p >= numberOfLevels_)
  {
    if(scalarization_)
    {
      if(emitWarnings)
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

  // check abilities of the required priority level
  auto it = abilities_.find(p);
  if(it != abilities_.end())
    it->second.check(c, req, emitWarnings);
  else
    abilities_.at(GeneralLevel).check(c, req, emitWarnings);
}

} // namespace internal

} // namespace scheme

} // namespace tvm
