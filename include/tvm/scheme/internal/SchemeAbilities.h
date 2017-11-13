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
#include <tvm/requirements/ViolationEvaluation.h>

#include <map>
#include <vector>

namespace tvm
{

namespace scheme
{

namespace internal
{

  class TVM_DLLAPI LevelAbilities
  {
  public:
    LevelAbilities(bool inequality, const std::vector<requirements::ViolationEvaluationType>& types);

    void check(const ConstraintPtr& c, const SolvingRequirementsPtr& req, bool emitWarnings = true) const;

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

    void check(const ConstraintPtr& c, const SolvingRequirementsPtr& req, bool emitWarnings = true) const;

  private:
    int numberOfLevels_;
    bool scalarization_;
    std::map<int, LevelAbilities> abilities_;
  };

}  // namespace internal

}  // namespace scheme

}  // namespace tvm
