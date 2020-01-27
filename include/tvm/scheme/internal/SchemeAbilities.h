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
