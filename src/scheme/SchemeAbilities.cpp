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

#include <tvm/scheme/internal/SchemeAbilities.h>

#include <tvm/constraint/abstract/Constraint.h>
#include <tvm/requirements/SolvingRequirements.h>

#include <iostream>
#include <sstream>

namespace tvm
{

namespace scheme
{

namespace internal
{
LevelAbilities::LevelAbilities(bool inequality, const std::vector<requirements::ViolationEvaluationType> & types)
: inequalities_(inequality), evaluationTypes_(types)
{
}

SchemeAbilities::SchemeAbilities(int numberOfLevels,
                                 const std::map<int, LevelAbilities> & abilities,
                                 bool scalarization)
: numberOfLevels_(numberOfLevels), scalarization_(scalarization), abilities_(abilities)
{
  assert(GeneralLevel < 0 && "Implementation of this class assumes that GeneralLevel is non positive.");
  assert(GeneralLevel == NoLimit && "Implementation of this class assumes that GeneralLevel is equal to NoLimit.");

  if(numberOfLevels < 0 && numberOfLevels != NoLimit)
    throw std::runtime_error("Incorrect number of levels. This number must be nonnegative or equal to NoLimit");

  if(numberOfLevels >= 0)
  {
    // if there is not general entry in abilities, then we check that each level has its own entry.
    if(abilities.count(GeneralLevel) == 0)
    {
      for(int i = 0; i < numberOfLevels; ++i)
      {
        if(abilities.count(i) == 0)
        {
          std::stringstream s;
          s << "No abilities given for level " << i << "." << std::endl;
          throw std::runtime_error(s.str());
        }
      }
    }
  }
  else
  {
    // we check that there is a general entry in abilities
    if(abilities.count(GeneralLevel) == 0)
      throw std::runtime_error("No general level abilities given.");
  }
}

} // namespace internal

} // namespace scheme

} // namespace tvm
