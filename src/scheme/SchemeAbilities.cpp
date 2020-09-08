/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

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
{}

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
