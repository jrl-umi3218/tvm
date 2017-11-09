#include <iostream>
#include <sstream>

#include "Constraint.h"
#include "SchemeAbilities.h"


namespace tvm
{
  namespace scheme
  {
    static int GeneralLevel = -1;
    static int NoLimit = GeneralLevel;

    LevelAbilities::LevelAbilities(bool inequality, const std::vector<ViolationEvaluationType>& types)
      : inequalities_(inequality)
      , evaluationTypes_(types)
    {
    }

    void LevelAbilities::check(const ConstraintPtr& c, const SolvingRequirementsPtr& req, bool /*emitWarnings*/) const
    {
      //checking the constraint type
      if (c->constraintType() != ConstraintType::EQUAL && !inequalities_)
        throw std::runtime_error("This level does not handle inequality constraints.");

      //checking the evaluation type
      auto it = std::find(evaluationTypes_.begin(), evaluationTypes_.end(), req->violationEvaluation().value());
      if (it == evaluationTypes_.end())
        throw std::runtime_error("This level does not handle the required violation evaluation value.");
    }

    SchemeAbilities::SchemeAbilities(int numberOfLevels, const std::map<int, LevelAbilities>& abilities, bool scalarization)
      : numberOfLevels_(numberOfLevels)
      , scalarization_(scalarization)
      , abilities_(abilities)
    {
      assert(GeneralLevel < 0 && "Implementation of this class assumes that GeneralLevel is non positive.");
      assert(GeneralLevel == NoLimit && "Implementation of this class assumes that GeneralLevel is equal to NoLimit.");

      if (numberOfLevels < 0 && numberOfLevels != NoLimit)
        throw std::runtime_error("Incorrect number of levels. This number must be nonnegative or equal to NoLimit");

      if (numberOfLevels >= 0)
      {
        // if there is not general entry in abilities, then we check that each level has its own entry.
        if (abilities.count(GeneralLevel) == 0)
        {
          for (int i = 0; i < numberOfLevels; ++i)
          {
            if (abilities.count(i) == 0)
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
        //we check that there is a general entry in abilities
        if (abilities.count(GeneralLevel) == 0)
          throw std::runtime_error("No general level abilities given.");
      }
    }

    void SchemeAbilities::check(const ConstraintPtr& c, const SolvingRequirementsPtr& req, bool emitWarnings) const
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
  }
}
