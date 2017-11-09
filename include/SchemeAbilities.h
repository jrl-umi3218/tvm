#pragma once

#include <map>
#include <vector>

#include "defs.h"
#include "SolvingRequirements.h"
#include "tvm/api.h"

namespace tvm
{
  namespace scheme
  {
    class TVM_DLLAPI LevelAbilities
    {
    public:
      LevelAbilities(bool inequality, const std::vector<ViolationEvaluationType>& types);

      void check(const ConstraintPtr& c, const SolvingRequirementsPtr& req, bool emitWarnings = true) const;

    private:
      bool inequalities_;
      std::vector<ViolationEvaluationType> evaluationTypes_;
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
  }
}
