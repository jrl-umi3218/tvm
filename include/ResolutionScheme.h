#pragma once

#include "SchemeAbilities.h"
#include "VariableVector.h"

namespace tvm
{
  class LinearizedControlProblem;

  namespace scheme
  {
    /** Base class for solving a ControlProblem*/
    class TVM_DLLAPI ResolutionScheme
    {
    protected:
      ResolutionScheme(const SchemeAbilities& abilities);

      void addVariable(VariablePtr var);
      void addVariable(const std::vector<VariablePtr>& vars);
      void removeVariable(Variable* v);
      void removeVariable(const std::vector<VariablePtr>& vars);

      /** The problem variable*/
      VariableVector x_;
      SchemeAbilities abilities_;
    };

    /** Base class for scheme solving linear problems*/
    class TVM_DLLAPI LinearResolutionScheme : public ResolutionScheme
    {
    public:
      void solve();

    protected:
      LinearResolutionScheme(const SchemeAbilities& abilities, std::shared_ptr<LinearizedControlProblem> pb);

      virtual void solve_() = 0;

      std::shared_ptr<LinearizedControlProblem> problem_;
    };
  }
}