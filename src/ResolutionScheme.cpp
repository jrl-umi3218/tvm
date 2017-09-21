#include "ResolutionScheme.h"
#include "LinearizedControlProblem.h"

namespace tvm
{
  namespace scheme
  {
    ResolutionScheme::ResolutionScheme(const SchemeAbilities& abilities)
      : abilities_(abilities)
    {
    }

    void ResolutionScheme::addVariable(VariablePtr var)
    {
      x_.add(var, true);
    }

    void ResolutionScheme::addVariable(const std::vector<VariablePtr>& vars)
    {
      for (const auto& v : vars)
        addVariable(v);
    }

    void ResolutionScheme::removeVariable(Variable* v)
    {
      //we don't raise an exception is the variable is not there, as we merge 
      //identical variables when we add them.
      x_.remove(*v, true);
    }

    void ResolutionScheme::removeVariable(const std::vector<VariablePtr>& vars)
    {
      for (const auto& v : vars)
        removeVariable(v.get());
    }
    void LinearResolutionScheme::solve()
    {
      problem_->update();
      solve_();
    }

    LinearResolutionScheme::LinearResolutionScheme(const SchemeAbilities& abilities, std::shared_ptr<LinearizedControlProblem> pb)
      : ResolutionScheme(abilities)
      , problem_(pb)
    {
    }
  }
}