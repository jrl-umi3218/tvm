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
    public:
      double big_number() const;
      void big_number(double big);

    protected:
      /** Constructor, meant only for derived classes 
        *
        * \param abilities The set of abilities for this scheme.
        * \param big A big number use to represent infinity, in particular when
        * specifying non-existing bounds (e.g. x <= Inf is given as x <= big).
        */
      ResolutionScheme(const SchemeAbilities& abilities, double big = std::numeric_limits<double>::max()/2);

      void addVariable(VariablePtr var);
      void addVariable(const std::vector<VariablePtr>& vars);
      void removeVariable(Variable* v);
      void removeVariable(const std::vector<VariablePtr>& vars);

      /** The problem variable*/
      VariableVector x_;
      SchemeAbilities abilities_;

      /** A number to use for infinite bounds*/
      double big_number_;
    };

    /** Base class for scheme solving linear problems*/
    class TVM_DLLAPI LinearResolutionScheme : public ResolutionScheme
    {
    public:
      void solve();

    protected:
      LinearResolutionScheme(const SchemeAbilities& abilities, std::shared_ptr<LinearizedControlProblem> pb, double big = std::numeric_limits<double>::max() / 2);

      virtual void solve_() = 0;

      std::shared_ptr<LinearizedControlProblem> problem_;
    };
  }
}