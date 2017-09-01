#pragma once

#include <memory>

#include <Eigen/Core>

#include "ConstraintEnums.h"
#include "FirstOrderProvider.h"
#include <tvm/data/OutputSelector.h>
#include <tvm/api.h>

namespace tvm
{
  class Variable;

  /** This is a helper class to define Constraint. Its sole purpose is to
    * declare the outputs L, U and E, (L and U being the lower and upper bounds
    * for inequality constraints, and E the term the constraint is equal to for
    * equality constraints), so that Constraint can the dynamically disable 
    * what it does not use.
    */
  class TVM_DLLAPI ConstraintBase : public internal::FirstOrderProvider
  {
  public:
    SET_OUTPUTS(ConstraintBase, L, U, E);

  protected:
    ConstraintBase(int m);
  };

  /** Base class for representing a constraint. 
    *
    * It manages the enabling/disabling of the outputs L, U and E (depending 
    * on its type), and the memory of the associated cache.
    *
    * FIXME: have the updateValue here and add an output check()
    */
  class TVM_DLLAPI Constraint : public data::OutputSelector<ConstraintBase>
  {
  public:
    using ConstraintBase::Output;

    const Eigen::VectorXd& l() const;
    const Eigen::VectorXd& u() const;
    const Eigen::VectorXd& e() const;

    ConstraintType constraintType() const;
    ConstraintRHS constraintRhs() const;

    /** Note: by default, these methods return the cached value.
      * However, they are virtual in case the user might want to bypass the cache.
      * This would be typically the case if he/she wants to directly return the
      * output of another method.
      *
      * Question: should they be made protected or stay public
      */
    virtual const Eigen::VectorXd& lNoCheck() const;
    virtual const Eigen::VectorXd& uNoCheck() const;
    virtual const Eigen::VectorXd& eNoCheck() const;

  protected:
    Constraint(ConstraintType ct, ConstraintRHS cr, int m=0);
    void resizeCache() override;

    Eigen::VectorXd l_;
    Eigen::VectorXd u_;
    Eigen::VectorXd e_;

  private:
    ConstraintType  cstrType_;
    ConstraintRHS   constraintRhs_;

    bool usel_;
    bool useu_;
    bool usee_;
  };
}
