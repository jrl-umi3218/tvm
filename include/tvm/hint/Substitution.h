#pragma once

/* Copyright 2017-2018 CNRS-UM LIRMM, CNRS-AIST JRL
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

#include <tvm/api.h>
#include <tvm/defs.h>

#include <tvm/hint/abstract/SubstitutionCalculator.h>
#include <tvm/hint/internal/AutoCalculator.h>

#include <vector>

namespace tvm
{

namespace hint
{

  namespace abstract
  {
    class SubstitutionCalculatorImpl;
  }

  /** Hint for a substitution that could be done by the solver.
    * A substitution is a set of variables and a set of constraints used to
    * presolve them, which allow to effectively removing them from the problem.
    */
  class TVM_DLLAPI Substitution
  {
  public:
    /** Constructor for a single constraint and a single variable
      * \param cstr The constraint used for the substitution.
      * \param x The variable to substitute.
      * \param rank the rank of the matrix multiplying \p x. By default it is
      * the row size of this matrix.
      * \param calc A class that performs matrix operations related to the
      * substitution \sa tvm::hint::abstract::SubstitutionCalculator,
      * tvm::hint::abstract::SubstitutionCalculatorImpl 
      *
      * \attention Rank matters and is supposed to be fixed. This is because it
      * is influencing the size of other matrices in the problem. However, the
      * matrix doesn't need to be full rank.
      */
    Substitution(LinearConstraintPtr cstr, VariablePtr x, int rank = constant::fullRank, 
                 const abstract::SubstitutionCalculator& calc = internal::AutoCalculator());
    /** Constructor for a set of constraints and a single variable
      * \param cstr The set of constraints used for the substitution.
      * \param x The variable to substitute.
      * \param rank the rank of the matrix multiplying \p x obtained by stacking
      * the matrices in factor of \p x in each constraint. By default it is
      * the row size of this matrix.
      * \param calc A class that performs matrix operations related to the
      * substitution \sa tvm::hint::abstract::SubstitutionCalculator,
      * tvm::hint::abstract::SubstitutionCalculatorImpl 
      *
      * \attention Rank matters and is supposed to be fixed. This is because it
      * is influencing the size of other matrices in the problem. However, the
      * matrix doesn't need to be full rank.
      */
    Substitution(const std::vector<LinearConstraintPtr>& cstr, VariablePtr x, int rank = constant::fullRank,
                 const abstract::SubstitutionCalculator& calc = internal::AutoCalculator());
    /** Constructor for a single constraint and a set of variables
      * \param cstr The constraint used for the substitution.
      * \param x The set variables to substitute.
      * \param rank the rank of the matrices multiplying \p x obtained by
      * concatening the matrices in front of each \p xi. By default it is
      * the row size of this matrix.
      * \param calc A class that performs matrix operations related to the
      * substitution \sa tvm::hint::abstract::SubstitutionCalculator,
      * tvm::hint::abstract::SubstitutionCalculatorImpl 
      *
      * \attention Rank matters and is supposed to be fixed. This is because it
      * is influencing the size of other matrices in the problem. However, the
      * matrix doesn't need to be full rank.
      */
    Substitution(LinearConstraintPtr cstr, std::vector<VariablePtr>& x, int rank = constant::fullRank,
                 const abstract::SubstitutionCalculator& calc = internal::AutoCalculator());
    /** Constructor for a set of constraints and a set of variables
      * \param cstr The set of constraints used for the substitution.
      * \param x The set variables to substitute.
      * \param rank the rank of the matrix multiplying \p x, i.e the agreggation
      * of all the matrices in front of the \p xi in all the constraints. By
      * default, it is the row size of this matrix.
      * \param calc A class that performs matrix operations related to the
      * substitution \sa tvm::hint::abstract::SubstitutionCalculator,
      * tvm::hint::abstract::SubstitutionCalculatorImpl 
      *
      * \attention Rank matters and is supposed to be fixed. This is because it
      * is influencing the size of other matrices in the problem. However, the
      * matrix doesn't need to be full rank.
      */
    Substitution(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank = constant::fullRank,
                 const abstract::SubstitutionCalculator& calc = internal::AutoCalculator());

    /** The rank of the matrix multiplying \p x.*/
    int rank() const;
    /** The total (row) size of the constraints.*/
    int m() const;
    /** The set of constraints used in the substitution.*/
    const std::vector<LinearConstraintPtr>& constraints() const;
    /** The set of variables to substitute*/
    const std::vector<VariablePtr>& variables() const;
    /** Return \p true is this subsitution is based on a single constraint and
      * a single variable.
      */
    bool isSimple() const;

    /** Return the calculator used by this substitution.*/
    std::shared_ptr<abstract::SubstitutionCalculatorImpl> calculator() const;

  private:
    /** Check the validity and coherence of the parameters passed to the
      * constructor.
      */
    void check() const;

    int rank_;
    int m_;
    std::vector<LinearConstraintPtr> constraints_;
    std::vector<VariablePtr> x_;

    std::shared_ptr<abstract::SubstitutionCalculatorImpl> calculator_;
  };
}

}