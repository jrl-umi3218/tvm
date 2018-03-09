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
#include <tvm/Range.h>
#include <tvm/constraint/BasicLinearConstraint.h>
#include <tvm/function/BasicLinearFunction.h>
#include <tvm/hint/Substitution.h>
#include <tvm/hint/abstract/SubstitutionCalculatorImpl.h>

#include <Eigen/Core>

#include <vector>

namespace tvm
{

namespace hint
{

namespace internal
{
  /** A class to perform all the computations related to a group of dependent
    * substitutions.
    */
  class TVM_DLLAPI SubstitutionUnit
  {
  public:
    /** Build a SubstitutionUnit containing the substitutions from 
      * \p substitutionPool with indices in \p groups[i] for each \p i in
      * \p order.
      * Substitutions in a same groups[i] are merged as a single subsitution.
      */
    SubstitutionUnit(const std::vector<Substitution>& substitutionPool, 
                     const std::vector<std::vector<size_t>>& groups,
                     const std::vector<size_t> order);

    /** Update the matrices and vectors used for the substitutions.*/
    void update();

    /** Return the substituted variables*/
    const std::vector<VariablePtr>& variables() const;
    /** Return the function giving the values of the substituted variables. The
      * order of the vector corresponds to the order of the vector returned by
      * \p variables()
      */
    const std::vector<std::shared_ptr<function::BasicLinearFunction>>& variableSubstitutions() const;
    /** Return all the z variables (including the empty ones).*/
    const std::vector<VariablePtr>& additionalVariables() const;
    /** Return the additional constraints to be added to the problem.*/
    const std::vector<std::shared_ptr<constraint::BasicLinearConstraint>>& additionalConstraints() const;


  private:
    /** Build \p substitutions_ from the inputs
      * \sa SubstitutionUnit:SubstitutionUnit
      */
    void extractSubstitutions(const std::vector<Substitution>& substitutionPool, const std::vector<std::vector<size_t>>& groups, const std::vector<size_t> order);
    /** Populates \p constraints_, \p x_, \p y_, \p z_, \p substitutionMRanges_,
      * \p substitutionNRanges_. \p constraintsY_, \p CXdependencies and \p m_
      * from \p substitutions_.
      */
    void scanSubstitutions();
    /** Compute the dependencies between the constraints, variables and susbsitutions.*/
    void computeDependencies();
    /** Resize the matrices and initialize to zero some parts that will be used*/
    void initializeMatrices();
    /** Build \p varSubstitutions_ and \p remainings_*/
    void createFunctions();


    /** Sum of the sizes of all the constraints in all the substitutions.*/
    int m_;
    /** The substitutions in this unit.*/
    std::vector<Substitution> substitutions_;
    /** The calculators associated to the substitutions*/
    std::vector<std::shared_ptr<abstract::SubstitutionCalculatorImpl>> calculators_;
    /** The list of all constraints appearing in the substitutions*/
    std::vector<LinearConstraintPtr> constraints_;
    /** The substituted variables, by order of substitution*/
    VariableVector x_;
    /** The non-substituted variables*/
    VariableVector y_;
    /** The additionnal nullspace variables*/
    VariableVector z_;
    /** Imagining all the constraints of all the substitutions stacked, the i-th
      * elements gives the rows corresponding to substitutions_[i].
      */
    std::vector<Range> substitutionMRanges_;
    /** Imagining all the constraints of all the substitutions stacked, the i-th
      * elements gives the columns corresponding to the variables of
      * substitutions_[i].
      */
    std::vector<Range> substitutionNRanges_;

    /** sub2cstr[i]_ gives the indices relative to constraints_ of the
      * constraints corresponding to susbtitutions_[i].
      */
    std::vector<std::vector<size_t>> sub2cstr_;
    /** sub2cstr[i]_ gives the indices relative to x_ of the variables x
      * corresponding to susbtitutions_[i].
      */
    std::vector<std::vector<size_t>> sub2x_;
    /** x2sub_[i] gives the index of the substitution from which x_[i] is
      * computed.
      */
    std::vector<size_t> x2sub_;
    /** xRange_[i] gives the range of x_[i] wrt substitutions_[i].variables().
      * We cache it for efficiency purpose.
      */
    std::vector<Range> xRange_;

    /** constraints_[i] contains y_[constraintsY_[i][j]].*/
    std::vector<std::vector<int>> constraintsY_;
    /** constraints_[i] depends on x_[CXdependencies_[i][j]].*/
    std::vector<std::vector<int>> CXdependencies_;
    /** x_[i] depends on y_[XYdependencies_[i][j]].*/
    std::vector<std::vector<int>> XYdependencies_;
    /** x_[i] depends on z_[XZdependencies_[i][j]].*/
    std::vector<std::vector<int>> XZdependencies_;
    /** substitutions_[i] depends on y_[SYdependencies_[i][j]].*/
    std::vector<std::vector<int>> SYdependencies_;
    /** substitutions_[i] depends on z_[SZdependencies_[i][j]].*/
    std::vector<std::vector<int>> SZdependencies_;

    /** The substituted variables as linear functions of the non-sustituted ones.*/
    std::vector<std::shared_ptr<function::BasicLinearFunction>> varSubstitutions_;
    /** The remaining constraints on the non-substituted variables.*/
    std::vector<std::shared_ptr<constraint::BasicLinearConstraint>> remaining_;

    /** All the matrices B_{i,j} assembled, where \p i corresponds to the i-th
      * constraint in \p constraints_ and \p j corresponds to the j-th variable
      * in \p y_.
      */
    Eigen::MatrixXd B_;
    /** All the matrices Z_{i,j} assembled, where \p i corresponds to the i-th
      * constraint in \p constraints_ and \p j corresponds to the j-th variable
      * in \p z_.
      */
    Eigen::MatrixXd Z_;
    /** All the rhs c_i assembled, where \p i corresponds to the i-th constraint
      * in \p constraints_.
      */
    Eigen::VectorXd c_;
    /** All the matrices M_{i,j} assembled, where \p i corresponds to the i-th
      * variables in \p x_ and \p j corresponds to the j-th variable in \p y_.
      * We have M_{i1:i2,j} = - A_{i1:i2}^# B_{i1:i2,j}, where i1:i2 is the 
      * range of \p x_ variables in a given substitution.
      */
    Eigen::MatrixXd M_;
    /** All the matrices Z_{i,j} assembled, where \p i corresponds to the i-th
      * variables in \p x_ and \p j corresponds to the j-th variable in \p z_.
      * We have AsZ_{i1:i2,j} = - A_{i1:i2}^# Z_{i1:i2,j}, where i1:i2 is the 
      * range of \p x_ variables in a given substitution.
      */
    Eigen::MatrixXd AsZ_;
    /** Assembly of all the u_i = A_i^# c_i*/
    Eigen::VectorXd u_;
    /** For each element \p substitutions_[i], an assembly of S_i^T B_{i,j},
      * where \p j corresponds to the j-th variable in y_.
      */
    std::vector<Eigen::MatrixXd> StB_;
    /** For each element \p substitutions_[i], an assembly of S_i^T Z_{i,j},
      * where \p j corresponds to the j-th variable in z_.
      */
    std::vector<Eigen::MatrixXd> StZ_;
     /** For each element \p substitutions_[i], an assembly of S_i^T c_i. */
    std::vector<Eigen::VectorXd> Stc_;

    /** A temporary vector used in the update.*/
    std::vector<bool> firstY_;
    /** A temporary vector used in the update.*/
    std::vector<bool> firstZ_;
  };


} // internal

} // hint

} // tvm