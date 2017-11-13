#pragma once

/* Copyright 2017 CNRS-UM LIRMM, CNRS-AIST JRL
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
#include <tvm/requirements/abstract/SingleSolvingRequirement.h>

#include <Eigen/Core>

namespace tvm
{

namespace requirements
{

  /** This class represents an anisotropic weight to give more or less
   * importance to the different rows of a constraint. It is given as a
   * vector w. It results in the violation v(x) of the constraint being
   * multiplied by diag(w'), where w' depends on the constraint violation
   * evaluation chosen.  w' is such that if w was a uniform vector with all
   * components equal to alpha, the result would be coherent with using a
   * Weight with value alpha.
   *
   * This class can be redundant with Weight, as having a Weight alpha and
   * an AnisotropicWeight w is the same as having a just an
   * AnisotropicWeight w.  As a guideline, it should be used only to
   * discriminate between the rows of a constraint, while Weight would be
   * used to discriminate between different constraints. As such the
   * "mean" value of w should be 1.
   *
   * This class replaces the notion of dimWeight in Tasks.
   *
   * FIXME Do we want to implement some kind of mechanism for constraints
   * whose size can change?
   */
  class TVM_DLLAPI AnisotropicWeight : public abstract::SingleSolvingRequirement<Eigen::VectorXd>
  {
  public:
    AnisotropicWeight();
    AnisotropicWeight(const Eigen::VectorXd& w);
  };

}  // namespace requirements

}  // namespace tvm
