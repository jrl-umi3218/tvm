/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

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
 * The default value for this class is weights of 1 on every row.
 *
 * \internal FIXME Do we want to implement some kind of mechanism for constraints
 * whose size can change?
 */
template<bool Lightweight = true>
class AnisotropicWeightBase : public abstract::SingleSolvingRequirement<Eigen::VectorXd, Lightweight>
{
public:
  /** Default constructor: all elements of w are 1*/
  AnisotropicWeightBase() : abstract::SingleSolvingRequirement<Eigen::VectorXd, Lightweight>(Eigen::VectorXd(), true) {}

  /** Constructor for a given vector of weights \p w*/
  AnisotropicWeightBase(const Eigen::VectorXd & w)
  : abstract::SingleSolvingRequirement<Eigen::VectorXd, Lightweight>(w, false)
  {
    if((w.array() < 0).any())
      throw std::runtime_error("weights must be non-negative.");
  }

  TVM_DEFINE_LW_NON_LW_CONVERSION_OPERATORS(AnisotropicWeightBase, Eigen::VectorXd, Lightweight)
};

using AnisotropicWeight = AnisotropicWeightBase<true>;

} // namespace requirements

} // namespace tvm
