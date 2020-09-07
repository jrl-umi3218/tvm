/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/constraint/enums.h>
#include <tvm/defs.h>

#include <vector>

namespace tvm
{
namespace hint
{
namespace internal
{
class Substitutions;
}
} // namespace hint
namespace function
{
class BasicLinearFunction;
}

namespace scheme
{
namespace internal
{
class ProblemComputationData;

/** Get the computation data linked to a particular resolution scheme.
 * If this data does not exist, create it, using the resolution scheme as
 * a factory
 * \param The resolution scheme for which we want to get back the
 * computation data. It needs to have a method \a createComputationData.
 */
template<typename Problem, typename Scheme>
inline ProblemComputationData * getComputationData(Problem & problem, const Scheme & resolutionScheme);

/** We consider as bound a constraint with a single variable and a diagonal,
 * invertible jacobian.
 * It would be possible to accept non-invertible sparse diagonal jacobians
 * as well, in which case the zero elements of the diagonal would
 * correspond to non-existing bounds, but this requires quite a lot of
 * work for something that is unlikely to happen and could be expressed
 * by changing the bound itself to +/- infinity.
 */
bool TVM_DLLAPI isBound(const ConstraintPtr & c);

/** Assert if a constraint is a bound in the presence of substitutions
 * \param c the constraint
 * \param subs the set of substitutions
 * We have a bound if c is a bound and the variable is not substituted or it
 * is by an expression with a single varaible and an invertible, diagonal
 * jacobian.
 */
bool TVM_DLLAPI isBound(const ConstraintPtr & c, const hint::internal::Substitutions & subs);

/** Assert if a constraint is a bound in the presence of substitutions
 * \param c the constraint
 * \param x the set of variables being substituted
 * \param xsub the set of substitution functions corresponding to those variables.
 * We have a bound if c is a bound and the variable is not substituted or it
 * is by an expression with a single varaible and an invertible, diagonal
 * jacobian.
 */
bool TVM_DLLAPI isBound(const ConstraintPtr & c,
                        const std::vector<VariablePtr> & x,
                        const std::vector<std::shared_ptr<function::BasicLinearFunction>> & xsub);

/** Check if the constraint can be used as a bound for a given target convention.
 * For example l <= x <= u is a bound, but it cannot be transformed into x >= lb.
 */
bool TVM_DLLAPI canBeUsedAsBound(const ConstraintPtr & c,
                                 const hint::internal::Substitutions & subs,
                                 constraint::Type targetConvention);

/** Version with separated susbtituted variables and substitution functions.*/
bool TVM_DLLAPI canBeUsedAsBound(const ConstraintPtr & c,
                                 const std::vector<VariablePtr> & x,
                                 const std::vector<std::shared_ptr<function::BasicLinearFunction>> & xsub,
                                 constraint::Type targetConvention);
} // namespace internal
} // namespace scheme
} // namespace tvm
