/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#include <tvm/constraint/abstract/Constraint.h>

#include <tvm/exception/exceptions.h>

namespace tvm
{

namespace constraint
{

namespace abstract
{

  Constraint::Constraint(Type ct, RHS cr, int m)
    : graph::abstract::OutputSelector<Constraint, tvm::internal::FirstOrderProvider>(m)
    , vectors_(ct, cr)
    , cstrType_(ct)
    , constraintRhs_(cr)
  {
    if (ct == Type::DOUBLE_SIDED && cr == RHS::ZERO)
      throw std::runtime_error("The combination (ConstraintType::DOUBLE_SIDED, ConstraintRHS::ZERO) is forbidden. Please use (ConstraintType::EQUAL, ConstraintRHS::ZERO) instead.");
    //FIXME: we make the choice here to have no "rhs" output when the ConstraintRHS is zero.
    //An alternative is to use and set to zero the relevant vectors, but then we need
    //to prevent a derived class to change their value.
    resizeCache();
    if (!vectors_.use_l())
      disableOutput(Output::L);
    if (!vectors_.use_u())
      disableOutput(Output::U);
    if (!vectors_.use_e())
      disableOutput(Output::E);
  }

  void Constraint::resizeCache()
  {
    tvm::internal::FirstOrderProvider::resizeCache();
    vectors_.resize(size());
  }

  Type Constraint::type() const
  {
    return cstrType_;
  }

  RHS Constraint::rhs() const
  {
    return constraintRhs_;
  }

}  // namespace abstract

}  // namespace constraint

}  // namespace tvm
