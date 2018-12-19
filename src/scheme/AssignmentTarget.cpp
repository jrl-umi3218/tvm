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

#include <tvm/scheme/internal/AssignmentTarget.h>

#include <tvm/Range.h>

namespace tvm
{

namespace scheme
{

namespace internal
{

  AssignmentTarget::AssignmentTarget(RangePtr range, MatrixRef A, constraint::Type ct)
    : AssignmentTarget(range, A, Eigen::Map<Eigen::VectorXd>(nullptr, 0), ct, constraint::RHS::ZERO)
  {
  }

  AssignmentTarget::AssignmentTarget(RangePtr range, MatrixRef A, VectorRef b, constraint::Type ct, constraint::RHS cr)
    : targetType_(TargetType::Linear), cstrType_(ct), constraintRhs_(cr), range_(range), A_(A), b_(b)
  {
    if (ct == constraint::Type::DOUBLE_SIDED)
      throw std::runtime_error("This constructor is only for single-sided constraints.");
  }

  AssignmentTarget::AssignmentTarget(RangePtr range, MatrixRef A, VectorRef l, VectorRef u, constraint::RHS cr)
    : targetType_(TargetType::Linear), cstrType_(constraint::Type::DOUBLE_SIDED), constraintRhs_(cr), range_(range), A_(A), l_(l), u_(u)
  {
    if (cr == constraint::RHS::ZERO)
      throw std::runtime_error("constraint::RHS::ZERO is not a valid input for this constructor. Please use the constructor for Ax=0, Ax<=0 and Ax>=0 instead.");
  }

  AssignmentTarget::AssignmentTarget(RangePtr range, VectorRef l, VectorRef u)
    : targetType_(TargetType::Linear), cstrType_(constraint::Type::DOUBLE_SIDED), constraintRhs_(constraint::RHS::AS_GIVEN), range_(range), l_(l), u_(u)
  {
  }

  AssignmentTarget::AssignmentTarget(RangePtr range, VectorRef lu, constraint::Type ct)
    : targetType_(TargetType::Linear), cstrType_(ct), range_(range)
  {
    if (ct == constraint::Type::GREATER_THAN)
    {
      l_ = lu;
    }
    else if (ct == constraint::Type::LOWER_THAN)
    {
      u_ = lu;
    }
    else
    {
      throw std::runtime_error("This constructor is for simple-bound constraint, only GREATER_THAN or LOWER_THAN are valid options.");
    }
  }

  AssignmentTarget::AssignmentTarget(MatrixRef Q, VectorRef q, constraint::RHS cr)
    : targetType_(TargetType::Quadratic), constraintRhs_(cr), Q_(Q), q_(q)
  {
  }

  TargetType AssignmentTarget::targetType() const
  {
    return targetType_;
  }

  constraint::Type AssignmentTarget::constraintType() const
  {
    return cstrType_;
  }

  constraint::RHS AssignmentTarget::constraintRhs() const
  {
    return constraintRhs_;
  }

  int AssignmentTarget::size() const
  {
    return range_->dim;
  }

  MatrixRef AssignmentTarget::A(int colStart, int colDim) const
  {
    //return MatrixRef(const_cast<AssignmentTarget*>(this)->A_.block(range_->start, colStart, range_->dim, colDim));
    return MatrixRef(static_cast<MatrixRef>(A_).block(range_->start, colStart, range_->dim, colDim));
  }

  MatrixRef AssignmentTarget::Q() const
  {
    return Q_;
  }

  VectorRef AssignmentTarget::l() const
  {
    return VectorRef(static_cast<VectorRef>(l_).segment(range_->start, range_->dim));
  }

  VectorRef AssignmentTarget::u() const
  {
    return VectorRef(static_cast<VectorRef>(u_).segment(range_->start, range_->dim));
  }

  VectorRef AssignmentTarget::b() const
  {
    return VectorRef(static_cast<VectorRef>(b_).segment(range_->start, range_->dim));
  }

  VectorRef AssignmentTarget::q() const
  {
    return q_;
  }

  MatrixRef AssignmentTarget::AFirstHalf(int colStart, int colDim) const
  {
    const int half = range_->dim / 2;
    return MatrixRef(static_cast<MatrixRef>(A_).block(range_->start, colStart, half, colDim));
  }

  MatrixRef AssignmentTarget::ASecondHalf(int colStart, int colDim) const
  {
    const int half = range_->dim / 2;
    return MatrixRef(static_cast<MatrixRef>(A_).block(range_->start+half, colStart, half, colDim));
  }

  VectorRef AssignmentTarget::bFirstHalf() const
  {
    const int half = range_->dim / 2;
    return VectorRef(static_cast<VectorRef>(b_).segment(range_->start, half));
  }

  VectorRef AssignmentTarget::bSecondHalf() const
  {
    const int half = range_->dim / 2;
    return VectorRef(static_cast<VectorRef>(b_).segment(range_->start + half, half));
  }

}  // namespace internal

}  // namespace scheme

}  // namespace tvm
