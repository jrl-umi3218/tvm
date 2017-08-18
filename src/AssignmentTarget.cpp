#include "AssignmentTarget.h"
#include "Variable.h" //for Range

namespace tvm
{
  AssignmentTarget::AssignmentTarget(RangePtr range, MatrixPtr A, ConstraintType ct)
    : AssignmentTarget(range, A, nullptr, ct, RHSType::ZERO)
  {
  }

  AssignmentTarget::AssignmentTarget(RangePtr range, MatrixPtr A, VectorPtr b, ConstraintType ct, RHSType rt)
    : cstrType_(ct), rhsType_(rt), range_(range), A_(A), b_(b)
  {
    if (ct == ConstraintType::DOUBLE_SIDED)
      throw std::runtime_error("This constructor is only for single-sided constraints.");
  }

  AssignmentTarget::AssignmentTarget(RangePtr range, MatrixPtr A, VectorPtr l, VectorPtr u, RHSType rt)
    : cstrType_(ConstraintType::DOUBLE_SIDED), rhsType_(rt), range_(range), A_(A), l_(l), u_(u)
  {
    if (rt == RHSType::ZERO)
      throw std::runtime_error("RHSType::ZERO is not a valid input for this constructor. Please use the constructor for Ax=0, Ax<=0 and Ax>=0 instead.");
  }

  ConstraintType AssignmentTarget::constraintType() const
  {
    return cstrType_;
  }

  RHSType AssignmentTarget::rhsType() const
  {
    return rhsType_;
  }

  MatrixRef AssignmentTarget::A(int colStart, int colDim) const
  {
    return MatrixRef((*A_).block(range_->start, colStart, range_->dim, colDim));
  }

  VectorRef AssignmentTarget::l() const
  {
    return VectorRef((*l_).segment(range_->start, range_->dim));
  }
  
  VectorRef AssignmentTarget::u() const
  {
    return VectorRef((*u_).segment(range_->start, range_->dim));
  }
  
  VectorRef AssignmentTarget::b() const
  {
    return VectorRef((*b_).segment(range_->start, range_->dim));
  }

  MatrixRef AssignmentTarget::AFirstHalf(int colStart, int colDim) const
  {
    const int half = range_->dim / 2;
    return MatrixRef((*A_).block(range_->start, colStart, half, colDim));
  }
  
  MatrixRef AssignmentTarget::ASecondHalf(int colStart, int colDim) const
  {
    const int half = range_->dim / 2;
    return MatrixRef((*A_).block(range_->start+half, colStart, half, colDim));
  }
  
  VectorRef AssignmentTarget::bFirstHalf() const
  {
    const int half = range_->dim / 2;
    return VectorRef((*b_).segment(range_->start, half));
  }
  
  VectorRef AssignmentTarget::bSecondHalf() const
  {
    const int half = range_->dim / 2;
    return VectorRef((*b_).segment(range_->start + half, half));
  }
}