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
#include <tvm/constraint/enums.h>

#include <Eigen/Core>

namespace tvm
{

namespace constraint
{

namespace internal
{

  /** This class manages the vectors l, u and e that appear in the various
    * types of constraints.*/
  class RHSVectors
  {
  public:
    RHSVectors(Type ct, RHS cr);

    void resize(int n);

    Eigen::VectorXd& l();
    const Eigen::VectorXd& l() const;
    Eigen::VectorXd& u();
    const Eigen::VectorXd& u() const;
    Eigen::VectorXd& e();
    const Eigen::VectorXd& e() const;

    /** Return the vector specified by t:
      * - l if t == Type::GREATER_THAN
      * - u if t == Type::LOWER_THAN
      * - e if t == Type::EQUAL
      * If t == Type::DOUBLE_SIDED, throw an exception
      */
    Eigen::VectorXd& rhs(Type t);
    const Eigen::VectorXd& rhs(Type t) const;

    bool use_l() const;
    bool use_u() const;
    bool use_e() const;

  private:
    Eigen::VectorXd l_;
    Eigen::VectorXd u_;
    Eigen::VectorXd e_;

    const bool use_l_;
    const bool use_u_;
    const bool use_e_;
  };



  inline Eigen::VectorXd& RHSVectors::l()
  {
    return l_;
  }

  inline const Eigen::VectorXd& RHSVectors::l() const
  {
    return l_;
  }

  inline Eigen::VectorXd& RHSVectors::u()
  {
    return u_;
  }

  inline const Eigen::VectorXd& RHSVectors::u() const
  {
    return u_;
  }

  inline Eigen::VectorXd& RHSVectors::e()
  {
    return e_;
  }

  inline const Eigen::VectorXd& RHSVectors::e() const
  {
    return e_;
  }


  inline Eigen::VectorXd& RHSVectors::rhs(Type t)
  {
    switch (t)
    {
    case Type::LOWER_THAN: return u_;
    case Type::GREATER_THAN: return l_;
    case Type::EQUAL: return e_;
    case Type::DOUBLE_SIDED: throw std::runtime_error("This methods is not available for DOUBLE_SIDED");
    }
  }

  inline const Eigen::VectorXd& RHSVectors::rhs(Type t) const
  {
    switch (t)
    {
    case Type::LOWER_THAN: return u_;
    case Type::GREATER_THAN: return l_;
    case Type::EQUAL: return e_;
    case Type::DOUBLE_SIDED: throw std::runtime_error("This methods is not available for DOUBLE_SIDED");
    }
  }

  inline bool RHSVectors::use_l() const
  {
    return use_l_;
  }

  inline bool RHSVectors::use_u() const
  {
    return use_u_;
  }

  inline bool RHSVectors::use_e() const
  {
    return use_e_;
  }

} // namespace internal

} // namespace constraint

} // namespace tvm