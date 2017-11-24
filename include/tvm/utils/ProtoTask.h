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
#include <tvm/utils/internal/ProtoTaskDetails.h>

namespace tvm
{

namespace utils
{
  /** A utiliy class to represent the "constraint" part of a Task*/
  template<constraint::Type T>
  class ProtoTask
  {
  public:
    ProtoTask(FunctionPtr f, const internal::RHS& rhs); //TODO move version

    FunctionPtr f_;
    internal::RHS rhs_;
  };

  template<>
  class ProtoTask<constraint::Type::DOUBLE_SIDED>
  {
  public:
    ProtoTask(FunctionPtr f, const internal::RHS& l, const internal::RHS& u)
      : f_(f), l_(l), u_(u)
    {
      if (l.type_ == internal::RHSType::Vector && f->size() != l.v_.size())
      {
        throw std::runtime_error("The lower bound vector you provided has not the correct size.");
      }
      if (u.type_ == internal::RHSType::Vector && f->size() != u.v_.size())
      {
        throw std::runtime_error("The upper bound vector you provided has not the correct size.");
      }
    }

    FunctionPtr f_;
    internal::RHS l_;
    internal::RHS u_;
  };

  /** Equality ProtoTask f = rhs*/
  using ProtoTaskEQ = ProtoTask<constraint::Type::EQUAL>;

  /** Inequality ProtoTask f <= rhs*/
  using ProtoTaskLT = ProtoTask<constraint::Type::LOWER_THAN>;

  /** Inequality ProtoTask f >= rhs*/
  using ProtoTaskGT = ProtoTask<constraint::Type::GREATER_THAN>;

  /** Double sided inequality ProtoTask l <= f <= u*/
  using ProtoTaskDS = ProtoTask<constraint::Type::DOUBLE_SIDED>;

} // namespace utils

} // namespace tvm

/** Conveniency operators to form ProtoTask f op rhs
* (or l <= f <= u)
*
* \param f the function to form the task
* \param rhs a double or a Eigen::Vector with the sane size as the function
* Note that for a double you explicitely need to write a double (e.g 0.,
* not 0), otherwise the compiler won't be able to decide wich overload to
* pick between this and shared_ptr operator.
*/
tvm::utils::ProtoTaskEQ operator==(tvm::FunctionPtr f, const tvm::utils::internal::RHS& rhs);
tvm::utils::ProtoTaskEQ operator==(const tvm::utils::internal::RHS& rhs, tvm::FunctionPtr f);
tvm::utils::ProtoTaskGT operator>=(tvm::FunctionPtr f, const tvm::utils::internal::RHS& rhs);
tvm::utils::ProtoTaskLT operator>=(const tvm::utils::internal::RHS& rhs, tvm::FunctionPtr f);
tvm::utils::ProtoTaskLT operator<=(tvm::FunctionPtr f, const tvm::utils::internal::RHS& rhs);
tvm::utils::ProtoTaskGT operator<=(const tvm::utils::internal::RHS& rhs, tvm::FunctionPtr f);

tvm::utils::ProtoTaskGT operator>=(tvm::VariablePtr x, const tvm::utils::internal::RHS& rhs);
tvm::utils::ProtoTaskLT operator>=(const tvm::utils::internal::RHS& rhs, tvm::VariablePtr f);
tvm::utils::ProtoTaskLT operator<=(tvm::VariablePtr f, const tvm::utils::internal::RHS& rhs);
tvm::utils::ProtoTaskGT operator<=(const tvm::utils::internal::RHS& rhs, tvm::VariablePtr f);

tvm::utils::ProtoTaskDS operator>=(const tvm::utils::ProtoTaskLT& ptl, const tvm::utils::internal::RHS& rhs);
tvm::utils::ProtoTaskDS operator<=(const tvm::utils::ProtoTaskGT& ptg, const tvm::utils::internal::RHS& rhs);


template<tvm::constraint::Type T>
inline tvm::utils::ProtoTask<T>::ProtoTask(tvm::FunctionPtr f, const tvm::utils::internal::RHS& rhs)
  : f_(f), rhs_(rhs)
{
  if (rhs.type_ == tvm::utils::internal::RHSType::Vector && f->size() != rhs.v_.size())
  {
    throw std::runtime_error("The vector you provided has not the correct size.");
  }
}

inline tvm::utils::ProtoTaskEQ operator==(tvm::FunctionPtr f, const tvm::utils::internal::RHS& rhs)
{
  return { f, rhs };
}

inline tvm::utils::ProtoTaskEQ operator==(const tvm::utils::internal::RHS& rhs, tvm::FunctionPtr f)
{
  return { f, rhs };
}

inline tvm::utils::ProtoTaskGT operator>=(tvm::FunctionPtr f, const tvm::utils::internal::RHS& rhs)
{
  return { f, rhs };
}

inline tvm::utils::ProtoTaskLT operator>=(const tvm::utils::internal::RHS& rhs, tvm::FunctionPtr f)
{
  return { f, rhs };
}

inline tvm::utils::ProtoTaskLT operator<=(tvm::FunctionPtr f, const tvm::utils::internal::RHS& rhs)
{
  return { f, rhs };
}

inline tvm::utils::ProtoTaskGT operator<=(const tvm::utils::internal::RHS& rhs, tvm::FunctionPtr f)
{
  return { f, rhs };
}

inline tvm::utils::ProtoTaskDS operator>=(const tvm::utils::ProtoTaskLT& ptl, const tvm::utils::internal::RHS& rhs)
{
  return { ptl.f_, rhs, ptl.rhs_ };
}

inline tvm::utils::ProtoTaskDS operator<=(const tvm::utils::ProtoTaskGT& ptg, const tvm::utils::internal::RHS& rhs)
{
  return { ptg.f_, ptg.rhs_, rhs };
}

inline tvm::utils::ProtoTaskGT operator>=(tvm::VariablePtr x, const tvm::utils::internal::RHS& rhs)
{
  return std::make_shared<tvm::function::IdentityFunction>(x) >= rhs;
}

inline tvm::utils::ProtoTaskLT operator>=(const tvm::utils::internal::RHS& rhs, tvm::VariablePtr x)
{
  return std::make_shared<tvm::function::IdentityFunction>(x) <= rhs;
}

inline tvm::utils::ProtoTaskLT operator<=(tvm::VariablePtr x, const tvm::utils::internal::RHS& rhs)
{
  return std::make_shared<tvm::function::IdentityFunction>(x) <= rhs;
}

inline tvm::utils::ProtoTaskGT operator<=(const tvm::utils::internal::RHS& rhs, tvm::VariablePtr x)
{
  return std::make_shared<tvm::function::IdentityFunction>(x) >= rhs;
}
