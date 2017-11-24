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

#include <exception>

namespace tvm
{

namespace exception
{

  /** Base exception class for TVM specific exceptions */
  class Exception : public std::exception {};

  /** General data-related exception */
  class DataException : public Exception {};

  /** Thrown when attempting to use unused output */
  class UnusedOutput : public DataException {};

  /** General function-related exception */
  class FunctionException : public Exception {};

  /** Thrown when attempting to call an output without an implementation */
  class UnimplementedOutput : public FunctionException {};

  /** Thrown when adding a variable already present */
  class DuplicateVariable : public FunctionException {};

  /** Thrown when attempting to access a non-existing variable */
  class NonExistingVariable : public FunctionException {};

  /** Thrown when attempting to use a feature that is not implemented in the
   * framework yet */
  class NotImplemented : public Exception {};

}  // namespace exception

}  // namespace tvm
