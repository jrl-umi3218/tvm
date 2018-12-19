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

#pragma once

#include <exception>
#include <string>

namespace tvm
{

namespace exception
{

  /** Base exception class for TVM specific exceptions */
  class Exception : public std::exception
  {
  public:
    Exception() : std::exception() {}
    Exception(const std::string & what) : what_(what) {}
    Exception(const char * what) : what_(what) {}
    const char * what() const noexcept override { return what_.c_str(); }
  private:
    std::string what_;
  };

  /** General data-related exception */
  class DataException : public Exception { public: using Exception::Exception; };

  /** Thrown when attempting to use unused output */
  class UnusedOutput : public DataException { public: using DataException::DataException; };

  /** General function-related exception */
  class FunctionException : public Exception { public: using Exception::Exception; };

  /** Thrown when attempting to call an output without an implementation */
  class UnimplementedOutput : public FunctionException { public: using FunctionException::FunctionException; };

  /** Thrown when adding a variable already present */
  class DuplicateVariable : public FunctionException { public: using FunctionException::FunctionException; };

  /** Thrown when attempting to access a non-existing variable */
  class NonExistingVariable : public FunctionException { public: using FunctionException::FunctionException; };

  /** Thrown when attempting to use a feature that is not implemented in the
   * framework yet */
  class NotImplemented : public Exception { public: using Exception::Exception; };

}  // namespace exception

}  // namespace tvm
