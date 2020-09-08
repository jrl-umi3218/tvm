/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

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
class DataException : public Exception
{
public:
  using Exception::Exception;
};

/** Thrown when attempting to use unused output */
class UnusedOutput : public DataException
{
public:
  using DataException::DataException;
};

/** General function-related exception */
class FunctionException : public Exception
{
public:
  using Exception::Exception;
};

/** Thrown when attempting to call an output without an implementation */
class UnimplementedOutput : public FunctionException
{
public:
  using FunctionException::FunctionException;
};

/** Thrown when adding a variable already present */
class DuplicateVariable : public FunctionException
{
public:
  using FunctionException::FunctionException;
};

/** Thrown when attempting to access a non-existing variable */
class NonExistingVariable : public FunctionException
{
public:
  using FunctionException::FunctionException;
};

/** Thrown when attempting to use a feature that is not implemented in the
 * framework yet */
class NotImplemented : public Exception
{
public:
  using Exception::Exception;
};

} // namespace exception

} // namespace tvm
