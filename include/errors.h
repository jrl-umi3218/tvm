#pragma once

#include <exception>

namespace tvm
{
  class DataError : public std::exception
  {

  };

  class UnusedOutput : public DataError
  {

  };




  class FunctionError : public std::exception
  {

  };

  class UnimplementedOutput : public FunctionError
  {

  };

  // when adding a variable already present
  class DuplicateVariable : public FunctionError
  {

  };

  // when asking for a variable that is not a variable of the function
  class NonExistingVariable : public FunctionError
  {

  };
}