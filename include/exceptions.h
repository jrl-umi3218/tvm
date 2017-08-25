#pragma once

#include <exception>

namespace tvm
{
  class tvmException : public std::exception
  {

  };


  class DataException : public tvmException
  {

  };

  class UnusedOutput : public DataException
  {

  };




  class FunctionException : public tvmException
  {

  };

  class UnimplementedOutput : public FunctionException
  {

  };

  // when adding a variable already present
  class DuplicateVariable : public FunctionException
  {

  };

  // when asking for a variable that is not a variable of the function
  class NonExistingVariable : public FunctionException
  {

  };
}