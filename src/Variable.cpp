#include "Variable.h"
#include "VariableVector.h"

#include <sstream>

namespace tvm
{ 
  Space::Space(int size)
    : Space(size, size, size)
  {
  }

  Space::Space(int size, int representationSize)
    : Space(size, representationSize, size)
  {
  }

  Space::Space(int size, int representationSize, int tangentRepresentationSize)
    : mSize_(size), rSize_(representationSize), tSize_(tangentRepresentationSize)
  {
  }

  std::unique_ptr<Variable> Space::createVariable(const std::string& name) const
  {
    return std::unique_ptr<Variable>(new Variable(*this, name));
  }

  int Space::size() const
  {
    return mSize_;
  }

  int Space::rSize() const
  {
    return rSize_;
  }

  int Space::tSize() const
  {
    return tSize_;
  }

  bool Space::isEuclidean() const
  {
    return mSize_ == rSize_;
  }


  VariablePtr dot(VariablePtr var, int ndiff)
  {
    assert(ndiff > 0 && "you cannot derive less than 1 time.");
    int i;
    VariablePtr derivative = var;

    //find the ndiff-th derivative of var, or the largest i such that the i-th
    //derivative exists.
    for (i = 0; i < ndiff; ++i)
    {
      if (derivative->derivative_.expired())
        break;
      else
        derivative = derivative->derivative_.lock();
    }

    if (i == ndiff)                 //the ndiff-th derivative already exists
      return derivative;
    else                            //we need to create the derivatives from i+1 to ndiff
    {
      for (; i < ndiff; ++i)
      {
        auto primitive = derivative;
        derivative = VariablePtr(new Variable(derivative));
        primitive->derivative_ = derivative;
      }
      return derivative;
    }
  }



  const std::string & Variable::name() const
  {
    return name_;
  }

  int Variable::size() const
  {
    return static_cast<int>(value_.size());
  }

  const Space & Variable::space() const
  {
    return space_;
  }

  const Eigen::VectorXd & Variable::value() const
  {
    return value_;
  }

  void Variable::value(const VectorConstRef& x)
  {
    if (x.size() == size())
      value_ = x;
    else
      throw std::runtime_error("x has not the correct size.");
  }

  int Variable::derivativeNumber() const
  {
    return derivativeNumber_;
  }

  bool Variable::isBasePrimitive() const
  {
    return derivativeNumber_ == 0;
  }

  VariablePtr Variable::basePrimitive() const
  {
    const Variable* ptr = this;
    for (int i = 0; i < derivativeNumber_-1; ++i)
      ptr = ptr->primitive_.get();

    return ptr->primitive_;
  }

  Range Variable::getMappingIn(const VariableVector& variables) const
  {
    if (mappingHelper_.stamp == variables.stamp())
      return{ mappingHelper_.start, size() };
    else
    {
      if (variables.contains(*this))
      {
        variables.computeMapping();
        return{ mappingHelper_.start, size() };
      }
      else
        throw std::runtime_error("This variables is not part of the vector of variables.");
    }
  }

  Variable::Variable(const Space & s, const std::string & name)
    : name_(name)
    , space_(s)
    , value_(s.rSize())
    , derivativeNumber_(0)
    , primitive_(nullptr)
    , derivative_()
  {
  }

  Variable::Variable(VariablePtr var)
    : space_(var->space_)
    , value_(var->space_.tSize())
    , derivativeNumber_(var->derivativeNumber_ + 1)
    , primitive_(var)
    , derivative_()
  {
    std::stringstream ss;
    ss << "d" << derivativeNumber_ << " " << basePrimitive()->name_ << " / dt" << derivativeNumber_;
    name_ = ss.str();
  }

}