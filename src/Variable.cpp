#include "Variable.h"

#include <sstream>

namespace taskvm
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



  std::shared_ptr<Variable> dot(std::shared_ptr<Variable> var, int ndiff)
  {
    assert(ndiff > 0 && "you cannot derive less than 1 time.");
    int i;
    std::shared_ptr<Variable> derivative = var;

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
        derivative = std::shared_ptr<Variable>(new Variable(derivative));
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

  int Variable::derivativeNumber() const
  {
    return derivativeNumber_;
  }

  bool Variable::isBasePrimitive() const
  {
    return derivativeNumber_ == 0;
  }

  std::shared_ptr<Variable> Variable::primitive() const
  {
    return primitive_;
  }

  std::shared_ptr<Variable> Variable::basePrimitive() const
  {
    const Variable* ptr = this;
    for (int i = 0; i < derivativeNumber_-1; ++i)
      ptr = ptr->primitive_.get();

    return ptr->primitive_;
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

  Variable::Variable(std::shared_ptr<Variable> var)
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