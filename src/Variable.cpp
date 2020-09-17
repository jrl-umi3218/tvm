/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Variable.h>

#include <tvm/VariableVector.h>

#include <sstream>

namespace tvm
{

VariablePtr dot(VariablePtr var, int ndiff)
{
  assert(ndiff > 0 && "you cannot derive less than 1 time.");
  int i;
  Variable * derivative = var.get();

  // find the ndiff-th derivative of var, or the largest i such that the i-th
  // derivative exists.
  for(i = 0; i < ndiff; ++i)
  {
    if(!derivative->derivative_)
      break;
    else
      derivative = derivative->derivative_.get();
  }

  if(i == ndiff) // the ndiff-th derivative already exists
    return {var, derivative};
  else // we need to create the derivatives from i+1 to ndiff
  {
    for(; i < ndiff; ++i)
    {
      auto primitive = derivative;
      primitive->derivative_.reset(new Variable(derivative));
      derivative = primitive->derivative_.get();
    }
    return {var, derivative};
  }
}

Variable::~Variable()
{
  if(!isSubvariable())
  {
    delete[] memory_;
  }
}

VariablePtr Variable::duplicate(std::string_view n) const
{
  VariablePtr newPrimitive;
  if(n.empty())
  {
    newPrimitive = space_.createVariable(basePrimitive()->name() + "'");
  }
  else
  {
    newPrimitive = space_.createVariable(n);
  }
  if(derivativeNumber_)
  {
    auto r = dot(newPrimitive, derivativeNumber_);
    r->value(value_);
    return r;
  }
  else
  {
    newPrimitive->value(value_);
    return newPrimitive;
  }
}

const std::string & Variable::name() const { return name_; }

int Variable::size() const { return static_cast<int>(value_.size()); }

const Space & Variable::space() const { return space_; }

const Space & Variable::spaceShift() const { return shift_; }

bool Variable::isEuclidean() const { return space_.isEuclidean() || !isBasePrimitive(); }

VectorConstRef Variable::value() const { return value_; }

void Variable::value(const VectorConstRef & x)
{
  if(x.size() == size())
    value_ = x;
  else
    throw std::runtime_error("x has not the correct size.");
}

void Variable::setZero() { value_.setZero(); }

int Variable::derivativeNumber() const { return derivativeNumber_; }

bool Variable::isDerivativeOf(const Variable & v) const
{
  return basePrimitive() == v.basePrimitive() && derivativeNumber_ > v.derivativeNumber();
}

bool Variable::isPrimitiveOf(const Variable & v) const
{
  return basePrimitive() == v.basePrimitive() && derivativeNumber_ < v.derivativeNumber();
}

bool Variable::isBasePrimitive() const { return derivativeNumber_ == 0; }

VariablePtr Variable::basePrimitive() const
{
  if(isBasePrimitive())
  {
    // while it does not seem ideal to cast the constness away, it is coherent
    // with the general case: from a const derived variable, we can get a non
    // const primitive. This is equivalent to have primitive_ be a shared_ptr
    // to this in the case of a base primitive, what we cannot obviously do.
    return const_cast<Variable *>(this)->shared_from_this();
  }
  else
  {
    const Variable * ptr = this;
    for(int i = 0; i < derivativeNumber_ - 1; ++i)
      ptr = ptr->primitive_;

    return ptr->primitive_->shared_from_this();
  }
}

bool Variable::isSubvariable() const { return superVariable_ != nullptr; }

VariablePtr Variable::superVariable() const
{
  if(superVariable_)
    return superVariable_->shared_from_this();
  else
    return const_cast<Variable *>(this)->shared_from_this();
}

VariablePtr Variable::subvariable(Space space, std::string_view baseName, Space shift) const
{
  if(!(space * shift <= space_))
    throw std::runtime_error("[Variable::subvariable] Invalid space and shift dimension");

  VariablePtr base = superVariable();
  if(isBasePrimitive())
  {
    return VariablePtr(base, new Variable(base.get(), space, baseName, shift_ * shift));
  }
  else
  {
    VariablePtr bp = base->basePrimitive();
    return dot(VariablePtr(bp, new Variable(bp.get(), space, baseName, shift_ * shift)), derivativeNumber_);
  }
}

Range Variable::getMappingIn(const VariableVector & variables) const
{
  if(mappingHelper_.stamp == variables.stamp())
    return {mappingHelper_.start, size()};
  else
  {
    if(variables.contains(*this))
    {
      variables.computeMapping();
      return {mappingHelper_.start, size()};
    }
    else
      throw std::runtime_error("This variable is not part of the vector of variables.");
  }
}

Variable::Variable(const Space & s, std::string_view name)
: name_(name), space_(s), shift_(0), memory_(new double[s.rSize()]),
  value_(Eigen::Map<Eigen::VectorXd>(memory_, s.rSize())), derivativeNumber_(0), primitive_(nullptr),
  superVariable_(nullptr), derivative_(nullptr), mappingHelper_()
{
  value_.setZero();
}

Variable::Variable(Variable * var)
: space_(var->space_), shift_(var->shift_), memory_(var->isSubvariable() ? nullptr : new double[var->space_.tSize()]),
  value_(Eigen::Map<Eigen::VectorXd>(var->isSubvariable() ? dot(var->superVariable())->memory_ + var->shift_.tSize()
                                                          : memory_,
                                     var->space_.tSize())),
  derivativeNumber_(var->derivativeNumber_ + 1), primitive_(var),
  superVariable_(var->isSubvariable() ? dot(var->superVariable()).get() : nullptr), derivative_(nullptr),
  mappingHelper_()
{
  std::stringstream ss;
  if(derivativeNumber_ == 1)
  {
    ss << "d " << basePrimitive()->name_ << " / dt";
  }
  else
  {
    ss << "d" << derivativeNumber_ << " " << basePrimitive()->name_ << " / dt" << derivativeNumber_;
  }
  name_ = ss.str();
  value_.setZero();
}

Variable::Variable(Variable * var, const Space & space, std::string_view name, const Space & shift)
: name_(name), space_(space), shift_(shift), memory_(nullptr),
  value_(Eigen::Map<Eigen::VectorXd>(var->memory_ + shift.rSize(), space.rSize())), derivativeNumber_(0),
  primitive_(nullptr), superVariable_(var), derivative_(nullptr), mappingHelper_()
{}

VariablePtr Variable::shared_from_this()
{
  if(isBasePrimitive())
  {
    if(isSubvariable())
      return {superVariable_->shared_from_this(), this};
    else
      return std::enable_shared_from_this<Variable>::shared_from_this();
  }
  else
  {
    if(isSubvariable())
      return {superVariable_->basePrimitive()->shared_from_this(), this};
    else
      return {basePrimitive()->shared_from_this(), this};
  }
}

} // namespace tvm
