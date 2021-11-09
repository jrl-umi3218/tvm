/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Variable.h>

#include <tvm/VariableVector.h>

#include <sstream>

namespace tvm
{

VariablePtr dot(VariablePtr var, int ndiff, bool autoName)
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
      primitive->derivative_.reset(new Variable(derivative, autoName));
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
    r->set(value_);
    return r;
  }
  else
  {
    newPrimitive->set(value_);
    return newPrimitive;
  }
}

const std::string & Variable::name() const { return name_; }

int Variable::size() const { return static_cast<int>(value_.size()); }

const Space & Variable::space() const { return space_; }

const Space & Variable::spaceShift() const { return shift_; }

bool Variable::isEuclidean() const { return space_.isEuclidean() || !isBasePrimitive(); }

VectorConstRef Variable::value() const { return value_; }

void Variable::set(const VectorConstRef & x)
{
  if(x.size() == size())
    value_ = x;
  else
    throw std::runtime_error("x has not the correct size.");
}

void Variable::set(Eigen::DenseIndex idx, double value)
{
  if(idx >= 0 && idx < size())
  {
    value_(idx) = value;
  }
  else
  {
    throw std::runtime_error("Variable assignment out of bounds");
  }
}

void Variable::set(Eigen::DenseIndex idx, Eigen::DenseIndex length, const VectorConstRef & value)
{
  if(idx >= 0 && idx + length <= size() && value.size() == length)
  {
    value_.segment(idx, length) = value;
  }
  else
  {
    throw std::runtime_error("Variable segment assignment out of bounds");
  }
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

bool Variable::isSuperVariable() const { return superVariable_ == nullptr; }

bool Variable::isSuperVariableOf(const Variable & v) const { return v.superVariable_.get() == this; }

Range Variable::subvariableRange() const
{
  if(isSuperVariable())
    return {0, size()};
  else
    return {static_cast<int>(value_.data() - superVariable_->value_.data()), size()};
}

bool Variable::contains(const Variable & v) const
{
  return superVariable() == v.superVariable() && subvariableRange().contains(v.subvariableRange());
}

bool Variable::intersects(const Variable & v) const
{
  return superVariable() == v.superVariable() && subvariableRange().intersects(v.subvariableRange());
}

VariablePtr Variable::subvariable(Space space, std::string_view baseName, Space shift) const
{
  return subvariable(space, baseName, shift, false);
}

VariablePtr Variable::subvariable(Space space, Space shift) const { return subvariable(space, "", shift, true); }

Range Variable::getMappingIn(const VariableVector & variables) const
{
  auto it = startIn_.find(variables.id());
  if(it != startIn_.end())
  {
    if(it->second.stamp == -1 || it->second.stamp == variables.stamp())
      return {it->second.start, size()};
  }

  return variables.getMappingOf(*this);
}

Variable::Variable(const Space & s, std::string_view name)
: name_(name), space_(s), shift_(0), memory_(new double[s.rSize()]),
  value_(Eigen::Map<Eigen::VectorXd>(memory_, s.rSize())), derivativeNumber_(0), primitive_(nullptr),
  superVariable_(nullptr), derivative_(nullptr), startIn_()
{
  value_.setZero();
}

Variable::Variable(Variable * var, bool autoName)
: space_(var->space_), shift_(var->shift_), memory_(var->isSubvariable() ? nullptr : new double[var->space_.tSize()]),
  value_(Eigen::Map<Eigen::VectorXd>(var->isSubvariable() ? dot(var->superVariable())->memory_ + var->shift_.tSize()
                                                          : memory_,
                                     var->space_.tSize())),
  derivativeNumber_(var->derivativeNumber_ + 1), primitive_(var),
  superVariable_(var->isSubvariable() ? dot(var->superVariable()) : nullptr), derivative_(nullptr), startIn_()
{
  std::stringstream ss;
  auto dNumber = derivativeNumber_ == 1 ? "" : std::to_string(derivativeNumber_);
  auto baseName = autoName ? basePrimitive()->superVariable()->name_ : basePrimitive()->name_;
  ss << "d" << dNumber << " " << baseName << " / dt" << dNumber;
  if(autoName)
  {
    ss << "[" << shift_.tSize() << ":" << shift_.tSize() + size() - 1 << "]";
  }
  name_ = ss.str();
  value_.setZero();
}

VariablePtr Variable::subvariable(Space space, std::string_view baseName, Space shift, bool autoName) const
{
  if(!(space * shift <= space_))
    throw std::runtime_error("[Variable::subvariable] Invalid space and shift dimension");

  VariablePtr base = superVariable()->basePrimitive();
  VariablePtr sub;

  // Create a subvariable of the supervariable base primitive
  if(autoName)
  {
    int start = base->spaceShift().rSize() + shift.rSize();
    std::stringstream ss;
    ss << base->name() << "[" << start << ":" << start + space.rSize() - 1 << "]";
    sub = std::make_shared<Variable>(make_shared_token{}, base, space, ss.str(), shift_ * shift);
  }
  else
    sub = std::make_shared<Variable>(make_shared_token{}, base, space, baseName, shift_ * shift);

  // Derive if necessary
  if(isBasePrimitive())
    return sub;
  else
    return dot(sub, derivativeNumber_, autoName);
}

Variable::Variable(make_shared_token, VariablePtr var, const Space & space, std::string_view name, const Space & shift)
: name_(name), space_(space), shift_(shift), memory_(nullptr),
  value_(Eigen::Map<Eigen::VectorXd>(var->memory_ + shift.rSize(), space.rSize())), derivativeNumber_(0),
  primitive_(nullptr), superVariable_(var), derivative_(nullptr), startIn_()
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
