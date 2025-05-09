/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#include <tvm/task_dynamics/None.h>

#include <tvm/function/abstract/LinearFunction.h>

namespace tvm
{

namespace task_dynamics
{

std::unique_ptr<abstract::TaskDynamicsImpl> None::impl_(FunctionPtr f,
                                                        constraint::Type t,
                                                        const Eigen::VectorXd & rhs) const
{
  return std::make_unique<Impl>(f, t, rhs);
}

Order None::order_() const { return Order::Zero; }

None::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs)
: TaskDynamicsImpl(Order::Zero, f, t, rhs)
{
  using function::abstract::LinearFunction;
  lf_ = dynamic_cast<const LinearFunction *>(f.get());
  if(!lf_)
  {
    throw std::runtime_error("The function is not linear.");
  }
  addInputDependency<Impl>(Update::UpdateValue, std::static_pointer_cast<LinearFunction>(f), LinearFunction::Output::B);
}

void None::Impl::updateValue()
{
  const auto & b = lf_->b();
  if(rhs().size() != b.size())
  {
    if(rhs().isConstant(rhs()(0)))
    {
      value_ = Eigen::VectorXd::Constant(b.size(), rhs()(0)) - b;
    }
    else
    {
      throw std::runtime_error("Task dynamics None: can't resize a non constant rhs");
    }
  }
  else
  {
    value_ = rhs() - b;
  }
}

} // namespace task_dynamics

} // namespace tvm
