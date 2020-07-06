/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#include <tvm/task_dynamics/abstract/TaskDynamicsImpl.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

  namespace task_dynamics
  {

    namespace abstract
    {

      TaskDynamicsImpl::TaskDynamicsImpl(Order order, FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs)
        : order_(order)
        , type_(t)
        , rhs_(rhs)
      {
        if (f->size() != rhs.size())
        {
          throw std::runtime_error("[TaskDynamicsImpl] rhs does not have the same size as f.");
        }
        if (!f->imageSpace().isEuclidean())
          throw std::runtime_error("[TaskDynamicsImpl::TaskDynamicsImpl] Task dynamics are for function into a Euclidean space.");
        setFunction(f);
        registerUpdates(Update::UpdateValue, &TaskDynamicsImpl::updateValue);
        addOutputDependency(Output::Value, Update::UpdateValue);
      }

      void TaskDynamicsImpl::setFunction(FunctionPtr f)
      {
        if (f)
        {
          f_ = f;
          addInput(f, internal::FirstOrderProvider::Output::Value); //FIXME it's not great to have to resort to internal::FirstOrderProvider
          addInputDependency(Update::UpdateValue, f, internal::FirstOrderProvider::Output::Value);
          if (order_ == Order::Two)
          {
            addInput(f, function::abstract::Function::Output::Velocity);
            addInputDependency(Update::UpdateValue, f, function::abstract::Function::Output::Velocity);
          }
          value_.resize(f->size());
        }
        else
          throw std::runtime_error("You cannot pass a nullptr as a function.");
      }

    }  // namespace abstract

  }  // namespace task_dynamics

}  // namespace tvm
