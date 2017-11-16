#include <tvm/task_dynamics/None.h>

#include <tvm/function/abstract/LinearFunction.h>

namespace tvm
{

  namespace task_dynamics
  {

    None::None()
      : TaskDynamics(Order::Geometric)
    {
    }

    void None::updateValue()
    {
      value_ = -lf_->b();
    }

    void None::setFunction_()
    {
      lf_ = dynamic_cast<const function::abstract::LinearFunction*>(&function());
      if (!lf_)
      {
        throw std::runtime_error("The function is not linear.");
      }
    }

  }  // namespace task_dynamics

}  // namespace tvm