#include <tvm/utils/UpdateLessFunction.h>

#include <tvm/Variable.h>

#include <sstream>
#include <string>

namespace tvm
{

namespace utils
{
  UpdatelessFunction::UpdatelessFunction(FunctionPtr f)
    : f_(f)
  {
    using Output = tvm::function::abstract::Function::Output;
    if (f->isOutputEnabled(Output::Value))
    {
      auto valueUser = std::make_shared<graph::internal::Inputs>();
      valueUser->addInput(f, Output::Value);
      valueGraph_.add(valueUser);
      valueGraph_.update();
    }

    if (f->isOutputEnabled(Output::Jacobian))
    {
      auto jacobianUser = std::make_shared<graph::internal::Inputs>();
      jacobianUser->addInput(f, Output::Jacobian);
      jacobianGraph_.add(jacobianUser);
      jacobianGraph_.update();
    }

    if (f->isOutputEnabled(Output::Velocity))
    {
      auto velocityUser = std::make_shared<graph::internal::Inputs>();
      velocityUser->addInput(f, Output::Velocity);
      velocityGraph_.add(velocityUser);
      velocityGraph_.update();
    }

    if (f->isOutputEnabled(Output::NormalAcceleration))
    {
      auto normalAccelerationUser = std::make_shared<graph::internal::Inputs>();
      normalAccelerationUser->addInput(f, Output::NormalAcceleration);
      normalAccelerationGraph_.add(normalAccelerationUser);
      normalAccelerationGraph_.update();
    }

    if (f->isOutputEnabled(Output::JDot))
    {
      auto JDotUser = std::make_shared<graph::internal::Inputs>();
      JDotUser->addInput(f, Output::JDot);
      JDotGraph_.add(JDotUser);
      JDotGraph_.update();
    }
  }

  Eigen::VectorXd UpdatelessFunction::toVec(std::initializer_list<double> val)
  {
    Eigen::VectorXd v(static_cast<int>(val.size()));
    Eigen::DenseIndex k = 0;
    for (auto d : val)
    {
      v[k] = d;
      ++k;
    }
    return v;
  }

  void UpdatelessFunction::assign(size_t i, const Eigen::VectorXd & val) const
  {
    const auto& x = f_->variables();
    if (i >= static_cast<int>(x.size()))
    {
      std::stringstream s;
      s << "Too many values provided (got " << i << ", expected " << x.size() << ")." << std::endl;
      throw std::runtime_error(s.str());
    }
    if (val.size() == x[i]->size())
    {
      x[i]->value(val);
    }
    else
    {
      std::stringstream s;
      s << "The size provided for the " << i << "-th variable (" << x[i]->name()
        << ") is incorrect (got" << val.size() << ", expected " << x[i]->size() << ")." << std::endl;
      throw std::runtime_error("");
    }
  }

  void UpdatelessFunction::assign(Variable& x, const Eigen::VectorXd & val) const
  {
    const auto& vars = f_->variables();
    if (vars.size() == 0)
    {
      throw std::runtime_error("This function has no variable.");
    }

    auto ptr = vars[0];
    VariablePtr p(ptr, &x);
    auto it = std::find(vars.begin(), vars.end(), p);
    if (it != vars.end())
    {
      x.value(val);
    }
    else
    {
      throw std::runtime_error("This function does not depend on variable " + x.name());
    }
  }

  void UpdatelessFunction::parseValues_(int i, const Eigen::VectorXd & v) const
  {
    const auto& x = f_->variables();
    if (i + 1 == static_cast<int>(x.size()))
    {
      if (v.size() == x[i]->size())
      {
        x[i]->value(v);
      }
      else
      {
        std::stringstream s;
        s << "The size provided for the " << i << "-th variable (" << x[i]->name()
          << ") is incorrect (got" << v.size() << ", expected " << x[i]->size() << ")." << std::endl;
        throw std::runtime_error(s.str());
      }
    }
    else
    {
      if (i < static_cast<int>(x.size()))
      {
        std::stringstream s;
        s << "Too few values provided (got " << i << ", expected " << x.size() << ")." << std::endl;
        throw std::runtime_error(s.str());
      }
      else
      {
        std::stringstream s;
        s << "Too many values provided (got " << i << ", expected " << x.size() << ")." << std::endl;
        throw std::runtime_error(s.str());
      }
    }
  }

  void UpdatelessFunction::parseValues_(int i, std::initializer_list<double> v) const
  {
    parseValues_(i, toVec(v));
  }

  void UpdatelessFunction::parseValues_(Variable & x, const Eigen::VectorXd & v) const
  {
    assign(x, v);
  }

  void UpdatelessFunction::parseValues_(Variable & x, std::initializer_list<double> v) const
  {
    parseValues_(x, toVec(v));
  }

} //namespace utils

} // namespace tvm
