#include <tvm/utils/UpdatelessFunction.h>

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

    for (const auto& x : f_->variables())
    {
      dx_.push_back(dot(x));
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

  void UpdatelessFunction::assign(size_t i, const Eigen::VectorXd & val, bool value) const
  {
    const auto& x = f_->variables().variables();
    if (i >= x.size())
    {
      std::stringstream s;
      s << "Too many values provided (got " << i << ", expected " << x.size() << ")." << std::endl;
      throw std::runtime_error(s.str());
    }
    if (value)
    {
      if (val.size() == x[i]->size())
      {
        x[i]->value(val);
      }
      else
      {
        std::stringstream s;
        s << "The size provided for the " << i << "-th variable's value (" << x[i]->name()
          << ") is incorrect (got" << val.size() << ", expected " << x[i]->size() << ")." << std::endl;
        throw std::runtime_error("");
      }
    }
    else
    {
      auto dxi = dot(x[i]);
      if (val.size() == dxi->size())
      {
        dxi->value(val);
      }
      else
      {
        std::stringstream s;
        s << "The size provided for the " << i << "-th variable's velocity (" << x[i]->name()
          << ") is incorrect (got" << val.size() << ", expected " << dxi->size() << ")." << std::endl;
        throw std::runtime_error("");
      }
    }
  }

  void UpdatelessFunction::assign(Variable& x, const Eigen::VectorXd & val, bool value) const
  {
    const auto& vars = f_->variables().variables();
    if (vars.size() == 0)
    {
      throw std::runtime_error("This function has no variable.");
    }

    auto ptr = vars[0];
    VariablePtr p(ptr, &x);
    auto it = std::find(vars.begin(), vars.end(), p);
    if (it != vars.end())
    {
      if (value)
      {
        if (val.size() == x.size())
        {
          x.value(val);
        }
        else
        {
          std::stringstream s;
          s << "The size provided for the value of variable " << x.name() << "is incorrect (got" 
            << val.size() << ", expected " << x.size() << ")." << std::endl;
          throw std::runtime_error("");
        }
      }
      else
      {
        auto dxi = dot(*it);
        if (val.size() == dxi->size())
        {
          dxi->value(val);
        }
        else
        {
          std::stringstream s;
          s << "The size provided for the velocity of variable " << x.name() << "is incorrect (got"
            << val.size() << ", expected " << dxi->size() << ")." << std::endl;
          throw std::runtime_error("");
        }
      }
    }
    else
    {
      throw std::runtime_error("This function does not depend on variable " + x.name());
    }
  }

  void UpdatelessFunction::assign(const Eigen::VectorXd & val) const
  {
    const auto& x = f_->variables().variables();

    auto s = x[0]->size();
    for (size_t i = 1; i < x.size(); ++i)
    {
      s += x[i]->size();
    }
    if (val.size() != s)
    {
      throw std::runtime_error("The length of the concatenated vector does not correspond to the total length of the variables.");
    }

    s = 0;
    for (size_t i = 0; i < x.size(); ++i)
    {
      auto ni = x[i]->size();
      x[i]->value(val.segment(s, ni));
      s += ni;
    }
  }

  void UpdatelessFunction::assign(const Eigen::VectorXd & val, const Eigen::VectorXd & vel) const
  {
    const auto& x = f_->variables().variables();

    auto sp = x[0]->size();
    auto sd = dx_[0]->size();
    for (size_t i = 1; i < x.size(); ++i)
    {
      sp += x[i]->size();
      sd += dx_[i]->size();
    }
    if (val.size() != sp)
    {
      throw std::runtime_error("The length of the concatenated value vector does not correspond to the total length of the variables.");
    }
    if (vel.size() != sd)
    {
      throw std::runtime_error("The length of the concatenated vector velocity does not correspond to the total length of the variables derivatives.");
    }

    sp = 0;
    sd = 0;
    for (size_t i = 0; i < x.size(); ++i)
    {
      auto spi = x[i]->size();
      auto sdi = dx_[i]->size();
      x[i]->value(val.segment(sp, spi));
      dx_[i]->value(vel.segment(sd, sdi));
      sp += spi;
      sd += sdi;
    }
  }

  void UpdatelessFunction::parseValues_(int i, const Eigen::VectorXd & v) const
  {
    const auto& x = f_->variables().variables();
    if (i + 1 == static_cast<int>(x.size()))
    {
      assign(i, v, true);
    }
    else
    {
      if (i < static_cast<int>(x.size()))
      {
        if (i == 0) //case of concatenated values and velocities
        {
          assign(v);
        }
        else
        {
          std::stringstream s;
          s << "Too few values provided (got " << i << ", expected " << x.size() << ")." << std::endl;
          throw std::runtime_error(s.str());
        }
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
    assign(x, v, true);
  }

  void UpdatelessFunction::parseValues_(Variable & x, std::initializer_list<double> v) const
  {
    parseValues_(x, toVec(v));
  }

  void UpdatelessFunction::parseValuesAndVelocities_(int i, const Eigen::VectorXd & val, const Eigen::VectorXd & vel) const
  {
    const auto& x = f_->variables().variables();
    if (i + 1 == static_cast<int>(x.size()))
    {
      assign(i, val, true);
      assign(i, vel, false);
    }
    else
    {
      if (i < static_cast<int>(x.size()))
      {
        if (i == 0) //case of concatenated values and velocities
        {
          assign(val, vel);
        }
        else
        {
          std::stringstream s;
          s << "Too few values provided (got " << i << " pairs value/velocity, expected " << x.size() << ")." << std::endl;
          throw std::runtime_error(s.str());
        }
      }
      else
      {
        std::stringstream s;
        s << "Too many values provided (got " << i << " pairs value/velocity, expected " << x.size() << ")." << std::endl;
        throw std::runtime_error(s.str());
      }
    }
  }

  void UpdatelessFunction::parseValuesAndVelocities_(int i, const Eigen::VectorXd & val, std::initializer_list<double> vel) const
  {
    parseValuesAndVelocities_(i, val, toVec(vel));
  }

  void UpdatelessFunction::parseValuesAndVelocities_(int i, std::initializer_list<double> val, const Eigen::VectorXd & vel) const
  {
    parseValuesAndVelocities_(i, toVec(val), vel);
  }

  void UpdatelessFunction::parseValuesAndVelocities_(int i, std::initializer_list<double> val, std::initializer_list<double> vel) const
  {
    parseValuesAndVelocities_(i, toVec(val), toVec(vel));
  }

  void UpdatelessFunction::parseValuesAndVelocities_(Variable & x, const Eigen::VectorXd & val, const Eigen::VectorXd & vel) const
  {
    assign(x, val, true);
    assign(x, vel, false);
  }

  void UpdatelessFunction::parseValuesAndVelocities_(Variable & x, const Eigen::VectorXd & val, std::initializer_list<double> vel) const
  {
    parseValuesAndVelocities_(x, val, toVec(vel));
  }

  void UpdatelessFunction::parseValuesAndVelocities_(Variable & x, std::initializer_list<double> val, const Eigen::VectorXd & vel) const
  {
    parseValuesAndVelocities_(x, toVec(val), vel);
  }

  void UpdatelessFunction::parseValuesAndVelocities_(Variable & x, std::initializer_list<double> val, std::initializer_list<double> vel) const
  {
    parseValuesAndVelocities_(x, toVec(val), toVec(vel));
  }

} //namespace utils

} // namespace tvm
