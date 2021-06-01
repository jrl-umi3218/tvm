/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/utils/checkFunction.h>

#include <tvm/Variable.h>
#include <tvm/utils/UpdatelessFunction.h>

#include <iostream>

using namespace Eigen;

namespace tvm
{

namespace utils
{

namespace
{

inline bool checkSize(const char * prefix, const char * type, const Eigen::VectorXd & ref, const Eigen::VectorXd & v)
{
  if(v.size() != ref.size())
  {
    std::cout << prefix << " Provided variable " << type << " does not have the correct size\n";
    return false;
  }
  return true;
}

template<auto Member>
const char * MemberName()
{
  using CheckConfiguration = CheckOptions::CheckConfiguration;
  if constexpr(Member == &CheckConfiguration::value)
  {
    return "value";
  }
  else if constexpr(Member == &CheckConfiguration::velocity)
  {
    return "velocity";
  }
  else
  {
    return "acceleration";
  }
}

template<auto Member>
bool getFromConfigOrRandom(const char * prefix, const CheckOptions::CheckConfiguration & c, Eigen::VectorXd & out)
{
  if(c.*Member)
  {
    const auto & v = *(c.*Member);
    if(!checkSize(prefix, MemberName<Member>(), out, v))
    {
      return false;
    }
    out = v;
  }
  else
  {
    out.setRandom();
    if constexpr(Member == &CheckOptions::CheckConfiguration::acceleration)
    {
      out.normalize();
    }
  }
  return true;
}

} // namespace

namespace
{

bool checkJacobian(const UpdatelessFunction & uf, const CheckOptions & opt, const Eigen::VectorXd & v0)
{
  const auto & x = uf.variables();
  const double h = opt.step;
  auto n = v0.size();
  VectorXd v = v0;
  VectorXd f0 = uf.value(v0);
  MatrixXd J(f0.size(), n);
  MatrixXd J0(f0.size(), n);

  int s = 0;
  for(const auto & xi : x)
  {
    int ni = xi->size();
    J0.middleCols(s, ni) = uf.jacobian(*xi, v0);
    s += ni;
  }

  bool b = true;
  int i = 0;
  double norm2 = 0;
  for(const auto & xi : x)
  {
    for(int j = 0; j < xi->size(); ++j)
    {
      v[i] += h;
      const VectorXd & f = uf.value(v);
      J.col(i) = (f - f0) / h;
      norm2 = std::max(norm2, std::min(J.col(i).cwiseAbs2().sum(), J0.col(i).cwiseAbs2().sum()));
      v[i] -= h;
      ++i;
    }
  }
  // To compare the two matrices column by column, we run a variant of isApprox, where the scale
  // is given by the two matrices rather than just the two columns.
  i = 0;
  for(const auto & xi : x)
  {
    for(int j = 0; j < xi->size(); ++j)
    {
      if((J.col(i) - J0.col(i)).cwiseAbs2().sum() > opt.prec * opt.prec * norm2)
      {
        b = false;
        if(opt.verbose)
        {
          std::cout << "col(" << j << ") of the jacobian matrix associated to variable " << xi->name() << " (col(" << i
                    << ") of the total jacobian) is not equivalent to the result of the ffd." << std::endl;
        }
      }
    }
  }

  if(!b && opt.verbose)
  {
    std::cout << "got:\n" << J0 << std::endl;
    std::cout << "ffd:\n" << J << std::endl;
    std::cout << "v0: " << v0.transpose() << "\n";
  }
  return b;
}

bool checkJacobian(const UpdatelessFunction & uf, const CheckOptions & opt)
{

  const auto & x = uf.variables().variables();

  int n = 0;
  for(const auto & xi : x)
  {
    if(!xi->isEuclidean())
    {
      // For now, we only accept Euclidean variables. This could be extended to generic
      // manifolds if we get the information on how to make finite differences on the
      // non-Euclidean variables (i.e. if we get a retraction).
      // Note to developers: ffd with manifolds can be found in PostureGenerator and
      // externally-provided retraction are used in GeometricFramework
      throw std::runtime_error("This function is implemented for Euclidean variables only.");
    }
    n += xi->size();
  }
  VectorXd v0 = VectorXd::Zero(n);
  size_t failed = 0;
  for(size_t i = 0; i < opt.samples; ++i)
  {
    v0.setRandom();
    failed += !checkJacobian(uf, opt, v0);
  }
  size_t failed_fixed = 0;
  size_t total_fixed = 0;
  for(const auto & c : opt.configs)
  {
    if(!c.value)
    {
      continue;
    }
    if(c.value->size() != v0.size())
    {
      std::cout << "[checkJacobian] Provided variable value does not have the correct size\n";
      continue;
    }
    total_fixed++;
    failed_fixed += !checkJacobian(uf, opt, *c.value);
  }
  if(opt.verbose && (failed != 0 || failed_fixed != 0))
  {
    if(opt.samples != 0 && failed != 0)
    {
      std::cout << failed << " random configurations failed out of " << opt.samples << "\n";
    }
    if(total_fixed != 0 && failed_fixed != 0)
    {
      std::cout << failed_fixed << " configurations failed out of " << total_fixed << "\n";
    }
  }
  return (failed + failed_fixed) == 0;
}

} // namespace

bool checkJacobian(FunctionPtr f, CheckOptions opt)
{
  using Output = tvm::function::abstract::Function::Output;
  if(!(f->isOutputEnabled(Output::Value) && f->isOutputEnabled(Output::Jacobian)))
  {
    throw std::runtime_error("The function must provide its value and jacobian for this test.");
  }

  UpdatelessFunction uf(f);
  return checkJacobian(uf, opt);
}

namespace
{

bool checkVelocity(const UpdatelessFunction & uf,
                   const CheckOptions & opt,
                   const Eigen::VectorXd & val,
                   const Eigen::VectorXd & vel)
{
  const auto & x = uf.variables();
  const auto & f = uf.function();
  VectorXd v0 = uf.velocity(val, vel);

  VectorXd v = VectorXd::Zero(v0.size());
  // Force computation of jacobian matrices
  uf.jacobian(*x[0], val);

  // sum df/dxi * dxi/dt
  for(const auto & xi : x)
  {
    v += f->jacobian(*xi) * dot(xi)->value();
  }

  if(v.isApprox(v0, opt.prec))
  {
    return true;
  }
  else
  {
    if(opt.verbose)
    {
      std::cout << "Incorrect velocity (or jacobian if you did not check it):\n"
                << "got\t\t" << v0.transpose() << "\nexpected\t" << v.transpose() << std::endl;
      std::cout << "For value: " << val.transpose() << "\n";
      std::cout << "And velocity: " << vel.transpose() << "\n";
    }
    return false;
  }
}

bool checkVelocity(const UpdatelessFunction & uf, const CheckOptions & opt)
{
  const auto & x = uf.variables();

  int n = 0;
  int nd = 0;
  for(const auto & xi : x)
  {
    // We can handle non-Euclidean variables here
    n += xi->size();
    nd += dot(xi)->size();
  }
  VectorXd val = VectorXd::Zero(n);
  VectorXd vel = VectorXd::Zero(nd);
  size_t failed = 0;
  for(size_t i = 0; i < opt.samples; ++i)
  {
    val.setRandom();
    vel.setRandom();
    failed += !checkVelocity(uf, opt, val, vel);
  }
  size_t failed_fixed = 0;
  size_t total_fixed = 0;
  using CheckConfiguration = CheckOptions::CheckConfiguration;
  const char * prefix = "[checkVelocity]";
  auto getValue = [&](const CheckConfiguration & c) {
    return getFromConfigOrRandom<&CheckConfiguration::value>(prefix, c, val);
  };
  auto getVelocity = [&](const CheckConfiguration & c) {
    return getFromConfigOrRandom<&CheckConfiguration::velocity>(prefix, c, vel);
  };
  for(const auto & c : opt.configs)
  {
    if(c.value && c.velocity)
    {
      if(getValue(c) && getVelocity(c))
      {
        total_fixed++;
        failed_fixed += checkVelocity(uf, opt, val, vel);
      }
    }
    else
    {
      for(size_t i = 0; i < c.samples; ++i)
      {
        if(getValue(c) && getVelocity(c))
        {
          total_fixed++;
          failed_fixed += checkVelocity(uf, opt, val, vel);
        }
      }
    }
  }
  if(opt.verbose && (failed != 0 || failed_fixed != 0))
  {
    if(opt.samples != 0 && failed != 0)
    {
      std::cout << failed << " random configurations failed out of " << opt.samples << "\n";
    }
    if(total_fixed != 0 && failed_fixed)
    {
      std::cout << failed_fixed << " configurations failed out of " << total_fixed << "\n";
    }
  }
  return (failed + failed_fixed) == 0;
}

} // namespace

bool checkVelocity(FunctionPtr f, CheckOptions opt)
{
  using Output = tvm::function::abstract::Function::Output;
  if(!(f->isOutputEnabled(Output::Velocity) && f->isOutputEnabled(Output::Jacobian)))
  {
    throw std::runtime_error("The function must provide its velocity and jacobian for this test.");
  }

  UpdatelessFunction uf(f);
  return checkVelocity(uf, opt);
}

namespace
{

bool checkNormalAcceleration(const UpdatelessFunction & uf,
                             const CheckOptions & opt,
                             const Eigen::VectorXd & val0,
                             const Eigen::VectorXd & vel0,
                             const Eigen::VectorXd & acc)
{
  const auto & x = uf.variables();
  const auto & f = uf.function();
  // we consider a constant-acceleration trajectory in variable space
  double dt = opt.step;
  VectorXd vel1 = vel0 + acc * dt;
  VectorXd val1 = val0 + vel0 * dt + 0.5 * acc.cwiseProduct(acc) * dt * dt;

  VectorXd v0 = uf.velocity(val0, vel0);
  VectorXd v1 = uf.velocity(val1, vel1);
  // Approximation of d2f/dt2
  VectorXd a = (v1 - v0) / dt;

  // Force computation of jacobian matrices
  uf.jacobian(*x[0], val0);

  // Compute (d2f/dt2-J*dx2/dt2)
  int s = 0;
  for(const auto & xi : x)
  {
    int ni = xi->size();
    a -= f->jacobian(*xi) * acc.segment(s, ni);
    s += ni;
  }

  const VectorXd & na = uf.normalAcceleration(val0, vel0);
  if(na.isApprox(a, opt.prec))
  {
    return true;
  }
  else
  {
    if(opt.verbose)
    {
      std::cout << "Incorrect normal acceleration:\n"
                << "got\t\t" << na.transpose() << "\nexpected\t" << a.transpose() << std::endl;
    }
    return false;
  }
}

bool checkNormalAcceleration(const UpdatelessFunction & uf, const CheckOptions & opt)
{
  const auto & x = uf.variables();

  int n = 0;
  int nd = 0;
  for(const auto & xi : x)
  {
    if(!xi->isEuclidean())
    {
      // For now, we only accept Euclidean variables. See note in checkJacobian's code.
      throw std::runtime_error("This function is implemented for Euclidean variables only.");
    }
    n += xi->size();
    nd += dot(xi)->size();
  }

  // value of x(t)
  VectorXd val0 = VectorXd::Zero(n);
  // value of dx(t)
  VectorXd vel0 = VectorXd::Zero(nd);
  // value of ddx(t)
  VectorXd acc = VectorXd::Zero(nd);

  size_t failed = 0;
  for(size_t i = 0; i < opt.samples; ++i)
  {
    val0.setRandom();
    vel0.setRandom();
    acc.setRandom().normalize();
    failed += !checkNormalAcceleration(uf, opt, val0, vel0, acc);
  }
  size_t failed_fixed = 0;
  size_t total_fixed = 0;
  using CheckConfiguration = CheckOptions::CheckConfiguration;
  const char * prefix = "[checkNormalAcceleration]";
  auto getValue = [&](const CheckConfiguration & c) {
    return getFromConfigOrRandom<&CheckConfiguration::value>(prefix, c, val0);
  };
  auto getVelocity = [&](const CheckConfiguration & c) {
    return getFromConfigOrRandom<&CheckConfiguration::velocity>(prefix, c, vel0);
  };
  auto getAcceleration = [&](const CheckConfiguration & c) {
    return getFromConfigOrRandom<&CheckConfiguration::acceleration>(prefix, c, acc);
  };
  auto handleConfiguration = [&](const CheckOptions::CheckConfiguration & c) {
    if(c.value && c.velocity && c.acceleration)
    {
      if(getValue(c) && getVelocity(c) && getAcceleration(c))
      {
        total_fixed++;
        failed_fixed += !checkNormalAcceleration(uf, opt, *c.value, *c.velocity, *c.acceleration);
      }
    }
    else
    {
      for(size_t i = 0; i < c.samples; ++i)
      {
        if(getValue(c) && getVelocity(c) && getAcceleration(c))
        {
          total_fixed++;
          failed_fixed += !checkNormalAcceleration(uf, opt, val0, vel0, acc);
        }
      }
    }
  };
  for(const auto & c : opt.configs)
  {
    handleConfiguration(c);
  }
  if(opt.verbose && (failed != 0 || failed_fixed != 0))
  {
    if(opt.samples != 0 && failed != 0)
    {
      std::cout << failed << " random configurations failed out of " << opt.samples << "\n";
    }
    if(total_fixed != 0 && failed_fixed != 0)
    {
      std::cout << failed_fixed << " configurations failed out of " << total_fixed << "\n";
    }
  }
  return (failed + failed_fixed) == 0;
}

} // namespace

bool checkNormalAcceleration(FunctionPtr f, CheckOptions opt)
{
  using Output = tvm::function::abstract::Function::Output;
  if(!(f->isOutputEnabled(Output::NormalAcceleration) && f->isOutputEnabled(Output::Velocity)
       && f->isOutputEnabled(Output::Jacobian)))
  {
    throw std::runtime_error("The function must provide its normal acceleration, velocity and jacobian for this test.");
  }

  UpdatelessFunction uf(f);
  return checkNormalAcceleration(uf, opt);
}

bool checkFunction(FunctionPtr f, CheckOptions opt)
{
  using Output = tvm::function::abstract::Function::Output;
  if(!(f->isOutputEnabled(Output::NormalAcceleration) && f->isOutputEnabled(Output::Velocity)
       && f->isOutputEnabled(Output::Jacobian) && f->isOutputEnabled(Output::Value)))
  {
    throw std::runtime_error(
        "The function must provide its normal acceleration, velocity, value and jacobian for this test.");
  }
  UpdatelessFunction uf(f);
  // Call the functions in order: checkVelocity will only be called if checkJacobian
  // passes, which is good as checkVelocity relies on having correct jacobian matrices.
  // Likewise checkNormalAcceleration relies on having correct jacobian matrices and
  // velocities.
  return checkJacobian(uf, opt) && checkVelocity(uf, opt) && checkNormalAcceleration(uf, opt);
}
} // namespace utils

} // namespace tvm
