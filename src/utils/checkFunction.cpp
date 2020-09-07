/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <tvm/utils/checkFunction.h>

#include <tvm/Variable.h>
#include <tvm/utils/UpdatelessFunction.h>

#include <iostream>

using namespace Eigen;

namespace tvm
{

namespace utils
{
bool TVM_DLLAPI checkJacobian(FunctionPtr f, CheckOptions opt)
{
  using Output = tvm::function::abstract::Function::Output;
  if(!(f->isOutputEnabled(Output::Value) && f->isOutputEnabled(Output::Jacobian)))
  {
    throw std::runtime_error("The function must provide its value and jacobian for this test.");
  }

  const double h = opt.step;

  UpdatelessFunction uf(f);
  const auto & x = f->variables().variables();

  int n = 0;
  for(const auto & xi : x)
  {
    if(!xi->isEuclidean())
    {
      // For now, we only accept Euclidean variables. This could be extended to generic
      // manifolds if we get the information on how to make finite differences on the
      // non-Euclidean variables (i.e. if we get a retraction).
      // Note to developpers: ffd with manifolds can be found in PostureGenerator and
      // externally-provided retraction are used in GeometricFramework
      throw std::runtime_error("This function is implemented for Euclidean variables only.");
    }
    n += xi->size();
  }
  VectorXd v0 = VectorXd::Random(n);
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
  for(const auto & xi : x)
  {
    for(int j = 0; j < xi->size(); ++j)
    {
      v[i] += h;
      const VectorXd & f = uf.value(v);
      J.col(i) = (f - f0) / h;
      if(!J.col(i).isApprox(J0.col(i), opt.prec))
      {
        b = false;
        if(opt.verbose)
        {
          std::cout << "col(" << j << ") of the jacobian matrix associated to variable " << xi->name() << " (col(" << i
                    << ") of the total jacobian) is not equivalent to the result of the ffd." << std::endl;
        }
      }
      v[i] -= h;
      ++i;
    }
  }

  if(!b && opt.verbose)
  {
    std::cout << "got:\n" << J0 << std::endl;
    std::cout << "ffd:\n" << J << std::endl;
  }
  return b;
}

bool TVM_DLLAPI checkVelocity(FunctionPtr f, CheckOptions opt)
{
  using Output = tvm::function::abstract::Function::Output;
  if(!(f->isOutputEnabled(Output::Velocity) && f->isOutputEnabled(Output::Jacobian)))
  {
    throw std::runtime_error("The function must provide its velocity and jacobian for this test.");
  }

  UpdatelessFunction uf(f);
  const auto & x = f->variables().variables();

  int n = 0;
  int nd = 0;
  for(const auto & xi : x)
  {
    // We can handle non-Euclidean variables here
    n += xi->size();
    nd += dot(xi)->size();
  }
  VectorXd val = VectorXd::Random(n);
  VectorXd vel = VectorXd::Random(nd);

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
    }
    return false;
  }
}

bool TVM_DLLAPI checkNormalAcceleration(FunctionPtr f, CheckOptions opt)
{
  using Output = tvm::function::abstract::Function::Output;
  if(!(f->isOutputEnabled(Output::Velocity) && f->isOutputEnabled(Output::Jacobian)))
  {
    throw std::runtime_error("The function must provide its velocity and jacobian for this test.");
  }

  UpdatelessFunction uf(f);
  const auto & x = f->variables().variables();

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
  VectorXd val0 = VectorXd::Random(n);
  // value of dx(t)
  VectorXd vel0 = VectorXd::Random(nd);
  // value of ddx(t)
  VectorXd acc = VectorXd::Random(nd).normalized();

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

bool TVM_DLLAPI checkFunction(FunctionPtr f, CheckOptions opt)
{
  // Call the functions in order: checkVelocity will only be called if checkj\Jacobian
  // passes, which is good as checkVelocity relies on having correct jacobian matrices.
  // Likewise checkNormalAcceleration relies on having correct jacobian matrices and
  // velocities.
  return checkJacobian(f, opt) && checkVelocity(f, opt) && checkNormalAcceleration(f, opt);
}
} // namespace utils

} // namespace tvm