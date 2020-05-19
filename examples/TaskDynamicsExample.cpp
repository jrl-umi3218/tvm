/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
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


/**
 * \file
 * \brief How to create a new task dynamics
 *
 * Preliminaries
 * =============
 *
 * Given a task \f$ f \circ rhs \f$ where \f$ f \f$ is a function, \f$ \circ \f$
 * is one of the following operators: \f$ \leq, = , \geq \f$ and \f$ rhs \f$ is
 * vector, we define the task error function \f$ e = f - rhs \f$. <br>
 * The goal of a task dynamics is to specify the desired value of one time
 * derivative of \f$ e \f$, such that the value of \f$ f \f$ stays in a region
 * that fulfills the task or converge to it.
 *
 * In this example we will see how to implement the task dynamics
 *
 * \f$
 *     \dot{e}^* = -k_p(e) e                           \qquad (1)
 * \f$
 *
 * that is a proportional-like dynamics with an adaptive gain, where <br>
 *
 * \f$
 *     k_p(e) = a \exp(-b \left\|e\right\|) + c        \qquad (2)
 * \f$
 *
 * To keep it simple \f$ a \f$,  \f$ b \f$, \f$ c \f$ and thus \f$ k_p \f$ will
 * be scalars.
 *
 * This task dynamics computes a desired first-order time derivative of the error
 * function \f$ e \f$. We say its order is one.
 *
 *
 * Implementation Outline
 * ======================
 *
 * From the user point of view, a task dynamics can be just specified by its type
 * and parameters (e.g. adaptive proportional with gain \p a, \p b and \p c in
 * this example). This can be done independently of the task \f$ f \circ rhs \f$
 * with which it will be used. From the computational point of view, the task
 * dynamics needs to be instantiated as a node of the computation graph taking
 * the value and possibly the derivatives of \f$f \f$ as an input. This step can
 * only be done if the task is known.
 * 
 * At the code level, this duality of viewpoints translates into the implementation
 * of two classes:
 *  1. A user-dedicated class, deriving from tvm::task_dynamics::abstract::TaskDynamics
 *     that acts as a lightweight description of the task and a factory for
 *  2. A computation-related class, deriving from
 *     tvm::task_dynamics::abstract::TaskDynamicsImpl
 *     that implements a node of the computation graph.
 *
 * We will call AdaptiveProportional the user-dedicated class. It simply needs to 
 * store \p a, \p b and \p c, and is required to override the two following methods
 *
 *     Order order_() const
 *     std::unique_ptr<TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const
 *
 * The first one has to return the task dynamics order (one, in our case), the
 * second creates an instance of the computation-related class.
 *
 * The second class could be implemented separately of the first but the
 * convention taken in TVM is to make it a subclass of the first, with the name
 * \p Impl. A few other classes rely on this convention such as
 * tvm::task_dynamics::Clamped, that would not be compatible with our example if
 * we didn't do so. This second class overrides the method
 *
 *     void updateValue()
 *
 * which implements the computation of the desired error derivative, i.e. the
 * formula (1) and (2) for our example. For that it will need to store store
 * \p a, \p b and \p c as well.
 *
 *
 * Implementation details
 * ======================
 * 
 * As a direct transcription of the outline above, our \p AdaptiveProportional
 * has the following declaration (assuming the proper namespaces):
 * <pre>\code
 *   class AdaptiveProportional : public abstract::TaskDynamics
 *   {
 *     public:
 *       class Impl : public task_dynamics::abstract::TaskDynamicsImpl { ... };
 *   
 *       AdaptiveProportional(double a, double b, double c);
 *   
 *     protected:
 *       task_dynamics::Order order_() const override;
 *       std::unique_ptr<task_dynamics::abstract::TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const override;
 *   
 *     private:
 *       double a_, b_, c_;
 *   };
 * \endcode</pre>
 * (The impl class is described below)
 *
 * The implementation is straightforward:
 * * the contructor trivially assigns \p a, \p b and \p c to the relevant fields
 * <pre>\code
 *   AdaptiveProportional::AdaptiveProportional(double a, double b, double c)
 *    : a_(a), b_(b), c_(c)
 *   {
 *   }
 * \endcode</pre>
 *
 * * \p order_() simply returns tvm::task_dynamics::Order::One.
 *
 * * \p impl_ is given the description of the task \f$ f \circ rhs \f$ through
 *   parameters \p f, \p t and \p rhs. From these and fields \p a_, \p b_, \p c_,
 *   it constructs a std::unique_ptr on \p AdaptiveProportional::Impl:
 * <pre>\code
 *   std::unique_ptr<task_dynamics::abstract::TaskDynamicsImpl> AdaptiveProportional::impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const
 *   {
 *     return std::unique_ptr<task_dynamics::abstract::TaskDynamicsImpl>(new Impl(f, t, rhs, a_, b_, c_));
 *   }
 * \endcode</pre>
 *
 * 
 * The \p Impl class is declared as
 * <pre>\code
 *   class Impl : public task_dynamics::abstract::TaskDynamicsImpl
 *   {
 *   public:
 *     Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs, double a, double b, double c);
 *     void updateValue() override;
 *   
 *   private:
 *     double a_, b_, c_;
 *   };
 * \endcode</pre>
 *
 * Once again, the constructor is straightforward. The main attention point is
 * that it needs to call the constructor of TaskDynamicsImpl to pass it not only
 * the parameters \p f, \p t and \p rhs, but also the order of the task dynamics.
 * <pre>\code
 *   AdaptiveProportional::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, double a, double b, double c)
 *     : TaskDynamicsImpl(task_dynamics::Order::One, f,t,rhs), a_(a), b_(b), c_(c)
 *   {
 *   }
 * \endcode</pre>
 *
 * The \p updateValue() method is the heart of the implementation. It will be
 * called by the computation graph to update the value of the TaskDynamicsImpl
 * instance (accessible through the \p value() method).
 * Within methods of \p Impl, we have access to the task parameters through the
 * \p function(), \p type() and \p rhs() methods. Furthermore, we have access to
 * the \p value_ member, which is where to put the output of our computations. <br>
 * The implementation is as follows
 * <pre>\code
 *   void AdaptiveProportional::Impl::updateValue()
 *   {
 *     value_ = function().value() - rhs();
 *     double kp = a_ * exp(-b_ * value_.norm()) + c_;
 *     value_ *= -kp;
 *   }
 * \endcode</pre>
 * The first line stores (temporarily) the value of the error function \f$ e \f$
 * in \p value_.
 * The second line computes the adaptive gain according to (2).
 * The last line computes the desired error velocity as in (1) and store it in
 * \p value_, as required.
 *
 * To go further
 * =============
 *
 * On this example
 * ---------------
 * The above implementation is a minimal working example, and could be improved 
 * or extended in several ways:
 *  * Checking the validity of the input parameters \p a_, \p b_, \p c_
 *    in the constructor of \p Impl. These parameters should be non-negative.
 *    It would be even be better to check the validity \a also in the constructor
 *    of \p AdaptiveProportional, to report errors as early as possible, and at a
 *    place that will be more user-friendly.
 *  * Adding getter and setters for the parameters (with the required checks).
 *  * Having \p a and \p c be possibly diagonal matrices (represented by a vector)
 *    or full matrices. This would require additionnal checks on the vector/matrix
 *    sizes with respect to the size of \f$ f \f$.
 *
 * General notes
 * -------------
 * Here are some points to keep in minds when implementing new task dynamics:
 *  * If you want your task dynamics to be included in the TVM you'll need to
 *    add the macro \p TVM_DLLAPI in front of both classes. This requires to
 *    include \p tvm/defs.h
 *  * By default the computation of the task dynamics value only relies on the
 *    value of \f$ f \f$ (and its velocity for second-order task dynamics). The
 *    constructor of \p TaskDynamicsImpl ensures that the computation graph is
 *    properly designed for this case. If your task dynamics depends on other
 *    computations, e.g. if your computation relies also on a function \f$ g \f$,
 *    you need to declare the computation dependencies in the constructor of
 *    \p Impl. The constructor of tvm::task_dynamics::abstract::TaskDynamicsImpl
 *    offers a good example of how to do so.
 */

#include <tvm/task_dynamics/abstract/TaskDynamics.h>

namespace tvm::example
{
  /** A class implementing the task dynamics 
    * \f$
    *     \dot{e}^* = - (a \exp(-b \left\|e\right\|) + c) e
    * \f$
    */
  class AdaptiveProportional : public task_dynamics::abstract::TaskDynamics
  {
    public:
      class Impl : public task_dynamics::abstract::TaskDynamicsImpl
      {
      public:
        Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs, double a, double b, double c);
        void updateValue() override;

      private:
        double a_;
        double b_;
        double c_;
      };

      AdaptiveProportional(double a, double b, double c);

    protected:
      std::unique_ptr<task_dynamics::abstract::TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const override;
      task_dynamics::Order order_() const override;

    private:
      double a_;
      double b_;
      double c_;
  };
}

// So far tvm::function::abstract::Function was only forward-declared 
#include<tvm/function/abstract/Function.h>

namespace tvm::example
{
  AdaptiveProportional::AdaptiveProportional(double a, double b, double c)
    : a_(a), b_(b), c_(c)
  {
  }

  std::unique_ptr<task_dynamics::abstract::TaskDynamicsImpl> AdaptiveProportional::impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const
  {
    return std::unique_ptr<task_dynamics::abstract::TaskDynamicsImpl>(new Impl(f, t, rhs, a_, b_, c_));
  }

  task_dynamics::Order AdaptiveProportional::order_() const
  {
    return task_dynamics::Order::One;
  }

  AdaptiveProportional::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, double a, double b, double c)
    : TaskDynamicsImpl(task_dynamics::Order::One, f,t,rhs), a_(a), b_(b), c_(c)
  {
  }

  void AdaptiveProportional::Impl::updateValue()
  {
    value_ = function().value() - rhs();              // e = f - rhs
    double kp = a_ * exp(-b_ * value_.norm()) + c_;   // k_p = a exp(-b ||e||) + c
    value_ *= -kp;                                    // \dot{e} = -k_p e
  }
}


//Let's run some quick tests to ensure this example is not outdated and compiles.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "../tests/doctest/doctest.h"

#include <tvm/function/IdentityFunction.h>

using namespace tvm;
using namespace Eigen;

TEST_CASE("AdaptiveProportional test")
{
  //Creation of a variable of size 3 and initialization
  VariablePtr x = Space(3).createVariable("x");
  x << .1, -.2, .3;

  //Creation of an identity function f and a rhs;
  auto f = std::make_shared<function::IdentityFunction>(x);
  VectorXd rhs = Vector3d(.1,.1,-.1);

  //Creating the TaskDynamics object
  example::AdaptiveProportional ap(1, 2, 0.1);

  //Getting the implementation (not usually done by the user)
  //The type is not important here
  auto impl = ap.impl(f, constraint::Type::EQUAL, rhs);

  //Usually, the computation graph is handled automatically, but here we need to
  //trigger the updates manually.
  f->updateValue();
  impl->updateValue();

  //We have e = f-rhs = (0,-.3,.4) and ||e|| = .5
  //a exp(-b ||e||) + c = exp(-1) + 0.1
  double kp = exp(-1) + 0.1;
  FAST_CHECK_UNARY(impl->value().isApprox(-kp * Vector3d(0, -.3, .4)));
}
