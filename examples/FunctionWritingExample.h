// Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM

/**
 * \file
 * \brief How to create a new TVM function.
 *
 * Preliminaries
 * =============
 *
 * In this example, we will be creating a <em>dot product</em> function between
 * two variables.
 *
 * Let's consider two variables \f$ x \f$ and \f$ y \f$ with the same size, and
 * let's define the function \f$ f(x,y) = x^T y \f$. The partial derivatives of
 * this function with respect to \f$ x \f$ and \f$ y \f$ are
 * 
 * \f$
 *    J_x = \frac{\partial f}{\partial x} = y^T \ \ \mbox{and}\ \ J_y = \frac{\partial f}{\partial y} = x^T \quad (1)
 * \f$
 *
 * If now \f$ x \f$ and \f$ y \f$ are themselves function of the time \f$ t \f$,
 * then we have
 *
 * \f$
 *    \dot{f} = \dot{x}^T y + x^T \dot{y} \quad (2)
 * \f$
 *
 * Given a function \f$ g(q) \f$, with Jacobian matrix \f$ J \f$, we call
 * <em>normal acceleration</em> the term \f$ \gamma = \dot{J}\dot{q} \f$. It 
 * appears when computing the acceleration of \f$ g(q) \f$:
 * \f$ \ddot{g}(q) = J \ddot{q} + \dot{J}\dot{q} \f$ (\f$ J \ddot{q} \f$ is in
 * the tangent space to \f$ g(q(t)) \f$. <br>
 * For the dot product, we have
 *
 * \f$
 *    \gamma = 2 \dot{x}^T \dot{y} \quad (3)
 * \f$
 *
 * We can also directly take the derivative of the Jacobian matrices:
 *
 * \f$
 *    \dot{J}_x = \dot{y}^T \ \ \mbox{and}\ \ \dot{J}_y = \dot{x}^T \quad (4)
 * \f$
 *
 * 
 * Implementation outline
 * ======================
 *
 * A TVM function is an object able to return the value of a mathematical
 * function \f$ f \f$ and some of its derivatives, given the values of the
 * variables \f$ x_i \f$ it depends on. It is implemented by creating a class
 * inheriting from tvm::function::abstract::Function.
 *
 * Behind the hood, TVM is making use of a <em>computation graph<\em>, that is a
 * set of computation units whose inputs and outputs are connected in some way.
 * A tvm::function::abstract::Function is meant to be a set of computation units
 * and pre-defines 5 outputs, accessible through 5 methods, as summarized in the
 * following table:
 * 
 * |          output id            |                mathematical meaning                 |        returned by        |        cache member (type)        |
 * | :---------------------------: | :-------------------------------------------------: | :-----------------------: | :-------------------------------: |
 * | \c Output::Value              | \f$ f(x_1, \ldots x_n) \f$                          | \c value()                | \c value_ (VectorXd)              |
 * | \c Output::Jacobian           | \f$ \frac{\partial f}{\partial x_i} \f$             | \c jacobian(xi)           | \c jacobian_ (map of MatrixXd)    |
 * | \c Output::Velocity           | \f$ \dot{f}(x_1, \ldots x_n) \f$                    | \c velocity()             | \c velocity_ (VectorXd)           |
 * | \c Output::NormalAcceleration | \f$ \sum \dot{J}_{x_i} \dot{x}_i \f$                | \c normalAcceleration(xi) | \c normalAcceleration_ (VectorXd) |
 * | \c Output::JDot               | \f$ \frac{d}{dt}\frac{\partial f}{\partial x_i} \f$ | \c JDot()                 | \c JDot_ (map of MatrixXd)        |
 * 
 * To avoid duplicated computations, the output values are cached. The update
 * of this cached values is done by <em>update methods</em> that will be
 * called when needed and in the correct order. The name of the pre-defined cache
 * member is given in the above table. <br>
 * The core of implementing a new TVM function consists thus in two steps:
 *  * declaring the update methods and their relations to the inputs, outputs
 *    (and in some cases other update methods)
 *  * implementing these methods
 * An update method may update several outputs, or be used by other update
 * methods.
 *
 * We will implement a class \c DotProduct with the following update methods:
 *  * \c updateValue, updating \c value_
 *  * \c updateJacobian, updating \c jacobian_
 *  * \c updateVelocityAndNormalAcc, updating \c velocity_ and \c normalAcceleration_
 *  * \c updateJDot, updating \c JDot_
 * Grouping the updates of the velocity and the normal acceleration can often have
 * a sense, because these quantities are often use in the same context in robotic
 * control. Here we also do it for showing the possibility.
 *
 *
 * Implemetation details
 * =====================
 *
 * Class declaration
 * -----------------
 * The class is defined as follows
 * \dontinclude FunctionWritingExample.cpp
 * <pre>\skipline class
 * \until }; 
 * </pre>
 *
 * With the \c SET_UPDATES macro, we are defining identifiers for the update
 * methods. The first argument needs to be the class name, the other are the
 * identifiers and can be chosen freely. <br>
 * \c SET_UPDATES is performing several tasks necessary for the computation graph
 * machinery, including creating an enumeration class DotProduct::Update, whose
 * elements are \c Value, \c Jacobian, \c VelocityAndNormalAcc and \c JDot.
 *
 * The constructor of the class simply takes (shared) pointers on the variables
 * \p x and \p y. Then comes the 4 update methods we want to define. Their names
 * are free, but it is good practice to use the pattern \c updateID, where ID is
 * the identifier chosen in \c SET_UPDATES.
 *
 * Finally, we declare four references on the variables \p x, \p y and their
 * first derivatives. These are merely convenience members, as the variables can
 * be accessed through the \c variables_ member, inherited from
 * tvm::function::abstract::Function.
 *
 *
 * Constructor
 * -----------
 * Three characteristics of the function should be specified before it is used
 * and this should generally be done in the constructor:
 *  * the size of its value \f$ f(x_1, \ldots x_n) \f$, what TVM refers to as the
 *    \e size of the function (1 in our \c DotProduct example)
 *  * the variables it depends on
 *  * its internal computation graph
 * The size is passed as an argument to the base tvm::function::abstract::Function
 * constructor. <bre>
 * The variables are specified by the \c addVariable method. <br>
 * For functions that depend only on their variables, and not on other functions
 * or computation nodes, the internal computation graph is described by
 *  * \c registerUpdates, which takes pairs of (update id, pointer to the
 *    corresponding update method), to register the existing update methods
 *  * \c addOutputDependency which takes an output id (or a list of them) and a
 *    update id, to specify that the given output(s) is (/are) updated by the
 *    corresponding update method.
 * In case an update method relies on the computation of another one, the
 * dependency is declared with \c addInternalDependency.
 *
 * The code of our constructor is
 * <pre>\skipline DotProduct::DotProduct
 * \until }
 * </pre>
 * \c Function(1) is the specification of the function size. 
 * The references are initialized from the shared pointer to the adequate
 * variables. Those variables lifetime is ensured
 */