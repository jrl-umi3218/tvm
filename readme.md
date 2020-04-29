The TVM Library
=============
TVM is a library meant for writing and solving linear control problems for robots.
At its heart lies an optimization framework with several helpful features (such as variable management and handling of convention differences in the way of writing constraint), on top of which robotic functionalities are added.
The library strives to separate the way a problem is written from the way it is solved. This allows to write problems in a way mirroring closely their natural mathematical formulations. The work of correctly assembling the corresponding matrices and vectors to be passed to a numerical solver is done automatically with little to no overhead over a painful and error-prone manual implementation.

TVM has typically three types of users:
 - the end-user, formulating the problem he/she want to solve. He/she does so by manipulating notions such as variables, functions, tasks and task dynamics, using the existing functions and task dynamics to do so,
 - the user adding robotics functionality, typically new functions or tasks dynamics. He/she need to understand some of the internals of the library such as the update mechanism,
 - the user adding solving capability, such as a new resolution scheme, with a deeper knowledge of the internals.
 
 The framework distinguishes two world: the task world and the optimization world. The end-user works in the tasks world. There, he/she creates variables, functions of them or their derivatives. A function is then associated with an equality or inequality goal and a prescribed dynamics to achieve this goal, forming a task. There is no differentiation between constraints and objectives at this point. The user can then add these tasks to a problem, completed by requirements on how to solve each task (level of priority, weight, ...).
 The problem can then be linearized and solved, all of which is not the concern of the end-user.
 The linearization makes the bridge to the second world. Here the tasks are turned into (linear) constraints, and a resolution scheme builds the inputs to be passed to a numerical solver according to the tasks' accompanying requirements.


#### Main features
 - strong decoupling between the formulation of a problem and the way to solve it
 - easy way to express a task from a function
 - extensibility and numerous points of customization
 - many classical robotic functions (Equation of dynamics, position and orientation of a body, collisions, ...)
 - lightweight variable management
 - variable substitution
 - update mechanism to ensure only quantities need are computed and computation happens only once
 - state-of-the-art resolution schemes
 - low-level efficient tools to help writing new resolution schemes
 

Installation
-------------
Compilation has been tested on Linux (gcc/clang) and Windows (Visual Studio).

### Dependencies

To compile you will need the following tools:

 * [Git](https://git-scm.com/)
 * [CMake](https://cmake.org/) >= 3.1.3
 * [doxygen](http://www.doxygen.org)
 * A compiler with C++17 support
 
and the following third-party dependencies:
 * [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) >= 3.2.8
 * [Boost](http://www.boost.org/) >= 1.49 (only for some tests)

TVM requires that you installed first some other JRL/LIRMM software:
 * [RBDyn](https://github.com/jrl-umi3218/RBDyn)
 * [sch-core](https://github.com/jrl-umi3218/sch-core)
 * At least one of the following solvers:
   + [eigen-qld](https://github.com/jrl-umi3218/eigen-qld)
   + [eigen-quadprog](https://github.com/jrl-umi3218/eigen-quadprog)
   + [eigen-lssol](git@gite.lirmm.fr:multi-contact/eigen-lssol.git) (private repository)
 * [Tasks](https://github.com/jrl-umi3218/Tasks) (optionally, for some comparison tests)

This repository also uses [jrl-cmakemodules](https://github.com/jrl-umi3218/jrl-cmakemodules), [jrl-travis](https://github.com/jrl-umi3218/jrl-travis) and [google benchmark](https://github.com/google/benchmark) as submodules.

### Building from source on Linux

Follow the standard CMake build procedure:

```sh
git clone --recursive https://github.com/jrl-umi3218/tvm
cd tvm
mkdir build && cd build
cmake [options] ..
make && make install
```

where the main options are:
 * `-DCMAKE_BUILD_TYPE=Release` Build in Release mode
 * `-DCMAKE_INSTALL_PREFIX=some/path/to/install` default is `/usr/local`
 
Documentation
--------------------
The main classes are documented with doxygen-compatible comments (WIP)

### Topics (todo):
  - how to write a problem
  - the update mechanism
  - how to write a function
  - the assignment mechanism
  - how to write a resolution scheme
