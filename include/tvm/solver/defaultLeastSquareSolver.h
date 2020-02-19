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

#pragma once

/** Since all solver might not be available depending on the option compilations,
  * this header offers conveniencies to include and use a default solver.
  * The first solver found in this (ordered) list is taken as default:
  * LSSOL, QLD, Quadprog
  */
#ifdef USE_LSSOL
# include <tvm/solver/LSSOLLeastSquareSolver.h>
#else
# ifdef USE_QLD
#   include <tvm/solver/QLDLeastSquareSolver.h>
# else
#   ifdef USE_QUADPROG
#     include <tvm/solver/QuadprogLeastSquareSolver.h>
#   else
#     error "You should at least have one solver. If not, there is a problem with the CMakeLists.txt"
#   endif
# endif
#endif

namespace tvm::solver
{
  class DefaultLSSolverFactory;

  /** A minimal set of options for the default solver.
    *
    * \internal There should only be non default options here.
    */
  class TVM_DLLAPI DefaultLSSolverOptions
  {
    TVM_ADD_NON_DEFAULT_OPTION(big_number, constant::big_number)
    TVM_ADD_NON_DEFAULT_OPTION(verbose, false)

  public:
    using Factory = DefaultLSSolverFactory;
  };

  /** A factory class to create default solver instances with a given
    * set of options.
    */
  class TVM_DLLAPI DefaultLSSolverFactory : public abstract::LSSolverFactory
  {
  public:
    std::unique_ptr<abstract::LSSolverFactory> clone() const override;

    /** Creation of a configuration from a set of options*/
    DefaultLSSolverFactory(const DefaultLSSolverOptions& options = {});

    std::unique_ptr<abstract::LeastSquareSolver> createSolver() const override;

  private:
    DefaultLSSolverOptions options_;
  };
}