/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/solver/defaultLeastSquareSolver.h>

namespace tvm::solver
{
std::unique_ptr<abstract::LSSolverFactory> DefaultLSSolverFactory::clone() const
{
  return std::make_unique<DefaultLSSolverFactory>(*this);
}

DefaultLSSolverFactory::DefaultLSSolverFactory(const DefaultLSSolverOptions & options)
: LSSolverFactory("default"), options_(options)
{}

std::unique_ptr<abstract::LeastSquareSolver> DefaultLSSolverFactory::createSolver() const
{
#ifdef TVM_USE_LSSOL
  using SolverType = LSSOLLeastSquareSolver;
  using SolverOption = LSSOLLSSolverOptions;
#else
#  ifdef TVM_USE_QLD
  using SolverType = QLDLeastSquareSolver;
  using SolverOption = QLDLSSolverOptions;
#  else
#    ifdef TVM_USE_QUADPROG
  using SolverType = QuadprogLeastSquareSolver;
  using SolverOption = QuadprogLSSolverOptions;
#    endif
#  endif
#endif
  return std::make_unique<SolverType>(SolverOption().big_number(*options_.big_number()).verbose(*options_.verbose()));
}
} // namespace tvm::solver
