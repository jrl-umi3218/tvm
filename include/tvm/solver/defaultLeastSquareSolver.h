/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

/** Since all solver might not be available depending on the option compilations,
 * this header offers conveniences to include and use a default solver.
 * The first solver found in this (ordered) list is taken as default:
 * LSSOL, QLD, Quadprog
 */
#include <tvm/solver/abstract/LeastSquareSolver.h>

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
  DefaultLSSolverFactory(const DefaultLSSolverOptions & options = {});

  std::unique_ptr<abstract::LeastSquareSolver> createSolver() const override;

private:
  DefaultLSSolverOptions options_;
};
} // namespace tvm::solver
