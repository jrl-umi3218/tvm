/* Copyright 2022 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/solver/internal/Option.h>

#include <lexls/lexlsi.h>

namespace tvm::solver
{
/** A set of options for the LexLSI solver.
 *
 * These are directly the lexlsi options as found in LexLS::ParametersLexLSI.
 * Documentation and default values are copied from lexls/typedefs.h which should serve as a
 *  reference in case of doubt/discrepancies.
 */
class LexLSSolverOptions
{
  /** Maximum number of factorizations (if reached, the solver terminates) - default: 200. */
  TVM_ADD_DEFAULT_OPTION(max_number_of_factorizations, LexLS::Index);
  /** Tolerance: linear dependence (used when solving an LexLSE problem) - default: 1e-12. */
  TVM_ADD_DEFAULT_OPTION(tol_linear_dependence, LexLS::RealScalar);
  /** Tolerance: absolute value of Lagrange multiplier to be considered with "wrong" sign - default: 1e-8. */
  TVM_ADD_DEFAULT_OPTION(tol_wrong_sign_lambda, LexLS::RealScalar);
  /** Tolerance: absolute value of Lagrange multiplier to be considered with "correct" sign - default: 1e-12. */
  TVM_ADD_DEFAULT_OPTION(tol_correct_sign_lambda, LexLS::RealScalar);
  /** Tolerance: used to determine whether a constraint has been violated
   *
   * \note This tolerance is used when checking for blocking constraint and when initializing v0.
   *
   * default: 1e-13.
   */
  TVM_ADD_DEFAULT_OPTION(tol_feasibility, LexLS::RealScalar);
  /** Type of regularization (Tikhonov, Basic Tikhonov, ...) - default: LexLS::REGULARIZATION_NONE. */
  TVM_ADD_DEFAULT_OPTION(regularization_type, LexLS::RegularizationType);
  /** Max number of iterations for cg_tikhonov(...)
   *
   * \note used only with regularization_type = REGULARIZATION_TIKHONOV_CG
   *
   * default: undefined
   */
  TVM_ADD_DEFAULT_OPTION(max_number_of_CG_iterations, LexLS::Index);
  /** When variable_regularization_factor = 0 the user specified regularization factors
   * are used directly. When variable_regularization_factor != 0 an estimation of the
   * conditioning (conditioning_estimate) of each level is made (during the LOD). Then if
   * conditioning_estimate > variable_regularization_factor no regularization is applied,
   * while if conditioning_estimate < variable_regularization_factor the regularization
   * factors (provided by the user) are modified (there are various approaches to do this, see
   * lexlse::factorize(...)).
   *
   * \attention This functionality is not mature yet (use with caution).
   *
   * default: undefined
   */
  TVM_ADD_DEFAULT_OPTION(variable_regularization_factor, LexLS::RealScalar);
  /** If cycling_handling_enabled == true, cycling handling is performed - default: true. */
  TVM_ADD_NON_DEFAULT_OPTION(cycling_handling_enabled, true);
  /** Maximum number of attempts for cycling handling - default: 50. */
  TVM_ADD_DEFAULT_OPTION(cycling_max_counter, LexLS::Index);
  /** Amount of relaxation performed during each attempt to handle cycling - default: 1e-8. */
  TVM_ADD_DEFAULT_OPTION(cycling_relax_step, LexLS::RealScalar);
  /** Used to output information for intermediate iterations of the solver */
  TVM_ADD_DEFAULT_OPTION(output_file_name, std::string);
  /** Allows modification of the user guess for x (see doc/hot_start.pdf) - default: false. */
  TVM_ADD_DEFAULT_OPTION(modify_x_guess_enabled, bool);
  /** Allows modification of the user guess for active constraints (see doc/hot_start.pdf) - default: false. */
  TVM_ADD_DEFAULT_OPTION(modify_type_active_enabled, bool);
  /** Allows modification of the user guess for inactive constraints (see doc/hot_start.pdf) - default: false. */
  TVM_ADD_DEFAULT_OPTION(modify_type_inactive_enabled, bool);
  /** Generate the smallest possible v0 (see doc/hot_start.pdf) - default: true. */
  TVM_ADD_DEFAULT_OPTION(set_min_init_ctr_violation, bool);
  /** If true, use phase1_v0() instead of phase1() - default: false. */
  TVM_ADD_DEFAULT_OPTION(use_phase1_v0, bool);
  /** If true, gather information about activations and deactivations - default: false. */
  TVM_ADD_DEFAULT_OPTION(log_working_set_enabled, bool);
  /** If true, deactivate first constraints with lambda with wrong sign. Otherwise, deactivate
   * constraints with largest lambda (with wrong sign).
   *
   * default: false.
   */
  TVM_ADD_DEFAULT_OPTION(deactivate_first_wrong_sign, bool);
};

} // namespace tvm::solver
