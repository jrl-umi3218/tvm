/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/Task.h>
#include <tvm/constraint/abstract/LinearConstraint.h>
#include <tvm/utils/ProtoTask.h>

namespace tvm
{

namespace constraint
{

namespace internal
{

/** Given a task \f$(e, op, rhs, e^*)\f$, this class derives the constraint
 * \f$d^k e/dt^k\ op\  e^*(e,de/dt,...de^{k-1}/dt^{k-1}, rhs [,g])\f$, where e is an
 * error function, op is ==, >= or <= and \f$e^*\f$ is a desired error dynamics.
 * k is specified by \f$e^*\f$ and (optional) g is any other quantities.
 *
 * EQUAL (E) \ GREATER_THAN (L) \ LOWER_THAN (U) cases. Dotted dependencies
 * correspond by default to the second order dynamics case (k=2), unless
 * specified otherwise by the task dynamics used.
 * \dot
 * digraph "update graph" {
 *   rankdir="LR";
 *   subgraph cluster1 {
 *     label="f"
 *     {
 *       rank=same; node [shape=diamond];
 *       fValue [label="Value"];
 *       fJacobian [label="Jacobian"];
 *       fVelocity [label="Velocity",style=dotted];
 *       fNormalAcceleration [label="NormalAcceleration",style=dotted];
 *     }
 *   }
 *   subgraph cluster2 {
 *     label="td"
 *     {
 *       tdValue [label="Value", shape=Mdiamond];
 *       tdUpdate [label="UpdateValue"];
 *     }
 *   }
 *   {
 *     rank=same;
 *     uValue [label=Value];
 *     updateRHS;
 *   }
 *   {
 *     rank = same; node [shape=hexagon];
 *     Value; Jacobian;
 *     E [label="E \\ L \\ U"];
 *   }
 *   {
 *     rank = same; node [style=invis, label=""];
 *     outValue; outJacobian; outE;
 *   }
 *   x_i [shape=box]
 *   fValue -> tdUpdate
 *   fVelocity -> tdUpdate [style=dotted]
 *   tdUpdate -> tdValue
 *   tdValue -> updateRHS
 *   updateRHS -> E
 *   fJacobian -> Jacobian
 *   fJacobian -> uValue
 *   fNormalAcceleration -> updateRHS [style=dotted]
 *   Value -> outValue [label="value()"];
 *   Jacobian -> outJacobian [label="jacobian(x_i)"];
 *   E -> outE [label="e() \\ l() \\ u()"];
 *   x_i -> uValue [label="value()"];
 *   uValue -> Value;
 * }
 * \enddot
 *
 * DOUBLE_SIDED case. Dotted dependencies correspond by default to the second
 * order dynamics case (k=2), unless specified otherwise by the task dynamics
 * used.
 * \dot
 * digraph "update graph" {
 *   rankdir="LR";
 *   subgraph cluster1 {
 *     label="f"
 *     {
 *       rank=same; node [shape=diamond];
 *       fValue [label="Value"];
 *       fJacobian [label="Jacobian"];
 *       fVelocity [label="Velocity",style=dotted];
 *       fNormalAcceleration [label="NormalAcceleration",style=dotted];
 *     }
 *   }
 *   subgraph cluster2 {
 *     label="td"
 *     {
 *       td1Value [label="Value", shape=Mdiamond];
 *       td1Update [label="UpdateValue"];
 *     }
 *   }
 *   subgraph cluster3 {
 *     label="td2"
 *     {
 *       td2Value [label="Value", shape=Mdiamond];
 *       td2Update [label="UpdateValue"];
 *     }
 *   }
 *   {
 *     rank=same;
 *     uValue [label=Value];
 *     updateRHS;
 *     updateRHS2;
 *   }
 *   {
 *     rank = same; node [shape=hexagon];
 *     Value; Jacobian; L; U
 *   }
 *   {
 *     rank = same; node [style=invis, label=""];
 *     outValue; outJacobian; outL; outU
 *   }
 *   x_i [shape=box]
 *   fValue -> td1Update
 *   fVelocity -> td1Update [style=dotted]
 *   td1Update -> td1Value
 *   td1Value -> updateRHS
 *   fValue -> td2Update
 *   fVelocity -> td2Update [style=dotted]
 *   td2Update -> td2Value
 *   td2Value -> updateRHS2
 *   updateRHS -> L
 *   updateRHS2 -> U
 *   fJacobian -> Jacobian
 *   fJacobian -> uValue
 *   fNormalAcceleration -> updateRHS [style=dotted]
 *   fNormalAcceleration -> updateRHS2 [style=dotted]
 *   Value -> outValue [label="value()"];
 *   Jacobian -> outJacobian [label="jacobian(x_i)"];
 *   L -> outL [label="l()"];
 *   U -> outU [label="u()"];
 *   x_i -> uValue [label="value()"];
 *   uValue -> Value;
 * }
 * \enddot
 *
 * \internal FIXME Consider the case where the TaskDynamics has its own variables?
 */
class TVM_DLLAPI LinearizedTaskConstraint : public abstract::LinearConstraint
{
public:
  SET_UPDATES(LinearizedTaskConstraint, UpdateRHS, UpdateRHS2)

  /** Constructor from a task*/
  LinearizedTaskConstraint(const Task & task);

  /** Constructor from a ProtoTask and a TaskDynamics*/
  template<constraint::Type T>
  LinearizedTaskConstraint(const utils::ProtoTask<T> & pt, const task_dynamics::abstract::TaskDynamics & td);

  /** Update the \p l vector, for kinematic tasks.*/
  void updateLKin();
  /** Update the \p l vector, for dynamic tasks.*/
  void updateLDyn();
  /** Update the \p u vector, for kinematic, single-sided tasks.*/
  void updateUKin();
  /** Update the \p u vector, for dynamic, single-sided tasks.*/
  void updateUDyn();
  /** Update the \p e vector, for kinematic tasks.*/
  void updateEKin();
  /** Update the \p e vector, for dynamic tasks.*/
  void updateEDyn();
  /** Update the \p u vector, for kinematic, double-sided tasks.*/
  void updateU2Kin();
  /** Update the \p u vector, for dynamic, double-sided tasks.*/
  void updateU2Dyn();

  /** Return the jacobian matrix corresponding to \p x */
  const tvm::internal::MatrixWithProperties & jacobian(const Variable & x) const override;

private:
  FunctionPtr f_;
  TaskDynamicsPtr td_;
  TaskDynamicsPtr td2_; // for double sided constraints only;
};

template<constraint::Type T>
LinearizedTaskConstraint::LinearizedTaskConstraint(const utils::ProtoTask<T> & pt,
                                                   const task_dynamics::abstract::TaskDynamics & td)
: LinearizedTaskConstraint(Task(pt, td))
{}

} // namespace internal

} // namespace constraint

} // namespace tvm
