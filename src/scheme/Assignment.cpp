/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/scheme/internal/Assignment.h>

#include <tvm/defs.h>

#include <tvm/VariableVector.h>
#include <tvm/constraint/abstract/Constraint.h>
#include <tvm/scheme/internal/helpers.h>

namespace tvm::scheme::internal
{
using constraint::RHS;
using constraint::Type;
using constraint::abstract::LinearConstraint;

double Assignment::big_ = constant::big_number;

Assignment::Assignment(LinearConstraintPtr source,
                       SolvingRequirementsPtr req,
                       const AssignmentTarget & target,
                       const VariableVector & variables,
                       const hint::internal::Substitutions * const substitutions,
                       double scalarizationWeight)
: source_(source), target_(target), scalarizationWeight_(scalarizationWeight), requirements_(req),
  substitutedVariables_(substitutions ? substitutions->variables() : VariableVector()),
  variableSubstitutions_(substitutions ? substitutions->variableSubstitutions()
                                       : std::vector<std::shared_ptr<function::BasicLinearFunction>>()),
  data_(new ReferenceableData())
{
  checkTarget();
  // TODO check also that the variables of source are in the variable vector
  processRequirements();
  build(variables);
}

Assignment::Assignment(LinearConstraintPtr source,
                       const AssignmentTarget & target,
                       const VariablePtr & variable,
                       bool first)
: source_(source), target_(target), requirements_(nullptr), useDefaultScalarWeight_(true),
  useDefaultAnisotropicWeight_(true), data_(new ReferenceableData())
{
  checkBounds();
  assert(source->variables()[0] == variable);
  build(variable, first);
}

Assignment Assignment::reprocess(const Assignment & other,
                                 const VariableVector & variables,
                                 const hint::internal::Substitutions * const subs)
{
  return Assignment(other.source_, other.requirements_, other.target_, variables, subs, other.scalarizationWeight_);
}

Assignment Assignment::reprocess(const Assignment & other, const VariablePtr & x, bool first)
{
  return Assignment(other.source_, other.target_, x, first);
}

AssignmentTarget & Assignment::target(IWontForgetToCallUpdates) { return target_; }

bool Assignment::changeScalarWeightIsAllowed() { return !useDefaultScalarWeight_; }

bool Assignment::changeVectorWeightIsAllowed() { return !useDefaultAnisotropicWeight_; }

void Assignment::onUpdatedSource()
{
  // TODO
}

void Assignment::onUpdatedTarget()
{
  for(auto & a : matrixAssignments_)
    a.updateTarget(target_);
  for(auto & a : matrixSubstitutionAssignments_)
    a.updateTarget(target_);
  for(auto & a : vectorAssignments_)
    a.assignment.to((target_.*a.getTargetVector)());
  for(auto & a : vectorSubstitutionAssignments_)
    a.assignment.to((target_.*a.getTargetVector)());
}

void Assignment::onUpdatedMapping(const VariableVector & newVar, bool updateMatrixTarget)
{
  for(auto & a : matrixAssignments_)
    a.updateMapping(newVar, target_, updateMatrixTarget);
  for(auto & a : matrixSubstitutionAssignments_)
    a.updateMapping(newVar, target_, updateMatrixTarget);
}

void Assignment::onUpdateWeights(bool scalar, bool vector)
{
  if(scalar)
  {
    if(useDefaultScalarWeight_ && requirements_->weight().value() != 1)
      throw std::runtime_error("[Assignment::onUpdateWeights] Can't update a default weight.");

    data_->scalarWeight_ = std::sqrt(requirements_->weight().value()) * scalarizationWeight_;
    data_->minusScalarWeight_ = -data_->scalarWeight_;
  }
  if(vector)
  {
    if(!requirements_->anisotropicWeight().isDefault()
       && requirements_->anisotropicWeight().value().size() != target_.size())
      throw std::runtime_error("[Assignment::onUpdateWeights] Anisotropic weight has not the correct size.");

    if(useDefaultAnisotropicWeight_ && !requirements_->anisotropicWeight().isDefault()
       && !requirements_->anisotropicWeight().value().isConstant(1))
      throw std::runtime_error("[Assignment::onUpdateWeights] Can't update a default weight.");

    data_->anisotropicWeight_ = requirements_->anisotropicWeight().value().cwiseSqrt();
    data_->anisotropicWeight_ *= data_->scalarWeight_;
    data_->minusAnisotropicWeight_ = -data_->anisotropicWeight_;
  }
}

void Assignment::run()
{
  for(auto & a : matrixAssignments_)
    a.assignment.run();

  for(auto & a : matrixSubstitutionAssignments_)
    a.assignment.run();

  for(auto & a : vectorAssignments_)
    a.assignment.run();

  for(auto & a : vectorSubstitutionAssignments_)
    a.assignment.run();
}

void Assignment::checkTarget()
{
  // check the type convention
  if(source_->type() == Type::EQUAL)
  {
    if(target_.constraintType() != Type::EQUAL && target_.constraintType() != Type::DOUBLE_SIDED)
      throw std::runtime_error("Incompatible conventions: source with type EQUAL can only "
                               "be assigned to target with type EQUAL or DOUBLE SIDED.");
  }
  else
  {
    if(target_.constraintType() == Type::EQUAL)
      throw std::runtime_error("Incompatible conventions: source with type other than EQUAL "
                               "cannot be assigned to a target with type EQUAL. ");
  }

  // check the rhs conventions
  bool hasNonZero = false;
  for(const auto & s : variableSubstitutions_)
  {
    const auto & p = s->b().properties();
    hasNonZero = hasNonZero || !p.isZero() || !p.isConstant();
  }
  if((source_->rhs() != RHS::ZERO || hasNonZero) && target_.constraintRhs() == RHS::ZERO)
    throw std::runtime_error("Incompatible conventions: source with rhs other than ZERO "
                             "(or when using substitutions with non-zero vector term) "
                             "cannot be assigned to target with rhs ZERO.");

  // check the sizes
  if(source_->type() == Type::DOUBLE_SIDED)
  {
    if(target_.constraintType() == Type::DOUBLE_SIDED)
    {
      if(target_.size() != source_->size())
        throw std::runtime_error("Size of the source and the target are not coherent with "
                                 "the conventions.");
    }
    else
    {
      if(target_.size() != 2 * source_->size())
        throw std::runtime_error("Size of the source and the target are not coherent with "
                                 "the conventions.");
    }
  }
  else
  {
    if(target_.size() != source_->size())
      throw std::runtime_error("Size of the source and the target are not coherent with "
                               "the conventions.");
  }
}

void Assignment::checkBounds()
{
  if(!canBeUsedAsBound(source_, substitutedVariables_.variables(), variableSubstitutions_, target_.constraintType()))
  {
    throw std::runtime_error("Incompatible bound conventions or properties");
  }
}

void Assignment::build(const VariableVector & variables)
{
  // In this function, we split up the global assignment into atomic assignments.
  // This is done according to the ConstraintType of both source and target
  if(source_->type() == constraint::Type::EQUAL)
  {
    /*
     *                 |  Ax = b
     * ----------------+--------------
     *      Cx = d     | C=A, d=b
     *  dl <= Cx <= du | C=A, dl=du=b
     */

    switch(target_.constraintType())
    {
      case constraint::Type::EQUAL:
        addAssignments(variables, &AssignmentTarget::A, &LinearConstraint::e, &AssignmentTarget::b, false);
        break;
      case constraint::Type::DOUBLE_SIDED:
        addAssignments(variables, &AssignmentTarget::A, &LinearConstraint::e, &AssignmentTarget::l,
                       &LinearConstraint::e, &AssignmentTarget::u);
        break;
      default:
        throw std::runtime_error("Impossible to assign source for the given target convention.");
    }
  }
  else
  {
    /*
     *                 |        Ax >= b        |        Ax <= b         |       bl <= Ax <= bu
     * ----------------+-----------------------+------------------------+---------------------------
     *     Cx >= d     | (1) C=A, d=b          | (4) C=-A, d=-b         | (7) C=[A,-A], d=[bl,-bu]
     *     Cx <= d     | (2) C=-A, d=-b        | (5) C=A, d=b           | (8) C=[A,-A], d=[bu,-bl]
     *  dl <= Cx <= du | (3) C=A, dl=b, du=inf | (6) C=A, dl=-inf, du=b | (9) C=A, dl = bl, du = bu
     */

    if(source_->type() == constraint::Type::DOUBLE_SIDED)
    {
      if(target_.constraintType() == constraint::Type::DOUBLE_SIDED)
      {
        // case 9
        addAssignments(variables, &AssignmentTarget::A, &LinearConstraint::l, &AssignmentTarget::l,
                       &LinearConstraint::u, &AssignmentTarget::u);
      }
      else
      {
        if(target_.constraintType() == constraint::Type::GREATER_THAN)
        {
          // case 7
          addAssignments(variables, &AssignmentTarget::AFirstHalf, &LinearConstraint::l, &AssignmentTarget::bFirstHalf,
                         false);
          addAssignments(variables, &AssignmentTarget::ASecondHalf, &LinearConstraint::u,
                         &AssignmentTarget::bSecondHalf, true);
        }
        else
        {
          // case 8
          addAssignments(variables, &AssignmentTarget::AFirstHalf, &LinearConstraint::u, &AssignmentTarget::bFirstHalf,
                         false);
          addAssignments(variables, &AssignmentTarget::ASecondHalf, &LinearConstraint::l,
                         &AssignmentTarget::bSecondHalf, true);
        }
      }
    }
    else
    {
      // case 4 and 5 are just opposite of case 1 and 2.
      bool flip = source_->type() == constraint::Type::LOWER_THAN;
      RHSFunction f;
      if(source_->type() == constraint::Type::LOWER_THAN)
        f = &LinearConstraint::u; // for case 1 and 2
      else
        f = &LinearConstraint::l; // for case 4 and 5

      switch(target_.constraintType())
      {
        case constraint::Type::EQUAL:
          throw std::runtime_error("Impossible to assign inequality source for equality target.");
        case constraint::Type::GREATER_THAN:
          // cases 1 and 4
          addAssignments(variables, &AssignmentTarget::A, f, &AssignmentTarget::b, flip);
          break;
        case constraint::Type::LOWER_THAN:
          // cases 2 and 5
          addAssignments(variables, &AssignmentTarget::A, f, &AssignmentTarget::b, !flip);
          break;
        case constraint::Type::DOUBLE_SIDED:
          // for case 3 and 6, the signs of A and C are the same
          if(source_->type() == constraint::Type::GREATER_THAN)
          {
            // case 3
            addAssignments(variables, &AssignmentTarget::A, f, &AssignmentTarget::l, false);
            addVectorAssignment(big_, &AssignmentTarget::u);
          }
          else
          {
            // case 6
            addAssignments(variables, &AssignmentTarget::A, f, &AssignmentTarget::u, false);
            addVectorAssignment(-big_, &AssignmentTarget::l);
          }
          break;
      }
    }
  }
}

void Assignment::build(const VariablePtr & variable, bool first)
{
  // We know that: source_ is a single-variable constraint with invertible
  // diagonal jacobian, and that if there is a substitution it is also by a
  // single variable with an invertible diagonal jacobian.
  // We also know that the target is compatible with the source + substitution,
  // and that it can't be of type EQUAL.

  assert(target_.constraintType() != Type::EQUAL);

  if(target_.constraintType() == Type::DOUBLE_SIDED)
  {
    switch(source_->type())
    {
      case Type::EQUAL:
        addBounds(variable, &LinearConstraint::e, &LinearConstraint::e, first);
        break;
      case Type::GREATER_THAN:
        addBounds(variable, &LinearConstraint::l, +big_, first);
        break;
      case Type::LOWER_THAN:
        addBounds(variable, -big_, &LinearConstraint::u, first);
        break;
      case Type::DOUBLE_SIDED:
        addBounds(variable, &LinearConstraint::l, &LinearConstraint::u, first);
        break;
      default:
        assert(false);
    }
  }
  else // target is GREATER_THAN or LOWER_THAN
  {
    if(source_->type() == Type::GREATER_THAN)
    {
      addBound(variable, &LinearConstraint::l, first);
    }
    else // source is LOWER_THAN
    {
      addBound(variable, &LinearConstraint::u, first);
    }
  }
  // VectorFunction l;
  // VectorFunction u;
  // bool flip;
  // if (source_->jacobian(*variable).properties().isIdentity())
  //{
  //  l = &AssignmentTarget::l;
  //  u = &AssignmentTarget::u;
  //  flip = false;
  //}
  // else if (source_->jacobian(*variable).properties().isMinusIdentity())
  //{
  //  l = &AssignmentTarget::u;
  //  u = &AssignmentTarget::l;
  //  flip = true;
  //}
  // else
  //{
  //  throw std::runtime_error("Pure diagonal case is not implemented yet.");
  //}

  // if (first)
  //{
  //  assignBounds(l, u, flip);
  //}
  // else
  //{
  //  if (flip)
  //    assignBounds<AssignType::MIN, AssignType::MAX>(l, u, flip);
  //  else
  //    assignBounds<AssignType::MAX, AssignType::MIN>(l, u, flip);
  //}
}

void Assignment::processRequirements()
{
  if(requirements_)
  {
    switch(requirements_->violationEvaluation().value())
    {
      case requirements::ViolationEvaluationType::L1:
        throw std::runtime_error("Unimplemented violation evaluation type.");
      case requirements::ViolationEvaluationType::L2:
        if(requirements_->weight().isDefault() && scalarizationWeight_ == 1)
        {
          useDefaultScalarWeight_ = true;
          data_->scalarWeight_ = 1;
          data_->minusScalarWeight_ = -1;
        }
        else
        {
          useDefaultScalarWeight_ = false;
          data_->scalarWeight_ = std::sqrt(requirements_->weight().value()) * scalarizationWeight_;
          data_->minusScalarWeight_ = -data_->scalarWeight_;
        }
        if(requirements_->anisotropicWeight().isDefault())
        {
          useDefaultAnisotropicWeight_ = true;
        }
        else
        {
          useDefaultAnisotropicWeight_ = false;
          data_->anisotropicWeight_ = requirements_->anisotropicWeight().value().cwiseSqrt();
          data_->anisotropicWeight_ *= data_->scalarWeight_;
          data_->minusAnisotropicWeight_ = -data_->anisotropicWeight_;
        }
        break;
      case requirements::ViolationEvaluationType::LINF:
        throw std::runtime_error("Unimplemented violation evaluation type.");
    }
  }
  else
  {
    useDefaultScalarWeight_ = true;
    useDefaultAnisotropicWeight_ = true;
  }
}

void Assignment::addMatrixAssignment(Variable & x, MatrixFunction M, const Range & range, bool flip)
{
  MatrixConstRef from = source_->jacobian(x);
  const MatrixRef & to = (target_.*M)(range.start, range.dim);
  auto w = createAssignment<Eigen::MatrixXd, AssignType::COPY>(from, to, flip);

  matrixAssignments_.push_back({w, &x, range, M});
}

void Assignment::addMatrixSubstitutionAssignments(const VariableVector & variables,
                                                  Variable & x,
                                                  MatrixFunction M,
                                                  const function::BasicLinearFunction & sub,
                                                  bool flip)
{
  auto J = source_->jacobian(x);
  for(const auto & xi : sub.variables())
  {
    Range range = xi->getMappingIn(variables);
    const MatrixRef & to = (target_.*M)(range.start, range.dim);
    auto Ai = sub.jacobian(*xi);
    CompiledAssignmentWrapper<Eigen::MatrixXd> w;
    if(J.properties().isIdentity())
    {
      if(flip)
        w = createAssignment<Eigen::MatrixXd, SUB>(Ai, to);
      else
        w = createAssignment<Eigen::MatrixXd, ADD>(Ai, to);
    }
    else if(J.properties().isMinusIdentity())
    {
      if(flip)
        w = createAssignment<Eigen::MatrixXd, ADD>(Ai, to);
      else
        w = createAssignment<Eigen::MatrixXd, SUB>(Ai, to);
    }
    else
    {
      if(flip)
        w = createMultiplicationAssignment<Eigen::MatrixXd, SUB>(J, to, Ai);
      else
        w = createMultiplicationAssignment<Eigen::MatrixXd, ADD>(J, to, Ai);
    }
    // Possible optimizations:
    // - detect more cases
    // - use CUSTOM multiplications
    // - in particular use the nullspace custom multiplication for variable z
    // when appropriate.
    matrixSubstitutionAssignments_.push_back({w, xi.get(), range, M});
  }
}

void Assignment::addVectorSubstitutionAssignments(const function::BasicLinearFunction & sub,
                                                  VectorFunction v,
                                                  Variable & x,
                                                  bool flip)
{
  bool useSource = !sub.b().properties().isConstant() || !sub.b().properties().isZero();
  if(useSource)
  {
    // The constraint has the form A x op b and we replace x by C y + d. We thus
    // want to add -Ad to b.
    // So far, the sign flip has been deduced only from the ConstraintType of the source
    // and the target. We still need to take into account the convention of the target.
    if(target_.constraintRhs() == RHS::AS_GIVEN)
      flip = !flip;

    const VectorRef & to = (target_.*v)();
    const VectorConstRef & from = sub.b();
    auto A = source_->jacobian(x);

    CompiledAssignmentWrapper<Eigen::VectorXd> w;
    if(A.properties().isIdentity())
    {
      if(flip)
        w = createAssignment<Eigen::VectorXd, SUB>(from, to);
      else
        w = createAssignment<Eigen::VectorXd, ADD>(from, to);
    }
    else if(A.properties().isMinusIdentity())
    {
      if(flip)
        w = createAssignment<Eigen::VectorXd, ADD>(from, to);
      else
        w = createAssignment<Eigen::VectorXd, SUB>(from, to);
    }
    else
    {
      if(flip)
        w = createMultiplicationAssignment<Eigen::VectorXd, SUB>(from, to, A);
      else
        w = createMultiplicationAssignment<Eigen::VectorXd, ADD>(from, to, A);
    }
    vectorSubstitutionAssignments_.push_back({w, v});
  }
}

void Assignment::addZeroAssignment(Variable & x, MatrixFunction M, const Range & range)
{
  const MatrixRef & to = (target_.*M)(range.start, range.dim);
  auto w = CompiledAssignmentWrapper<Eigen::MatrixXd>::make<COPY, NONE, IDENTITY, ZERO>(to);

  matrixAssignments_.push_back({w, &x, range, M});
}

void Assignment::addAssignments(const VariableVector & variables,
                                MatrixFunction M,
                                RHSFunction f,
                                VectorFunction v,
                                bool flip)
{
  addAssignments(variables, M, f, v, nullptr, nullptr, flip);
}

void Assignment::addAssignments(const VariableVector & variables,
                                MatrixFunction M,
                                RHSFunction f1,
                                VectorFunction v1,
                                RHSFunction f2,
                                VectorFunction v2)
{
  addAssignments(variables, M, f1, v1, f2, v2, false);
}

void Assignment::addAssignments(const VariableVector & variables,
                                MatrixFunction M,
                                RHSFunction f1,
                                VectorFunction v1,
                                RHSFunction f2,
                                VectorFunction v2,
                                bool flip)
{
  const auto & xs = substitutedVariables_; // substituted variables
  const auto & xc = source_->variables();  // variables of the source constraint

  addVectorAssignment(f1, v1, flip);
  if(f2)
  {
    assert(v2);
    addVectorAssignment(f2, v2, flip);
  }

  // For all variables that are substituted into but are not part of the source variables,
  // we set the corresponding matrix block to zero.
  VariableVector substitutedInto;
  for(const auto & x : xc)
  {
    int i = xs.indexOf(*x);
    if(i >= 0) // x needs to be substituted
    {
      substitutedInto.add(variableSubstitutions_[static_cast<int>(i)]->variables());
    }
  }
  for(const auto & x : substitutedInto)
  {
    if(!xc.contains(*x))
    {
      Range cols = x->getMappingIn(variables);
      addZeroAssignment(*x, M, cols);
    }
  }

  // Each variable of xc needs to be either in variables or in xs.
  // In the former case, we create a normal copy assignment, in the latter we
  // need to carry out a substitution
  for(const auto & x : xc)
  {
    if(x->size() == 0)
      continue;
    int i = xs.indexOf(*x);
    if(i >= 0) // x needs to be substituted
    {
      const auto & sub = *variableSubstitutions_[static_cast<int>(i)];
      addMatrixSubstitutionAssignments(variables, *x, M, sub, flip);
      addVectorSubstitutionAssignments(sub, v1, *x, flip);
      if(f2)
        addVectorSubstitutionAssignments(sub, v2, *x, flip);
    }
    else // usual case
    {
      Range cols = x->getMappingIn(variables);
      addMatrixAssignment(*x, M, cols, flip);
    }
  }
}

void Assignment::addBound(const VariablePtr & variable, RHSFunction f, bool first)
{
  auto J = source_->jacobian(*variable);
  bool gt = target_.constraintType() == Type::GREATER_THAN;
  VectorFunction v;

  if(gt)
    v = &AssignmentTarget::l;
  else
    v = &AssignmentTarget::u;

  if(substitutedVariables_.contains(*variable))
  {
    throw std::runtime_error("Substitution is not yet implemented for bounds");
  }
  else
  {
    if(J.properties().isIdentity())
    {
      assert(target_.constraintType() == source_->type());
      if(first)
        addVectorAssignment(f, v, false);
      else
        gt ? addVectorAssignment<MAX>(f, v, false) : addVectorAssignment<MIN>(f, v, false);
    }
    else if(J.properties().isMinusIdentity())
    {
      assert(target_.constraintType() != source_->type());
      if(J.properties().isIdentity())
      {
        if(first)
          addVectorAssignment(f, v, true);
        else
          gt ? addVectorAssignment<MAX>(f, v, true) : addVectorAssignment<MIN>(f, v, true);
      }
    }
    else
    {
      assert(J.properties().isDiagonal());
      if(first)
        addVectorAssignment(f, v, J, false);
      else
        gt ? addVectorAssignment<MAX>(f, v, J, false) : addVectorAssignment<MIN>(f, v, J, false);
    }
  }
}
} // namespace tvm::scheme::internal
