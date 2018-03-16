#include <tvm/scheme/internal/Assignment.h>

#include <tvm/defs.h>
#include <tvm/constraint/abstract/Constraint.h>
#include <tvm/VariableVector.h>

namespace tvm
{

namespace scheme
{

namespace internal
{
  using constraint::abstract::LinearConstraint;
  using constraint::Type;
  using constraint::RHS;

  double Assignment::big_ = constant::big_number;

  Assignment::Assignment(LinearConstraintPtr source, 
                         std::shared_ptr<requirements::SolvingRequirements> req,
                         const AssignmentTarget& target, 
                         const VariableVector& variables, 
                         const hint::internal::Substitutions& substitutions,
                         double scalarizationWeight)
    : source_(source)
    , target_(target)
    , scalarizationWeight_(scalarizationWeight)
    , requirements_(req)
    , substitutedVariables_(substitutions.variables())
    , variableSubstitutions_(substitutions.variableSubstitutions())
  {
    checkTarget();
    //TODO check also that the variables of source are in the variable vector
    processRequirements();
    build(variables);
  }

  Assignment::Assignment(LinearConstraintPtr source, const AssignmentTarget& target, const VariablePtr& variable, bool first)
    : source_(source)
    , target_(target)
    , requirements_(new requirements::SolvingRequirements())
    , first_(first)
  {
    checkTarget();
    assert(source->variables()[0] == variable);
    scalarWeight_ = 1;
    build(variable);
  }

  void Assignment::onUpdatedSource()
  {
    //TODO
  }

  void Assignment::onUpdatedTarget()
  {
    for (auto& a : matrixAssignments_)
      a.assignment.to((target_.*a.getTargetMatrix)(a.colRange.start, a.colRange.dim));
    for (auto& a : matrixSubstitutionAssignments_)
      a.assignment.to((target_.*a.getTargetMatrix)(a.colRange.start, a.colRange.dim));
    for (auto& a : vectorAssignments_)
      a.assignment.to((target_.*a.getTargetVector)());
    for (auto& a : vectorSubstitutionAssignments_)
      a.assignment.to((target_.*a.getTargetVector)());
  }

  void Assignment::onUpdatedMapping(const VariableVector& /*variables*/)
  {
    //TODO
  }

  void Assignment::weight(double /*alpha*/)
  {
    //TODO
  }

  void Assignment::weight(const Eigen::VectorXd& /*w*/)
  {
    //TODO
  }

  void Assignment::run()
  {
    for (auto& a : matrixAssignments_)
      a.assignment.run();

    for (auto& a : matrixSubstitutionAssignments_)
      a.assignment.run();

    for (auto& a : vectorAssignments_)
      a.assignment.run();

    for (auto& a : vectorSubstitutionAssignments_)
      a.assignment.run();
  }

  void Assignment::checkTarget()
  {
    if (source_->type() == Type::EQUAL)
    {
      if (target_.constraintType() != Type::EQUAL
        && target_.constraintType() != Type::DOUBLE_SIDED)
        throw std::runtime_error("Incompatible conventions: source with type EQUAL can only "
          "be assigned to target with type EQUAL or DOUBLE SIDED.");
    }
    else
    {
      if (target_.constraintType() == Type::EQUAL)
        throw std::runtime_error("Incompatible conventions: source with type other than EQUAL "
          "cannot be assigned to a target with type EQUAL. ");
    }

    // check the rhs conventions
    if (source_->rhs() != RHS::ZERO && target_.constraintRhs() == RHS::ZERO)
       throw std::runtime_error("Incompatible conventions: source with rhs other than ZERO "
          "cannot be assigned to target with rhs ZERO.");

    //check the sizes
    if (source_->type() == Type::DOUBLE_SIDED)
    {
      if (target_.constraintType() == Type::DOUBLE_SIDED)
      {
        if (target_.size() != source_->size())
          throw std::runtime_error("Size of the source and the target are not coherent with "
            "the conventions.");
      }
      else
      {
        if (target_.size() != 2*source_->size())
          throw std::runtime_error("Size of the source and the target are not coherent with "
            "the conventions.");
      }
    }
    else
    {
      if (target_.size() != source_->size())
          throw std::runtime_error("Size of the source and the target are not coherent with "
            "the conventions.");
    }
  }

  void Assignment::build(const VariableVector& variables)
  {
    // In this function, we split up the global assignment into atomic assignments.
    // This is done according to the ConstraintType of both source and target
    if (source_->type() == constraint::Type::EQUAL)
    {
      /*
       *                 |  Ax = b
       * ----------------+--------------
       *      Cx = d     | C=A, d=b
       *  dl <= Cx <= du | C=A, dl=du=b
       */

      switch (target_.constraintType())
      {
      case constraint::Type::EQUAL:
        addAssignments(variables, &AssignmentTarget::A,
                       &LinearConstraint::e, &AssignmentTarget::b, false);
        break;
      case constraint::Type::DOUBLE_SIDED:
        addAssignments(variables, &AssignmentTarget::A,
                       &LinearConstraint::e, &AssignmentTarget::l,
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

      if (source_->type() == constraint::Type::DOUBLE_SIDED)
      {
        if (target_.constraintType() == constraint::Type::DOUBLE_SIDED)
        {
          //case 9
          addAssignments(variables, &AssignmentTarget::A,
                         &LinearConstraint::l, &AssignmentTarget::l,
                         &LinearConstraint::u, &AssignmentTarget::u);
        }
        else
        {
          if (target_.constraintType() == constraint::Type::GREATER_THAN)
          {
            //case 7
            addAssignments(variables, &AssignmentTarget::AFirstHalf,
                          &LinearConstraint::l, &AssignmentTarget::bFirstHalf, false);
            addAssignments(variables, &AssignmentTarget::ASecondHalf,
                           &LinearConstraint::u, &AssignmentTarget::bSecondHalf, true);
          }
          else
          {
            //case 8
            addAssignments(variables, &AssignmentTarget::AFirstHalf,
                           &LinearConstraint::u, &AssignmentTarget::bFirstHalf, false);
            addAssignments(variables, &AssignmentTarget::ASecondHalf,
                           &LinearConstraint::l, &AssignmentTarget::bSecondHalf, true);
          }
        }
      }
      else
      {
        // case 4 and 5 are just opposite of case 1 and 2.
        bool flip = source_->type() == constraint::Type::LOWER_THAN;
        RHSFunction f;
        if (source_->type() == constraint::Type::LOWER_THAN)
          f = &LinearConstraint::u;   //for case 1 and 2
        else
          f = &LinearConstraint::l;   //for case 4 and 5

        switch (target_.constraintType())
        {
        case constraint::Type::EQUAL:
          throw std::runtime_error("Impossible to assign inequality source for equality target.");
        case constraint::Type::GREATER_THAN:
          // cases 1 and 4
          addAssignments(variables, &AssignmentTarget::A, f, &AssignmentTarget::b, flip);
          break;
        case constraint::Type::LOWER_THAN:
          //cases 2 and 5
          addAssignments(variables, &AssignmentTarget::A, f, &AssignmentTarget::b, !flip);
          break;
        case constraint::Type::DOUBLE_SIDED:
          // for case 3 and 6, the signs of A and C are the same
          if (source_->type() == constraint::Type::GREATER_THAN)
          {
            //case 3
            addAssignments(variables, &AssignmentTarget::A, f, &AssignmentTarget::l, false);
            addConstantAssignment(big_, &AssignmentTarget::u);
          }
          else
          {
            //case 6
            addAssignments(variables, &AssignmentTarget::A, f, &AssignmentTarget::u, false);
            addConstantAssignment(-big_, &AssignmentTarget::l);
          }
          break;
        }
      }
    }
  }

  void Assignment::build(const VariablePtr& variable)
  {
    VectorFunction l;
    VectorFunction u;
    bool flip;
    if (source_->jacobian(*variable).properties().isIdentity())
    {
      l = &AssignmentTarget::l;
      u = &AssignmentTarget::u;
      flip = false;
    }
    else if (source_->jacobian(*variable).properties().isMinusIdentity())
    {
      l = &AssignmentTarget::u;
      u = &AssignmentTarget::l;
      flip = true;
    }
    else
    {
      throw std::runtime_error("Pure diagonal case is not implemented yet.");
    }

    if (first_)
    {
      switch (source_->type())
      {
      case constraint::Type::EQUAL:
        addVectorAssignment(&LinearConstraint::e, l, flip);
        addVectorAssignment(&LinearConstraint::e, u, flip);
        break;
      case constraint::Type::GREATER_THAN:
        addVectorAssignment(&LinearConstraint::l, l, flip);
        addConstantAssignment(flip?-big_:+big_, u);
        break;
      case constraint::Type::LOWER_THAN:
        addConstantAssignment(flip?+big_:-big_, l);
        addVectorAssignment(&LinearConstraint::u, u, flip);
        break;
      case constraint::Type::DOUBLE_SIDED:
        addVectorAssignment(&LinearConstraint::l, l, flip);
        addVectorAssignment(&LinearConstraint::u, u, flip);
        break;
      }
    }
    else
    {
      if (flip)
      {
        switch (source_->type())
        {
        case constraint::Type::EQUAL:
          addVectorAssignment<AssignType::MIN>(&LinearConstraint::e, l, flip);
          addVectorAssignment<AssignType::MAX>(&LinearConstraint::e, u, flip);
          break;
        case constraint::Type::GREATER_THAN:
          addVectorAssignment<AssignType::MIN>(&LinearConstraint::l, l, flip);
          addConstantAssignment<AssignType::MAX>(-big_, u);
          break;
        case constraint::Type::LOWER_THAN:
          addConstantAssignment<AssignType::MIN>(+big_, l);
          addVectorAssignment<AssignType::MAX>(&LinearConstraint::u, u, flip);
          break;
        case constraint::Type::DOUBLE_SIDED:
          addVectorAssignment<AssignType::MIN>(&LinearConstraint::l, l, flip);
          addVectorAssignment<AssignType::MAX>(&LinearConstraint::u, u, flip);
          break;
        }
      }
      else
      {
        switch (source_->type())
        {
        case constraint::Type::EQUAL:
          addVectorAssignment<AssignType::MAX>(&LinearConstraint::e, l, flip);
          addVectorAssignment<AssignType::MIN>(&LinearConstraint::e, u, flip);
          break;
        case constraint::Type::GREATER_THAN:
          addVectorAssignment<AssignType::MAX>(&LinearConstraint::l, l, flip);
          addConstantAssignment<AssignType::MIN>(+big_, u);
          break;
        case constraint::Type::LOWER_THAN:
          addConstantAssignment<AssignType::MAX>(-big_, l);
          addVectorAssignment<AssignType::MIN>(&LinearConstraint::u, u, flip);
          break;
        case constraint::Type::DOUBLE_SIDED:
          addVectorAssignment<AssignType::MAX>(&LinearConstraint::l, l, flip);
          addVectorAssignment<AssignType::MIN>(&LinearConstraint::u, u, flip);
          break;
        }
      }
    }
  }

  void Assignment::processRequirements()
  {
    switch (requirements_->violationEvaluation().value())
    {
    case requirements::ViolationEvaluationType::L1:
      throw std::runtime_error("Unimplemented violation evaluation type.");
    case requirements::ViolationEvaluationType::L2:
      scalarWeight_ = std::sqrt(requirements_->weight().value()) * scalarizationWeight_;
      if (!requirements_->anisotropicWeight().isDefault())
      {
        anisotropicWeight_ = requirements_->anisotropicWeight().value().cwiseSqrt();
        anisotropicWeight_ *= scalarWeight_;
        minusAnisotropicWeight_ = -anisotropicWeight_;
      }
      break;
    case requirements::ViolationEvaluationType::LINF:
      throw std::runtime_error("Unimplemented violation evaluation type.");
    }
  }

  void Assignment::addMatrixAssignment(Variable* x, MatrixFunction M, const Range& range, bool flip)
  {
    const MatrixConstRef& from = source_->jacobian(*x);
    const MatrixRef& to = (target_.*M)(range.start, range.dim);
    auto w = createAssignment<Eigen::MatrixXd, AssignType::COPY>(from, to, flip);

    matrixAssignments_.push_back({ w, x, range, M });
  }

  void Assignment::addAssignments(const VariableVector& variables, MatrixFunction M,
                                  RHSFunction f, VectorFunction v, bool flip)
  {
    for (const auto& x : source_->variables().variables())
    {
      Range cols = x->getMappingIn(variables);
      addMatrixAssignment(x.get(), M, cols, flip);
    }
    addVectorAssignment(f, v, flip);
  }

  void Assignment::addAssignments(const VariableVector& variables, MatrixFunction M,
                                  RHSFunction f1, VectorFunction v1,
                                  RHSFunction f2, VectorFunction v2)
  {
    for (const auto& x : source_->variables().variables())
    {
      Range cols = x->getMappingIn(variables);
      addMatrixAssignment(x.get(), M, cols, false);
    }
    addVectorAssignment(f1, v1, false);
    addVectorAssignment(f2, v2, false);
  }

}  // namespace internal

}  // namespace scheme

}  // namespace tvm
