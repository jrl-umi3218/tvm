#include <tvm/scheme/internal/Assignment.h>

#include <tvm/defs.h>
#include <tvm/constraint/abstract/Constraint.h>
#include <tvm/constraint/abstract/LinearConstraint.h>
#include <tvm/VariableVector.h>

namespace tvm
{

namespace scheme
{

namespace internal
{

  double Assignment::big_ = constant::big_number;

  Assignment::Assignment(LinearConstraintPtr source, std::shared_ptr<requirements::SolvingRequirements> req,
                         const AssignmentTarget& target, const VariableVector& variables, double scalarizationWeight)
    : source_(source)
    , target_(target)
    , scalarizationWeight_(scalarizationWeight)
    , requirements_(req)
  {
    if (!checkTarget())
      throw std::runtime_error("target conventions are not compatible with the source.");
    //TODO check also that the variables of source are in the variable vector
    processRequirements();
    build(variables);
  }

  Assignment::Assignment(LinearConstraintPtr source, const AssignmentTarget& target, const VariablePtr& variable, bool first)
    : source_(source)
    , target_(target)
    , first_(first)
    , requirements_(new requirements::SolvingRequirements())
  {
    if (!checkTarget())
      throw std::runtime_error("target conventions are not compatible with the source.");
    assert(source->variables()[0] == variable);
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
    for (auto& a : vectorAssignments_)
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

    for (auto& a : vectorAssignments_)
      a.assignment.run();
  }

  bool Assignment::checkTarget()
  {
    //TODO implement checks
    // - compatibility of conventions
    // - size
    return true;
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
        addVectorAssignment(&constraint::abstract::LinearConstraint::e, &AssignmentTarget::b, false);
        break;
      case constraint::Type::DOUBLE_SIDED:
        addVectorAssignment(&constraint::abstract::LinearConstraint::e, &AssignmentTarget::l, false);
        addVectorAssignment(&constraint::abstract::LinearConstraint::e, &AssignmentTarget::u, false);
        break;
      default:
        throw std::runtime_error("Impossible to assign source for the given target convention.");
      }
      for (const auto& x : source_->variables())
      {
        Range cols = x->getMappingIn(variables);
        addMatrixAssignment(x.get(), &AssignmentTarget::A, cols, false);
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
          for (const auto& x : source_->variables())
          {
            Range cols = x->getMappingIn(variables);
            addMatrixAssignment(x.get(), &AssignmentTarget::A, cols, false);
          }
          addVectorAssignment(&constraint::abstract::LinearConstraint::l, &AssignmentTarget::l, false);
          addVectorAssignment(&constraint::abstract::LinearConstraint::u, &AssignmentTarget::u, false);
        }
        else
        {
          // case 7 and 8
          for (const auto& x : source_->variables())
          {
            Range cols = x->getMappingIn(variables);
            addMatrixAssignment(x.get(), &AssignmentTarget::AFirstHalf, cols, false);
            addMatrixAssignment(x.get(), &AssignmentTarget::ASecondHalf, cols, true);
          }
          if (target_.constraintType() == constraint::Type::GREATER_THAN)
          {
            //case 7
            addVectorAssignment(&constraint::abstract::LinearConstraint::l, &AssignmentTarget::bFirstHalf, false);
            addVectorAssignment(&constraint::abstract::LinearConstraint::u, &AssignmentTarget::bSecondHalf, true);
          }
          else
          {
            //case 8
            addVectorAssignment(&constraint::abstract::LinearConstraint::u, &AssignmentTarget::bFirstHalf, false);
            addVectorAssignment(&constraint::abstract::LinearConstraint::l, &AssignmentTarget::bSecondHalf, true);
          }
        }
      }
      else
      {
        // case 4 and 5 are just opposite of case 1 and 2.
        bool flip = source_->type() == constraint::Type::LOWER_THAN;
        RHSFunction f;
        if (source_->type() == constraint::Type::LOWER_THAN)
          f = &constraint::abstract::LinearConstraint::u;   //for case 1 and 2
        else
          f = &constraint::abstract::LinearConstraint::l;   //for case 4 and 5

        switch (target_.constraintType())
        {
        case constraint::Type::EQUAL:
          throw std::runtime_error("Impossible to assign inequality source for equality target.");
        case constraint::Type::GREATER_THAN:
          addVectorAssignment(f, &AssignmentTarget::b, flip);  // RHS for cases 1 and 4
          break;
        case constraint::Type::LOWER_THAN:
          addVectorAssignment(f, &AssignmentTarget::b, !flip);  // RHS for cases 2 and 5
          break;
        case constraint::Type::DOUBLE_SIDED:
          flip = false; // for case 3 and 6, the signe of A and C are the same
          if (source_->type() == constraint::Type::GREATER_THAN)
          {
            //case 3
            addVectorAssignment(f, &AssignmentTarget::l, false);
            addConstantAssignment(big_, &AssignmentTarget::u);
          }
          else
          {
            //case 5
            addVectorAssignment(f, &AssignmentTarget::u, false);
            addConstantAssignment(-big_, &AssignmentTarget::l);
          }
          break;
        }
        //Matrics for case 1 to 6
        for (const auto& x : source_->variables())
        {
          Range cols = x->getMappingIn(variables);
          addMatrixAssignment(x.get(), &AssignmentTarget::A, cols, flip);
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
        addVectorAssignment(&constraint::abstract::LinearConstraint::e, l, flip);
        addVectorAssignment(&constraint::abstract::LinearConstraint::e, u, flip);
        break;
      case constraint::Type::GREATER_THAN:
        addVectorAssignment(&constraint::abstract::LinearConstraint::l, l, flip);
        addConstantAssignment(flip?-big_:+big_, u);
        break;
      case constraint::Type::LOWER_THAN:
        addConstantAssignment(flip?+big_:-big_, l);
        addVectorAssignment(&constraint::abstract::LinearConstraint::u, u, flip);
        break;
      case constraint::Type::DOUBLE_SIDED:
        addVectorAssignment(&constraint::abstract::LinearConstraint::l, l, flip);
        addVectorAssignment(&constraint::abstract::LinearConstraint::u, u, flip);
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
          addVectorAssignment<AssignType::MIN>(&constraint::abstract::LinearConstraint::e, l, flip);
          addVectorAssignment<AssignType::MAX>(&constraint::abstract::LinearConstraint::e, u, flip);
          break;
        case constraint::Type::GREATER_THAN:
          addVectorAssignment<AssignType::MIN>(&constraint::abstract::LinearConstraint::l, l, flip);
          addConstantAssignment<AssignType::MAX>(-big_, u);
          break;
        case constraint::Type::LOWER_THAN:
          addConstantAssignment<AssignType::MIN>(+big_, l);
          addVectorAssignment<AssignType::MAX>(&constraint::abstract::LinearConstraint::u, u, flip);
          break;
        case constraint::Type::DOUBLE_SIDED:
          addVectorAssignment<AssignType::MIN>(&constraint::abstract::LinearConstraint::l, l, flip);
          addVectorAssignment<AssignType::MAX>(&constraint::abstract::LinearConstraint::u, u, flip);
          break;
        }
      }
      else
      {
        switch (source_->type())
        {
        case constraint::Type::EQUAL:
          addVectorAssignment<AssignType::MAX>(&constraint::abstract::LinearConstraint::e, l, flip);
          addVectorAssignment<AssignType::MIN>(&constraint::abstract::LinearConstraint::e, u, flip);
          break;
        case constraint::Type::GREATER_THAN:
          addVectorAssignment<AssignType::MAX>(&constraint::abstract::LinearConstraint::l, l, flip);
          addConstantAssignment<AssignType::MIN>(+big_, u);
          break;
        case constraint::Type::LOWER_THAN:
          addConstantAssignment<AssignType::MAX>(-big_, l);
          addVectorAssignment<AssignType::MIN>(&constraint::abstract::LinearConstraint::u, u, flip);
          break;
        case constraint::Type::DOUBLE_SIDED:
          addVectorAssignment<AssignType::MAX>(&constraint::abstract::LinearConstraint::l, l, flip);
          addVectorAssignment<AssignType::MIN>(&constraint::abstract::LinearConstraint::u, u, flip);
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

}  // namespace internal

}  // namespace scheme

}  // namespace tvm
