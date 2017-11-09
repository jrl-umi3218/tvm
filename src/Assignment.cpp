#include "Assignment.h"
#include "LinearConstraint.h"
#include "Variable.h"
#include "VariableVector.h"

namespace tvm
{
  Assignment::Assignment(LinearConstraintPtr source, std::shared_ptr<SolvingRequirements> req,
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
    //FIXME move at a better place
    static const double large = 1e6;

    // In this function, we split up the global assignment into atomic assignments.
    // This is done according to the ConstraintType of both source and target
    if (source_->constraintType() == ConstraintType::EQUAL)
    {
      /*
       *                 |  Ax = b
       * ----------------+--------------
       *      Cx = d     | C=A, d=b
       *  dl <= Cx <= du | C=A, dl=du=b
       */

      switch (target_.constraintType())
      {
      case ConstraintType::EQUAL:
        addVectorAssignment(&LinearConstraint::e, &AssignmentTarget::b, false);
        break;
      case ConstraintType::DOUBLE_SIDED:
        addVectorAssignment(&LinearConstraint::e, &AssignmentTarget::l, false);
        addVectorAssignment(&LinearConstraint::e, &AssignmentTarget::u, false);
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

      if (source_->constraintType() == ConstraintType::DOUBLE_SIDED)
      {
        if (target_.constraintType() == ConstraintType::DOUBLE_SIDED)
        {
          //case 9
          for (const auto& x : variables.variables())
          {
            Range cols = x->getMappingIn(variables);
            addMatrixAssignment(x.get(), &AssignmentTarget::A, cols, false);
          }
          addVectorAssignment(&LinearConstraint::l, &AssignmentTarget::l, false);
          addVectorAssignment(&LinearConstraint::u, &AssignmentTarget::u, false);
        }
        else
        {
          // case 7 and 8
          for (const auto& x : variables.variables())
          {
            Range cols = x->getMappingIn(variables);
            addMatrixAssignment(x.get(), &AssignmentTarget::AFirstHalf, cols, false);
            addMatrixAssignment(x.get(), &AssignmentTarget::ASecondHalf, cols, true);
          }
          if (target_.constraintType() == ConstraintType::GREATER_THAN)
          {
            //case 7
            addVectorAssignment(&LinearConstraint::l, &AssignmentTarget::bFirstHalf, false);
            addVectorAssignment(&LinearConstraint::u, &AssignmentTarget::bSecondHalf, true);
          }
          else
          {
            //case 8
            addVectorAssignment(&LinearConstraint::u, &AssignmentTarget::bFirstHalf, false);
            addVectorAssignment(&LinearConstraint::l, &AssignmentTarget::bSecondHalf, true);
          }
        }
      }
      else
      {
        // case 4 and 5 are just opposite of case 1 and 2.
        bool flip = source_->constraintType() == ConstraintType::LOWER_THAN;
        RHSFunction f;
        if (source_->constraintType() == ConstraintType::LOWER_THAN)
          f = &LinearConstraint::u;   //for case 1 and 2
        else
          f = &LinearConstraint::l;   //for case 4 and 5

        switch (target_.constraintType())
        {
        case ConstraintType::EQUAL:
          throw std::runtime_error("Impossible to assign inequality source for equality target.");
        case ConstraintType::GREATER_THAN:
          addVectorAssignment(f, &AssignmentTarget::b, flip);  // RHS for cases 1 and 4
          break;
        case ConstraintType::LOWER_THAN:
          addVectorAssignment(f, &AssignmentTarget::b, !flip);  // RHS for cases 2 and 5
          break;
        case ConstraintType::DOUBLE_SIDED:
          flip = false; // for case 3 and 6, the signe of A and C are the same
          if (source_->constraintType() == ConstraintType::GREATER_THAN)
          {
            //case 3
            addVectorAssignment(f, &AssignmentTarget::l, false);
            addConstantAssignment(large, &AssignmentTarget::u);
          }
          else
          {
            //case 5
            addVectorAssignment(f, &AssignmentTarget::u, false);
            addConstantAssignment(-large, &AssignmentTarget::l);
          }
          break;
        }
        //Matrics for case 1 to 6
        for (const auto& x : variables.variables())
        {
          Range cols = x->getMappingIn(variables);
          addMatrixAssignment(x.get(), &AssignmentTarget::A, cols, flip);
        }
      }
    }
  }

  void Assignment::processRequirements()
  {
    switch (requirements_->violationEvaluation().value())
    {
    case ViolationEvaluationType::L1: 
      throw std::runtime_error("Unimplemented violation evaluation type.");
    case ViolationEvaluationType::L2:
      scalarWeight_ = std::sqrt(requirements_->weight().value()) * scalarizationWeight_;
      if (!requirements_->anisotropicWeight().isDefault())
      {
        anisotropicWeight_ = requirements_->anisotropicWeight().value().cwiseSqrt();
        anisotropicWeight_ *= scalarWeight_;
        minusAnisotropicWeight_ = -anisotropicWeight_;
      }
      break;
    case ViolationEvaluationType::LINF:
      throw std::runtime_error("Unimplemented violation evaluation type.");
    }
  }

  void Assignment::addMatrixAssignment(Variable* x, MatrixFunction M, const Range& range, bool flip)
  {
    const MatrixConstRef& from = source_->jacobian(*x);
    const MatrixRef& to = (target_.*M)(range.start, range.dim);
    auto w = createAssignment<Eigen::MatrixXd>(from, to, flip);

    matrixAssignments_.push_back({ w, x, range, M });
  }

  void Assignment::addVectorAssignment(RHSFunction f, VectorFunction v, bool flip)
  {
    const VectorRef& to = (target_.*v)();

    bool useSource = source_->constraintRhs() != ConstraintRHS::ZERO;
    if (useSource)
    {
      // So far, the sign flip has been deduced only from the ConstraintType of the source
      // and the target. Now we need to take into account the ConstraintRHS as well.
      if (source_->constraintRhs() == ConstraintRHS::OPPOSITE)
        flip = !flip;
      if (target_.constraintRhs() == ConstraintRHS::OPPOSITE)
        flip = !flip;

      const VectorConstRef& from = (source_.get()->*f)();
      auto w = createAssignment<Eigen::VectorXd>(from, to, flip);
      vectorAssignments_.push_back({ w, true, f, v });
    }
    else
    {
      auto w = utils::CompiledAssignmentWrapper<Eigen::VectorXd>::make<utils::COPY>(to);
      vectorAssignments_.push_back({ w, false, nullptr, v });
    }
  }

  void Assignment::addConstantAssignment(double d, VectorFunction v)
  {
    const VectorRef& to = (target_.*v)();
    auto w = createAssignment<Eigen::VectorXd>(d, to, false);
    vectorAssignments_.push_back({ w, false, nullptr, v });
  }

}
