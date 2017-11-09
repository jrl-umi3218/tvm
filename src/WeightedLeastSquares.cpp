#include <iostream>

#include "LinearizedControlProblem.h"
#include "LinearizedTaskConstraint.h"
#include "WeightedLeastSquares.h"

namespace tvm
{
  namespace scheme
  {
    using VET = ViolationEvaluationType;
    WeightedLeastSquares::WeightedLeastSquares(std::shared_ptr<LinearizedControlProblem> pb, double scalarizationWeight)
      : LinearResolutionScheme({ 2, {{0, {true, {VET::L2}}}, {1,{false, {VET::L2}}}}, true }, pb)
      , scalarizationWeight_(scalarizationWeight)
    {
      build();
    }

    void WeightedLeastSquares::solve_()
    {
      for (auto& a : assignments_)
        a.run();

      std::cout << "A =\n" << memory_->A << std::endl;
      std::cout << "b = " << memory_->b.transpose() << std::endl;
      std::cout << "C =\n" << memory_->C << std::endl;
      std::cout << "l = " << memory_->u.transpose() << std::endl;
      std::cout << "u = " << memory_->l.transpose() << std::endl;

    }

    void WeightedLeastSquares::build()
    {
      const auto& constraints = problem_->constraints();

      //scanning constraints
      int m0 = 0;
      int m1 = 0;
      for (auto c : constraints)
      {
        abilities_.check(c.constraint, c.requirements); //FIXME: should be done in a parent class
        addVariable(c.constraint->variables()); //FIXME: idem

        if (c.requirements->priorityLevel().value() == 0)
          m0 += c.constraint->size();
        else
          m1 += c.constraint->size();  //note: we cannot have double sided constraints at this level.
      }

      //allocating memory for the solver
      memory_ = std::shared_ptr<Memory>(new Memory(x_.size(), m0, m1, big_number_));

      //assigments
      m0 = 0;
      m1 = 0;
      for (auto c : constraints)
      {
        int p = c.requirements->priorityLevel().value();
        if (p == 0)
        {
          RangePtr r = std::make_shared<Range>(m0, c.constraint->size()); //FIXME: for now we do not keep a pointer on the range nor the target.
          AssignmentTarget target(r, {memory_, &memory_->C}, { memory_, &memory_->l }, { memory_, &memory_->u }, ConstraintRHS::AS_GIVEN, x_.size());
          assignments_.emplace_back(Assignment(c.constraint, c.requirements, target, x_));
          m0 += c.constraint->size();
        }
        else
        {
          RangePtr r = std::make_shared<Range>(m1, c.constraint->size()); //FIXME: for now we do not keep a pointer on the range nor the target.
          AssignmentTarget target(r, { memory_, &memory_->A }, { memory_, &memory_->b }, ConstraintType::EQUAL, ConstraintRHS::AS_GIVEN);
          assignments_.emplace_back(Assignment(c.constraint, c.requirements, target, x_, std::pow(scalarizationWeight_,p-1)));
          m1 += c.constraint->size();
        }
      }
    }

    WeightedLeastSquares::Memory::Memory(int n, int m0, int m1, double big_number)
      : A(Eigen::MatrixXd::Zero(m1, n))
      , C(Eigen::MatrixXd::Zero(m0, n))
      , b(Eigen::VectorXd::Zero(m1))
      , l(Eigen::VectorXd::Constant(m0 + n, -big_number))
      , u(Eigen::VectorXd::Constant(m0 + n, +big_number))
    {
    }
  }
}