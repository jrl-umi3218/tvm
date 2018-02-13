#include <tvm/scheme/WeightedLeastSquares.h>

#include <tvm/LinearizedControlProblem.h>
#include <tvm/constraint/internal/LinearizedTaskConstraint.h>
#include <tvm/constraint/abstract/LinearConstraint.h>

#include <iostream>

namespace tvm
{

namespace scheme
{
  using namespace internal;
  using VET = requirements::ViolationEvaluationType;

  WeightedLeastSquares::WeightedLeastSquares(bool verbose, double scalarizationWeight)
    : LinearResolutionScheme<WeightedLeastSquares>({ 2, {{0, {true, {VET::L2}}}, {1,{false, {VET::L2}}}}, true })
    , verbose_(verbose), scalarizationWeight_(scalarizationWeight)
  {
  }

  bool WeightedLeastSquares::solve_(LinearizedControlProblem& problem, Memory& memory) const
  {
    memory.A.setZero();
    for (auto& a : memory.assignments)
      a.run();

    if(verbose_)
    {
      std::cout << "A =\n" << memory.A << std::endl;
      std::cout << "b = " << memory.b.transpose() << std::endl;
      std::cout << "C =\n" << memory.C << std::endl;
      std::cout << "l = " << memory.l.transpose() << std::endl;
      std::cout << "u = " << memory.u.transpose() << std::endl;
    }

    bool b = memory.ls.solve(memory.A, memory.b, memory.C, memory.l, memory.u);
    memory.setSolution(memory.ls.result());
    if(verbose_ || !b)
    {
      std::cout << memory.ls.inform() << std::endl;
      memory.ls.print_inform();
      if(verbose_)
      {
        std::cout << memory.ls.result().transpose() << std::endl;
      }
    }
    return b;
  }

  std::unique_ptr<WeightedLeastSquares::Memory> WeightedLeastSquares::createComputationData_(const LinearizedControlProblem& problem) const
  {
    auto memory = std::unique_ptr<Memory>(new Memory(id()));
    const auto& constraints = problem.constraints();

    //scanning bounds
    for (auto b : problem.bounds())
    {
      abilities_.check(b.constraint, b.requirements); //FIXME: should be done in a parent class
      memory->addVariable(b.constraint->variables()); //FIXME: should be done in a parent class
    }

    //scanning constraints
    int m0 = 0;
    int m1 = 0;
    for (auto c : constraints)
    {
      abilities_.check(c.constraint, c.requirements); //FIXME: should be done in a parent class
      memory->addVariable(c.constraint->variables()); //FIXME: should be done in a parent class

      if (c.requirements->priorityLevel().value() == 0)
        m0 += c.constraint->size();
      else
      {
        if (c.constraint->type() != constraint::Type::EQUAL)
          throw std::runtime_error("This scheme do not handle inequality constraints with priority level > 0.");
        m1 += c.constraint->size();  //note: we cannot have double sided constraints at this level.
      }
    }

    if (m1 == 0)
      m1 = memory->variables().size();

    //allocating memory for the solver
    memory->resize(m0, m1, big_number_);

    // configure assignments. FIXME: can we find a better way ?
    Assignment::big_ = big_number_;

    //assigments for general constraints
    m0 = 0;
    m1 = 0;
    const auto& x = memory->variables();
    auto l = memory->l.tail(memory->C.rows());
    auto u = memory->u.tail(memory->C.rows());
    for (auto c : constraints)
    {
      int p = c.requirements->priorityLevel().value();
      if (p == 0)
      {
        RangePtr r = std::make_shared<Range>(m0, c.constraint->size()); //FIXME: for now we do not keep a pointer on the range nor the target.
        AssignmentTarget target(r, memory->C, l, u, constraint::RHS::AS_GIVEN);
        memory->assignments.emplace_back(Assignment(c.constraint, c.requirements, target, x));
        m0 += c.constraint->size();
      }
      else
      {
        RangePtr r = std::make_shared<Range>(m1, c.constraint->size()); //FIXME: for now we do not keep a pointer on the range nor the target.
        AssignmentTarget target(r, memory->A, memory->b, constraint::Type::EQUAL, constraint::RHS::AS_GIVEN);
        memory->assignments.emplace_back(Assignment(c.constraint, c.requirements, target, x, std::pow(scalarizationWeight_, p - 1)));
        m1 += c.constraint->size();
      }
    }
    if (m1 == 0)
    {
      memory->A.setIdentity();
      memory->b.setZero();
    }

    //assigments for bounds
    std::map<Variable*, bool> first;
    for (const auto& xi : x.variables())
    {
      first[xi.get()] = true;
    }
    for (auto b : problem.bounds())
    {
      const auto& xi = b.constraint->variables()[0];
      int p = b.requirements->priorityLevel().value();
      if (p == 0)
      {
        RangePtr range = std::make_shared<Range>(xi->getMappingIn(x)); //FIXME: for now we do not keep a pointer on the range nor the target.
        AssignmentTarget target(range, memory->l, memory->u);
        memory->assignments.emplace_back(Assignment(b.constraint, target, xi, first[xi.get()]));
        first[xi.get()] = false;
      }
      else
      {
        throw std::runtime_error("This scheme do not handle inequality constraints with priority level > 0.");
      }
    }

    return memory;
  }

  WeightedLeastSquares::Memory::Memory(int solverId)
    : ProblemComputationData(solverId)
    , basePtr(new int)
  {
  }

  void WeightedLeastSquares::Memory::resize(int m0, int m1, double big_number)
  {
    int n = x_.size();
    A.resize(m1, n);
    A.setZero();
    C.resize(m0, n);
    C.setZero();
    b.resize(m1);
    b.setZero();
    l = Eigen::VectorXd::Constant(m0 + n, -big_number);
    u = Eigen::VectorXd::Constant(m0 + n, +big_number);
    ls.resize(n, m0, Eigen::lssol::eType::LS1);
  }

}  // namespace scheme

}  // namespace tvm
