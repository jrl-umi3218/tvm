#include <iostream>
#include <vector>

#define AUTHORIZE_MALLOC_FOR_CACHE
#include "CompiledAssignment.h"
#include "CompiledAssignmentWrapper.h"

using namespace Eigen;
using namespace tvm::utils;

MatrixXd runOperator(AssignType A, const MatrixXd& from, const MatrixXd& to)
{
  MatrixXd tmp;
  switch (A)
  {
  case REPLACE: return from; break;
  case ADD: return to + from; break;
  case SUB: return to - from; break;
  case MIN: tmp = to; tmp.array().min(from.array()); return tmp; break;
  case MAX: tmp = to; tmp.array().max(from.array()); return tmp; break;
  default: return{};
  }
}

MatrixXd scalarMult(ScalarMult S, double s, const MatrixXd& M)
{
  switch (S)
  {
  case NONE: return M; break;
  case MINUS: return -M; break;
  case SCALAR: return s*M; break;
  default: return{};
  }
}

MatrixXd matrixMult(MatrixMult M, MultPos P, const MatrixXd& A, const MatrixXd& wOrM)
{
  switch (M)
  {
  case IDENTITY: return A; break;
  case DIAGONAL: if (P == PRE) return wOrM.asDiagonal()*A; else return A*wOrM.asDiagonal(); break;
  case GENERAL: if (P == PRE) return wOrM*A; else return A*wOrM; break;
  default: return{};
  }
}


void assign(AssignType A, ScalarMult S, MatrixMult M, MultPos P,
  const Ref<const MatrixXd>& from, Ref<MatrixXd> to, double s = 0, const MatrixXd& wOrM = MatrixXd())
{
  MatrixXd tmp1 = matrixMult(M, P, from, wOrM);
  MatrixXd tmp2 = scalarMult(S, s, tmp1);
  MatrixXd tmp3 = runOperator(A, tmp2, to);
  to = tmp3;
}

void assign(AssignType A, Ref<MatrixXd> to)
{
  if (A == REPLACE)
    to.setZero();
}

template<AssignType A, ScalarMult S, MatrixMult M, MultPos P>
struct Test
{
  template<typename Derived1, typename U>
  static bool run(const MatrixBase<Derived1>& from, U& to)
  {
    assert(from.rows() == to.rows() && from.cols() == to.cols());

    typedef MatrixBase<Derived1> MatrixType;
    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };
    typedef typename MatrixType::Scalar Scalar;
    typedef Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, ColMajor, MaxRowsAtCompileTime, MaxColsAtCompileTime> Type;

    //generate possibly needed multipliers
    double s = 3;
    Eigen::DenseIndex r, c;
    if (M == DIAGONAL)
    {
      c = 1;
      if (P == PRE)
        r = from.rows();
      else
        r = from.cols();
    }
    else
    {
      if (P == PRE)
      {
        r = to.rows();
        c = from.rows();
      }
      else
      {
        r = from.cols();
        c = to.cols();
      }
    }
    typename std::conditional<M == DIAGONAL, VectorXd, MatrixXd>::type wOrM(r, c);
    wOrM.setRandom();

    MatrixXd f = from;
    MatrixXd t = to;

    Eigen::internal::set_is_malloc_allowed(false);
    CompiledAssignment<Type, A, S, M, P> ca(from, to, s, &wOrM);
    ca.run();
    Eigen::internal::set_is_malloc_allowed(true);
    assign(A, S, M, P, f, t, s, wOrM);

    return t.isApprox(to);
  }

  template<typename Derived1, typename U>
  static void runAndPrint(const MatrixBase<Derived1>& from, U& to)
  {
    std::cout << "run (" << A << ", " << S << ", " << M << ", " << P << ")    " << std::flush;
    bool b = run(from, to);
    std::cout << (b ? "ok" : "error") << std::endl;
  }
};

template<AssignType A>
struct TestNoFrom
{
  template<typename U>
  static bool run(U& to)
  {
    typedef U MatrixType;
    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };
    typedef typename MatrixType::Scalar Scalar;
    typedef Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, ColMajor, MaxRowsAtCompileTime, MaxColsAtCompileTime> Type;

    MatrixXd t = to;

    Eigen::internal::set_is_malloc_allowed(false);
    CompiledAssignment<Type, A, NONE, IDENTITY, POST, false> ca(to);
    ca.run();
    Eigen::internal::set_is_malloc_allowed(true);
    assign(A, t);

    return t.isApprox(to);
  }

  template<typename U>
  static void runAndPrint(U& to)
  {
    std::cout << "no from run " << A << "    " << std::flush;
    bool b = run(to);
    std::cout << (b ? "ok" : "error") << std::endl;
  }
};

template<typename U, typename V>
void testBatch(const U& from, V& to)
{
  Test<REPLACE, NONE, IDENTITY, PRE>::runAndPrint(from, to);
  Test<REPLACE, NONE, IDENTITY, POST>::runAndPrint(from, to);
  Test<REPLACE, NONE, DIAGONAL, PRE>::runAndPrint(from, to);
  Test<REPLACE, NONE, DIAGONAL, POST>::runAndPrint(from, to);
  Test<REPLACE, NONE, GENERAL, PRE>::runAndPrint(from, to);
  Test<REPLACE, NONE, GENERAL, POST>::runAndPrint(from, to);
  Test<REPLACE, MINUS, IDENTITY, PRE>::runAndPrint(from, to);
  Test<REPLACE, MINUS, IDENTITY, POST>::runAndPrint(from, to);
  Test<REPLACE, MINUS, DIAGONAL, PRE>::runAndPrint(from, to);
  Test<REPLACE, MINUS, DIAGONAL, POST>::runAndPrint(from, to);
  Test<REPLACE, MINUS, GENERAL, PRE>::runAndPrint(from, to);
  Test<REPLACE, MINUS, GENERAL, POST>::runAndPrint(from, to);
  Test<REPLACE, SCALAR, IDENTITY, PRE>::runAndPrint(from, to);
  Test<REPLACE, SCALAR, IDENTITY, POST>::runAndPrint(from, to);
  Test<REPLACE, SCALAR, DIAGONAL, PRE>::runAndPrint(from, to);
  Test<REPLACE, SCALAR, DIAGONAL, POST>::runAndPrint(from, to);
  Test<REPLACE, SCALAR, GENERAL, PRE>::runAndPrint(from, to);
  Test<REPLACE, SCALAR, GENERAL, POST>::runAndPrint(from, to);

  Test<ADD, NONE, IDENTITY, PRE>::runAndPrint(from, to);
  Test<ADD, NONE, IDENTITY, POST>::runAndPrint(from, to);
  Test<ADD, NONE, DIAGONAL, PRE>::runAndPrint(from, to);
  Test<ADD, NONE, DIAGONAL, POST>::runAndPrint(from, to);
  Test<ADD, NONE, GENERAL, PRE>::runAndPrint(from, to);
  Test<ADD, NONE, GENERAL, POST>::runAndPrint(from, to);
  Test<ADD, MINUS, IDENTITY, PRE>::runAndPrint(from, to);
  Test<ADD, MINUS, IDENTITY, POST>::runAndPrint(from, to);
  Test<ADD, MINUS, DIAGONAL, PRE>::runAndPrint(from, to);
  Test<ADD, MINUS, DIAGONAL, POST>::runAndPrint(from, to);
  Test<ADD, MINUS, GENERAL, PRE>::runAndPrint(from, to);
  Test<ADD, MINUS, GENERAL, POST>::runAndPrint(from, to);
  Test<ADD, SCALAR, IDENTITY, PRE>::runAndPrint(from, to);
  Test<ADD, SCALAR, IDENTITY, POST>::runAndPrint(from, to);
  Test<ADD, SCALAR, DIAGONAL, PRE>::runAndPrint(from, to);
  Test<ADD, SCALAR, DIAGONAL, POST>::runAndPrint(from, to);
  Test<ADD, SCALAR, GENERAL, PRE>::runAndPrint(from, to);
  Test<ADD, SCALAR, GENERAL, POST>::runAndPrint(from, to);

  Test<SUB, NONE, IDENTITY, PRE>::runAndPrint(from, to);
  Test<SUB, NONE, IDENTITY, POST>::runAndPrint(from, to);
  Test<SUB, NONE, DIAGONAL, PRE>::runAndPrint(from, to);
  Test<SUB, NONE, DIAGONAL, POST>::runAndPrint(from, to);
  Test<SUB, NONE, GENERAL, PRE>::runAndPrint(from, to);
  Test<SUB, NONE, GENERAL, POST>::runAndPrint(from, to);
  Test<SUB, MINUS, IDENTITY, PRE>::runAndPrint(from, to);
  Test<SUB, MINUS, IDENTITY, POST>::runAndPrint(from, to);
  Test<SUB, MINUS, DIAGONAL, PRE>::runAndPrint(from, to);
  Test<SUB, MINUS, DIAGONAL, POST>::runAndPrint(from, to);
  Test<SUB, MINUS, GENERAL, PRE>::runAndPrint(from, to);
  Test<SUB, MINUS, GENERAL, POST>::runAndPrint(from, to);
  Test<SUB, SCALAR, IDENTITY, PRE>::runAndPrint(from, to);
  Test<SUB, SCALAR, IDENTITY, POST>::runAndPrint(from, to);
  Test<SUB, SCALAR, DIAGONAL, PRE>::runAndPrint(from, to);
  Test<SUB, SCALAR, DIAGONAL, POST>::runAndPrint(from, to);
  Test<SUB, SCALAR, GENERAL, PRE>::runAndPrint(from, to);
  Test<SUB, SCALAR, GENERAL, POST>::runAndPrint(from, to);

  Test<MIN, NONE, IDENTITY, PRE>::runAndPrint(from, to);
  Test<MIN, NONE, IDENTITY, POST>::runAndPrint(from, to);
  Test<MIN, NONE, DIAGONAL, PRE>::runAndPrint(from, to);
  Test<MIN, NONE, DIAGONAL, POST>::runAndPrint(from, to);
  Test<MIN, NONE, GENERAL, PRE>::runAndPrint(from, to);
  Test<MIN, NONE, GENERAL, POST>::runAndPrint(from, to);
  Test<MIN, MINUS, IDENTITY, PRE>::runAndPrint(from, to);
  Test<MIN, MINUS, IDENTITY, POST>::runAndPrint(from, to);
  Test<MIN, MINUS, DIAGONAL, PRE>::runAndPrint(from, to);
  Test<MIN, MINUS, DIAGONAL, POST>::runAndPrint(from, to);
  Test<MIN, MINUS, GENERAL, PRE>::runAndPrint(from, to);
  Test<MIN, MINUS, GENERAL, POST>::runAndPrint(from, to);
  Test<MIN, SCALAR, IDENTITY, PRE>::runAndPrint(from, to);
  Test<MIN, SCALAR, IDENTITY, POST>::runAndPrint(from, to);
  Test<MIN, SCALAR, DIAGONAL, PRE>::runAndPrint(from, to);
  Test<MIN, SCALAR, DIAGONAL, POST>::runAndPrint(from, to);
  Test<MIN, SCALAR, GENERAL, PRE>::runAndPrint(from, to);
  Test<MIN, SCALAR, GENERAL, POST>::runAndPrint(from, to);

  Test<MAX, NONE, IDENTITY, PRE>::runAndPrint(from, to);
  Test<MAX, NONE, IDENTITY, POST>::runAndPrint(from, to);
  Test<MAX, NONE, DIAGONAL, PRE>::runAndPrint(from, to);
  Test<MAX, NONE, DIAGONAL, POST>::runAndPrint(from, to);
  Test<MAX, NONE, GENERAL, PRE>::runAndPrint(from, to);
  Test<MAX, NONE, GENERAL, POST>::runAndPrint(from, to);
  Test<MAX, MINUS, IDENTITY, PRE>::runAndPrint(from, to);
  Test<MAX, MINUS, IDENTITY, POST>::runAndPrint(from, to);
  Test<MAX, MINUS, DIAGONAL, PRE>::runAndPrint(from, to);
  Test<MAX, MINUS, DIAGONAL, POST>::runAndPrint(from, to);
  Test<MAX, MINUS, GENERAL, PRE>::runAndPrint(from, to);
  Test<MAX, MINUS, GENERAL, POST>::runAndPrint(from, to);
  Test<MAX, SCALAR, IDENTITY, PRE>::runAndPrint(from, to);
  Test<MAX, SCALAR, IDENTITY, POST>::runAndPrint(from, to);
  Test<MAX, SCALAR, DIAGONAL, PRE>::runAndPrint(from, to);
  Test<MAX, SCALAR, DIAGONAL, POST>::runAndPrint(from, to);
  Test<MAX, SCALAR, GENERAL, PRE>::runAndPrint(from, to);
  Test<MAX, SCALAR, GENERAL, POST>::runAndPrint(from, to);

  TestNoFrom<REPLACE>::runAndPrint(to);
  to.setRandom();
  TestNoFrom<ADD>::runAndPrint(to);
  to.setRandom();
  TestNoFrom<SUB>::runAndPrint(to);
  to.setRandom();
  TestNoFrom<MIN>::runAndPrint(to);
  to.setRandom();
  TestNoFrom<MAX>::runAndPrint(to);
}



void testCompiledAssignment()
{
  MatrixXd A = MatrixXd::Ones(5, 5);
  MatrixXd B = MatrixXd::Zero(5, 5);
  testBatch(A, B);

  testBatch(A.block(1, 1, 3, 2), B.topLeftCorner<3, 2>());

  VectorXd a = VectorXd::Ones(5);
  VectorXd b = VectorXd::Zero(5);
  testBatch(a, b);
}

void testCompiledAssignmentWrapper()
{
  typedef CompiledAssignmentWrapper<MatrixXd> MatrixAssignment;
  MatrixXd A1 = MatrixXd::Constant(3, 7, 1);
  MatrixXd A2 = MatrixXd::Constant(2, 7, 2);
  MatrixXd A3 = MatrixXd::Constant(3, 7, 3);
  MatrixXd A4 = MatrixXd::Constant(4, 7, 4);
  MatrixXd B = MatrixXd::Ones(12, 7);
  double s = 2;
  VectorXd w = Vector3d(1, 2, 3);

  std::vector<MatrixAssignment> a;
  a.push_back(MatrixAssignment::make<ADD, NONE, IDENTITY, PRE>(A1, B.middleRows(0, 3)));
  a.push_back(MatrixAssignment::make<REPLACE, MINUS, IDENTITY, PRE>(A2, B.middleRows(3, 2)));
  a.push_back(MatrixAssignment::make<REPLACE, NONE, DIAGONAL, PRE>(A3, B.middleRows(5, 3), 1, &w));
  a.push_back(MatrixAssignment::make<REPLACE, SCALAR, IDENTITY, PRE>(A4, B.middleRows(8, 4), s));

  for (const auto& assignment : a)
    assignment.run();

  std::cout << B << std::endl;
}
