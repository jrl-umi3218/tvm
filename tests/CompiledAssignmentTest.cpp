#include <vector>

// boost
#define BOOST_TEST_MODULE CompiledAssignmentTest
#include <boost/test/unit_test.hpp>

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
  case COPY: return from; break;
  case ADD: return to + from; break;
  case SUB: return to - from; break;
  case MIN: tmp = to.array().min(from.array()); return tmp; break;
  case MAX: tmp = to.array().max(from.array()); return tmp; break;
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

//s*M*F or s*F*M
void assign(AssignType A, ScalarMult S, MatrixMult M, MultPos P,
  double from, Ref<MatrixXd> to, double s = 0, const MatrixXd& wOrM = MatrixXd())
{
  Eigen::DenseIndex r, c;
  if (P == PRE)
  {
    c = to.cols();
    if (M == GENERAL)
      r = wOrM.cols();
    else
      r = to.rows();
  }
  else
  {
    r = to.rows();
    if (M == GENERAL)
      c = wOrM.rows();
    else
      c = to.cols();
  }

  MatrixXd tmp1 = matrixMult(M, P, MatrixXd::Constant(r,c,from), wOrM);
  MatrixXd tmp2 = scalarMult(S, s, tmp1);
  MatrixXd tmp3 = runOperator(A, tmp2, to);
  to = tmp3;
}

void assign(AssignType A, Ref<MatrixXd> to)
{
  if (A == COPY)
    to.setZero();
}

template<AssignType A, ScalarMult S, MatrixMult M, MultPos P, Source F>
struct Test
{
  template<typename Derived, typename U>
  static bool run_check(const MatrixBase<Derived>& from, U& to)
  {
    assert(from.rows() == to.rows() && from.cols() == to.cols());

    typedef MatrixBase<Derived> MatrixType;
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
    CompiledAssignment<Type, A, S, M, P, F> ca(from, to, s, &wOrM);
    ca.run();
    Eigen::internal::set_is_malloc_allowed(true);
    assign(A, S, M, P, f, t, s, wOrM);

    return t.isApprox(to);
  }

  template<typename U>
  static bool run_check(double from, U& to)
  {
    //generate possibly needed multipliers
    double s = 3;
    Eigen::DenseIndex r, c;
    if (M == DIAGONAL)
    {
      c = 1;
      if (P == PRE)
        r = to.rows();
      else
        r = 1;
    }
    else
    {
      if (P == PRE)
      {
        r = to.rows();
        c = to.rows();
      }
      else
      {
        r = 1;
        c = to.cols();
      }
    }
    typename std::conditional<M == DIAGONAL, VectorXd, MatrixXd>::type wOrM(r, c);
    wOrM.setRandom();

    double f = from;
    MatrixXd t = to;

    Eigen::internal::set_is_malloc_allowed(false);
    CompiledAssignment<VectorXd, A, S, M, P, F> ca(from, to, s, &wOrM);
    ca.run();
    Eigen::internal::set_is_malloc_allowed(true);
    assign(A, S, M, P, f, t, s, wOrM);

    return t.isApprox(to);
  }

  template<typename V, typename U>
  static void run(const U& from, V& to, typename std::enable_if<F == EXTERNAL || (V::ColsAtCompileTime == 1 && P == PRE)>::type * = nullptr)
  {
    BOOST_CHECK(run_check(from, to));
  }

  template<typename V, typename U>
  static void run(const U&/*from*/, V& /*to*/, typename std::enable_if<!(F == EXTERNAL || (V::ColsAtCompileTime == 1 && P == PRE))>::type * = nullptr)
  {
  }
};

template<AssignType A>
struct TestNoFrom
{
  template<typename U>
  static void run(U& to)
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
    CompiledAssignment<Type, A, NONE, IDENTITY, POST, ZERO> ca(to);
    ca.run();
    Eigen::internal::set_is_malloc_allowed(true);
    assign(A, t);

    BOOST_CHECK(t.isApprox(to));
  }
};

template<Source F=EXTERNAL, typename U, typename V>
void testBatch(const U & from, V && to)
{
  Test<COPY, NONE, IDENTITY, PRE, F>::run(from, to);
  Test<COPY, NONE, IDENTITY, POST, F>::run(from, to);
  Test<COPY, NONE, DIAGONAL, PRE, F>::run(from, to);
  Test<COPY, NONE, DIAGONAL, POST, F>::run(from, to);
  Test<COPY, NONE, GENERAL, PRE, F>::run(from, to);
  Test<COPY, NONE, GENERAL, POST, F>::run(from, to);
  Test<COPY, MINUS, IDENTITY, PRE, F>::run(from, to);
  Test<COPY, MINUS, IDENTITY, POST, F>::run(from, to);
  Test<COPY, MINUS, DIAGONAL, PRE, F>::run(from, to);
  Test<COPY, MINUS, DIAGONAL, POST, F>::run(from, to);
  Test<COPY, MINUS, GENERAL, PRE, F>::run(from, to);
  Test<COPY, MINUS, GENERAL, POST, F>::run(from, to);
  Test<COPY, SCALAR, IDENTITY, PRE, F>::run(from, to);
  Test<COPY, SCALAR, IDENTITY, POST, F>::run(from, to);
  Test<COPY, SCALAR, DIAGONAL, PRE, F>::run(from, to);
  Test<COPY, SCALAR, DIAGONAL, POST, F>::run(from, to);
  Test<COPY, SCALAR, GENERAL, PRE, F>::run(from, to);
  Test<COPY, SCALAR, GENERAL, POST, F>::run(from, to);

  Test<ADD, NONE, IDENTITY, PRE, F>::run(from, to);
  Test<ADD, NONE, IDENTITY, POST, F>::run(from, to);
  Test<ADD, NONE, DIAGONAL, PRE, F>::run(from, to);
  Test<ADD, NONE, DIAGONAL, POST, F>::run(from, to);
  Test<ADD, NONE, GENERAL, PRE, F>::run(from, to);
  Test<ADD, NONE, GENERAL, POST, F>::run(from, to);
  Test<ADD, MINUS, IDENTITY, PRE, F>::run(from, to);
  Test<ADD, MINUS, IDENTITY, POST, F>::run(from, to);
  Test<ADD, MINUS, DIAGONAL, PRE, F>::run(from, to);
  Test<ADD, MINUS, DIAGONAL, POST, F>::run(from, to);
  Test<ADD, MINUS, GENERAL, PRE, F>::run(from, to);
  Test<ADD, MINUS, GENERAL, POST, F>::run(from, to);
  Test<ADD, SCALAR, IDENTITY, PRE, F>::run(from, to);
  Test<ADD, SCALAR, IDENTITY, POST, F>::run(from, to);
  Test<ADD, SCALAR, DIAGONAL, PRE, F>::run(from, to);
  Test<ADD, SCALAR, DIAGONAL, POST, F>::run(from, to);
  Test<ADD, SCALAR, GENERAL, PRE, F>::run(from, to);
  Test<ADD, SCALAR, GENERAL, POST, F>::run(from, to);

  Test<SUB, NONE, IDENTITY, PRE, F>::run(from, to);
  Test<SUB, NONE, IDENTITY, POST, F>::run(from, to);
  Test<SUB, NONE, DIAGONAL, PRE, F>::run(from, to);
  Test<SUB, NONE, DIAGONAL, POST, F>::run(from, to);
  Test<SUB, NONE, GENERAL, PRE, F>::run(from, to);
  Test<SUB, NONE, GENERAL, POST, F>::run(from, to);
  Test<SUB, MINUS, IDENTITY, PRE, F>::run(from, to);
  Test<SUB, MINUS, IDENTITY, POST, F>::run(from, to);
  Test<SUB, MINUS, DIAGONAL, PRE, F>::run(from, to);
  Test<SUB, MINUS, DIAGONAL, POST, F>::run(from, to);
  Test<SUB, MINUS, GENERAL, PRE, F>::run(from, to);
  Test<SUB, MINUS, GENERAL, POST, F>::run(from, to);
  Test<SUB, SCALAR, IDENTITY, PRE, F>::run(from, to);
  Test<SUB, SCALAR, IDENTITY, POST, F>::run(from, to);
  Test<SUB, SCALAR, DIAGONAL, PRE, F>::run(from, to);
  Test<SUB, SCALAR, DIAGONAL, POST, F>::run(from, to);
  Test<SUB, SCALAR, GENERAL, PRE, F>::run(from, to);
  Test<SUB, SCALAR, GENERAL, POST, F>::run(from, to);

  Test<MIN, NONE, IDENTITY, PRE, F>::run(from, to);
  Test<MIN, NONE, IDENTITY, POST, F>::run(from, to);
  Test<MIN, NONE, DIAGONAL, PRE, F>::run(from, to);
  Test<MIN, NONE, DIAGONAL, POST, F>::run(from, to);
  Test<MIN, NONE, GENERAL, PRE, F>::run(from, to);
  Test<MIN, NONE, GENERAL, POST, F>::run(from, to);
  Test<MIN, MINUS, IDENTITY, PRE, F>::run(from, to);
  Test<MIN, MINUS, IDENTITY, POST, F>::run(from, to);
  Test<MIN, MINUS, DIAGONAL, PRE, F>::run(from, to);
  Test<MIN, MINUS, DIAGONAL, POST, F>::run(from, to);
  Test<MIN, MINUS, GENERAL, PRE, F>::run(from, to);
  Test<MIN, MINUS, GENERAL, POST, F>::run(from, to);
  Test<MIN, SCALAR, IDENTITY, PRE, F>::run(from, to);
  Test<MIN, SCALAR, IDENTITY, POST, F>::run(from, to);
  Test<MIN, SCALAR, DIAGONAL, PRE, F>::run(from, to);
  Test<MIN, SCALAR, DIAGONAL, POST, F>::run(from, to);
  Test<MIN, SCALAR, GENERAL, PRE, F>::run(from, to);
  Test<MIN, SCALAR, GENERAL, POST, F>::run(from, to);

  Test<MAX, NONE, IDENTITY, PRE, F>::run(from, to);
  Test<MAX, NONE, IDENTITY, POST, F>::run(from, to);
  Test<MAX, NONE, DIAGONAL, PRE, F>::run(from, to);
  Test<MAX, NONE, DIAGONAL, POST, F>::run(from, to);
  Test<MAX, NONE, GENERAL, PRE, F>::run(from, to);
  Test<MAX, NONE, GENERAL, POST, F>::run(from, to);
  Test<MAX, MINUS, IDENTITY, PRE, F>::run(from, to);
  Test<MAX, MINUS, IDENTITY, POST, F>::run(from, to);
  Test<MAX, MINUS, DIAGONAL, PRE, F>::run(from, to);
  Test<MAX, MINUS, DIAGONAL, POST, F>::run(from, to);
  Test<MAX, MINUS, GENERAL, PRE, F>::run(from, to);
  Test<MAX, MINUS, GENERAL, POST, F>::run(from, to);
  Test<MAX, SCALAR, IDENTITY, PRE, F>::run(from, to);
  Test<MAX, SCALAR, IDENTITY, POST, F>::run(from, to);
  Test<MAX, SCALAR, DIAGONAL, PRE, F>::run(from, to);
  Test<MAX, SCALAR, DIAGONAL, POST, F>::run(from, to);
  Test<MAX, SCALAR, GENERAL, PRE, F>::run(from, to);
  Test<MAX, SCALAR, GENERAL, POST, F>::run(from, to);

  TestNoFrom<COPY>::run(to);
  to.setRandom();
  TestNoFrom<ADD>::run(to);
  to.setRandom();
  TestNoFrom<SUB>::run(to);
  to.setRandom();
  TestNoFrom<MIN>::run(to);
  to.setRandom();
  TestNoFrom<MAX>::run(to);
}



BOOST_AUTO_TEST_CASE(CompiledAssignmentTest)
{
  MatrixXd A = MatrixXd::Ones(5, 5);
  MatrixXd B = MatrixXd::Zero(5, 5);
  testBatch(A, B);

  testBatch(A.block(1, 1, 3, 2), B.topLeftCorner<3, 2>());

  VectorXd a = VectorXd::Ones(5);
  VectorXd b = VectorXd::Zero(5);
  testBatch(a, b);
  testBatch<CONSTANT>(3, b);
}

BOOST_AUTO_TEST_CASE(CompiledAssignmentWrapperTest)
{
  typedef CompiledAssignmentWrapper<MatrixXd> MatrixAssignment;
  MatrixXd A1 = MatrixXd::Constant(3, 7, 1);
  MatrixXd A2 = MatrixXd::Constant(2, 7, 2);
  MatrixXd A3 = MatrixXd::Constant(3, 7, 3);
  MatrixXd A4 = MatrixXd::Constant(4, 7, 4);
  MatrixXd B = MatrixXd::Ones(12, 7);
  MatrixXd B_ref = B;
  double s = 2;
  VectorXd w = Vector3d(1, 2, 3);

  std::vector<MatrixAssignment> a;
  a.push_back(MatrixAssignment::make<ADD, NONE, IDENTITY, PRE>(A1, B.middleRows(0, 3)));
  a.push_back(MatrixAssignment::make<COPY, MINUS, IDENTITY, PRE>(A2, B.middleRows(3, 2)));
  a.push_back(MatrixAssignment::make<COPY, NONE, DIAGONAL, PRE>(A3, B.middleRows(5, 3), 1, &w));
  a.push_back(MatrixAssignment::make<COPY, SCALAR, IDENTITY, PRE>(A4, B.middleRows(8, 4), s));

  for (const auto& assignment : a)
    assignment.run();

  MatrixXd C(3, 7);
  a[2].from(A1);
  a[2].to(C);
  a[2].run();

  BOOST_CHECK(B.middleRows(0, 3) == A1 + B_ref.middleRows(0,3));
  BOOST_CHECK(B.middleRows(3, 2) == -A2);
  BOOST_CHECK(B.middleRows(5, 3) == w.asDiagonal() * A3);
  BOOST_CHECK(B.middleRows(8, 4) == s * A4);

  BOOST_CHECK(C == w.asDiagonal() * A1);
}
