/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#define EIGEN_RUNTIME_NO_MALLOC
#include <tvm/scheme/internal/CompiledAssignment.h>
#include <tvm/scheme/internal/CompiledAssignmentWrapper.h>

#include <tuple>
#include <vector>

using namespace Eigen;
using namespace tvm::scheme::internal;

MatrixXd runOperator(AssignType A, const MatrixXd & from, const MatrixXd & to)
{
  MatrixXd tmp;
  switch(A)
  {
    case COPY:
      return from;
      break;
    case ADD:
      return to + from;
      break;
    case SUB:
      return to - from;
      break;
    case MIN:
      tmp = to.array().min(from.array());
      return tmp;
      break;
    case MAX:
      tmp = to.array().max(from.array());
      return tmp;
      break;
    default:
      return {};
  }
}

struct Weight
{
  double s;
  VectorXd v;
};

MatrixXd weightMult(WeightMult W, const Weight & w, const MatrixXd & M)
{
  switch(W)
  {
    case NONE:
      return M;
      break;
    case MINUS:
      return -M;
      break;
    case SCALAR:
      return w.s * M;
      break;
    case DIAGONAL:
      return w.v.asDiagonal() * M;
      break;
    default:
      return {};
  }
}

MatrixXd matrixMult(MatrixMult M, const MatrixXd & A, const MatrixXd & Mult)
{
  switch(M)
  {
    case IDENTITY:
      return A;
      break;
      // the test A.cols() is not accurate. We would like to test the number of columns
      // at compile time, but we can't. For these tests, we assume that we don't test
      // matrices with one column
    case GENERAL:
      if(A.cols() == 1)
        return Mult * A;
      else
        return A * Mult;
      break;
    case INVERSE_DIAGONAL:
      if(A.cols() == 1)
        return Mult.diagonal().cwiseInverse().asDiagonal() * A;
      else
        return A * Mult.diagonal().cwiseInverse().asDiagonal();
      break;
    default:
      return {};
  }
}

void assign(AssignType A,
            WeightMult W,
            MatrixMult M,
            const Ref<const MatrixXd> & from,
            Ref<MatrixXd> to,
            const Weight & w,
            const MatrixXd & Mult = MatrixXd())
{
  MatrixXd tmp1 = matrixMult(M, from, Mult);
  MatrixXd tmp2 = weightMult(W, w, tmp1);
  MatrixXd tmp3 = runOperator(A, tmp2, to);
  to = tmp3;
}

// s*M*F or s*F*M
void assign(AssignType A,
            WeightMult W,
            MatrixMult M,
            double from,
            Ref<MatrixXd> to,
            const Weight & w,
            const MatrixXd & Mult = MatrixXd())
{
  Eigen::DenseIndex r, c;
  if(to.cols() == 1)
  {
    c = to.cols();
    if(M == GENERAL)
      r = Mult.cols();
    else
      r = to.rows();
  }
  else
  {
    r = to.rows();
    if(M == GENERAL)
      c = Mult.rows();
    else
      c = to.cols();
  }

  MatrixXd tmp1 = matrixMult(M, MatrixXd::Constant(r, c, from), Mult);
  MatrixXd tmp2 = weightMult(W, w, tmp1);
  MatrixXd tmp3 = runOperator(A, tmp2, to);
  to = tmp3;
}

void assign(AssignType A, Ref<MatrixXd> to)
{
  if(A == COPY)
    to.setZero();
  else if(A == MIN)
    to.array() = to.array().min(0);
  else if(A == MAX)
    to.array() = to.array().max(0);
}

// an example of free function. Here, we reverse the columns of in
void freePostMult(Ref<MatrixXd> out, const Ref<const MatrixXd> & in)
{
  DenseIndex ci = 0;
  for(DenseIndex c = out.cols() - 1; c >= 0; --c)
  {
    out.col(c) = in.col(ci);
    ci = (ci + 1) % in.cols();
  }
}

void freePreMult(Ref<VectorXd> out, const Ref<const VectorXd> & in)
{
  DenseIndex ri = 0;
  for(DenseIndex r = out.rows() - 1; r >= 0; --r)
  {
    out.row(r) = in.row(ri);
    ri = (ri + 1) % in.rows();
  }
}

template<typename T>
using TFun = void (*)(Ref<T>, const Ref<const T> &);

void getFreeFun(TFun<MatrixXd> & f) { f = &freePostMult; }

void getFreeFun(TFun<VectorXd> & f) { f = &freePreMult; }

// detail implementation for the call function below
// adapted from https://stackoverflow.com/a/10766422
namespace detail
{
template<typename MatrixType,
         AssignType A,
         WeightMult W,
         MatrixMult M,
         Source F,
         typename Tuple,
         bool Done,
         int Total,
         size_t... N>
struct call_impl
{
  static CompiledAssignmentWrapper<MatrixType> call(Tuple && t)
  {
    return call_impl < MatrixType, A, W, M, F, Tuple, Total == 1 + sizeof...(N), Total, N...,
           sizeof...(N) > ::call(std::forward<Tuple>(t));
  }
};

template<typename MatrixType, AssignType A, WeightMult W, MatrixMult M, Source F, typename Tuple, int Total, size_t... N>
struct call_impl<MatrixType, A, W, M, F, Tuple, true, Total, N...>
{
  static CompiledAssignmentWrapper<MatrixType> call(Tuple && t)
  {
    return CompiledAssignmentWrapper<MatrixType>::template make<A, W, M, F>(std::get<N>(std::forward<Tuple>(t))...);
  }
};
} // namespace detail

// This function construct a CompiledAssignmentWrapper with a call to
// make(Args&&... args)
// Its main purpose is to unpack the tuple passed as argument into the variadic
// form of args
template<typename MatrixType, AssignType A, WeightMult W, MatrixMult M, Source F, typename Tuple>
CompiledAssignmentWrapper<MatrixType> call(Tuple && t)
{
  typedef typename std::decay<Tuple>::type ttype;
  return detail::call_impl < MatrixType, A, W, M, F, Tuple, 0 == std::tuple_size<ttype>::value,
         std::tuple_size<ttype>::value > ::call(std::forward<Tuple>(t));
}

template<WeightMult W>
struct WArg
{
  static std::tuple<> get(const double &, const Ref<const VectorXd> &) { return {}; }
};
template<>
struct WArg<SCALAR>
{
  static std::tuple<const double &> get(const double & s, const Ref<const VectorXd> &)
  {
    return std::forward_as_tuple(s);
  }
};
template<>
struct WArg<DIAGONAL>
{
  static std::tuple<const Ref<const VectorXd> &> get(const double &, const Ref<const VectorXd> & w)
  {
    return std::forward_as_tuple(w);
  }
};

template<MatrixMult M, typename MatrixType>
struct MArg
{
  static std::tuple<> get(const Ref<const MatrixXd> &, TFun<MatrixType>) { return {}; }
};
template<typename MatrixType>
struct MArg<GENERAL, MatrixType>
{
  static std::tuple<const Ref<const MatrixXd> &> get(const Ref<const MatrixXd> & Mult, TFun<MatrixType>)
  {
    return std::forward_as_tuple(Mult);
  }
};
template<typename MatrixType>
struct MArg<INVERSE_DIAGONAL, MatrixType>
{
  static std::tuple<const Ref<const MatrixXd> &> get(const Ref<const MatrixXd> & Mult, TFun<MatrixType>)
  {
    return std::forward_as_tuple(Mult);
  }
};
template<typename MatrixType>
struct MArg<CUSTOM, MatrixType>
{
  static std::tuple<TFun<MatrixType>> get(const Ref<const MatrixXd> &, TFun<MatrixType> f)
  {
    return std::forward_as_tuple(f);
  }
};

template<Source F, typename MatrixType>
struct SArg
{
  static std::tuple<const Ref<const MatrixType> &> get(const Ref<const MatrixType> & from, double)
  {
    return std::forward_as_tuple(from);
  }
};
template<typename MatrixType>
struct SArg<CONSTANT, MatrixType>
{
  static std::tuple<double> get(const Ref<const MatrixType> &, double constant)
  {
    return std::forward_as_tuple(constant);
  }
};
template<typename MatrixType>
struct SArg<ZERO, MatrixType>
{
  static std::tuple<> get(const Ref<const MatrixType> &, double) { return {}; }
};

/** This function build a CompiledAssignementWrapper by picking the correct arguments
 * to pass to the constructor. The main work is to create a tuple containing
 * exactly the arguments needed by the constructor.
 */
template<typename MatrixType, AssignType A, WeightMult W, MatrixMult M, Source F>
CompiledAssignmentWrapper<MatrixType> build(Ref<MatrixType> to,
                                            const double & s,
                                            const Ref<const VectorXd> & w,
                                            const Ref<const MatrixXd> & Mult,
                                            TFun<MatrixType> f,
                                            const Ref<const MatrixType> & from,
                                            double constant)
{
  auto args = std::tuple_cat(std::make_tuple(to), SArg<F, MatrixType>::get(from, constant), WArg<W>::get(s, w),
                             MArg<M, MatrixType>::get(Mult, f));
  return call<MatrixType, A, W, M, F>(args);
}

template<AssignType A, WeightMult W, MatrixMult M, Source F>
struct Test
{
  template<typename Derived, typename U>
  static bool run_check(const MatrixBase<Derived> & from, U & to)
  {
    typedef MatrixBase<Derived> MatrixType;
    enum
    {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };
    typedef typename MatrixType::Scalar Scalar;
    typedef Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, ColMajor, MaxRowsAtCompileTime, MaxColsAtCompileTime>
        Type;

    // generate possibly needed multipliers
    double s = 3;
    double cst = 1;
    Eigen::VectorXd v;
    MatrixXd Mult;
    Weight w;
    TFun<Type> custom;
    if(ColsAtCompileTime == 1)
    {
      assert(to.cols() == from.cols());
      Mult.resize(to.rows(), from.rows());
    }
    else
    {
      assert(to.rows() == from.rows());
      Mult.resize(from.cols(), to.cols());
    }
    v.resize(to.rows());
    v.setRandom();
    Mult.setRandom();
    getFreeFun(custom);

    if(W == DIAGONAL)
      w.v = v;
    else
      w.s = s;

    MatrixXd f = from;
    MatrixXd t = to;

    tvm::utils::override_is_malloc_allowed(false);
    auto ca = build<Type, A, W, M, F>(to, s, v, Mult, custom, from, cst);
    ca.run();
    tvm::utils::restore_is_malloc_allowed();
    assign(A, W, M, f, t, w, Mult);

    return t.isApprox(to);
  }

  template<typename U>
  static bool run_check(double from, U & to)
  {
    // generate possibly needed multipliers
    double s = 3;
    double cst = from;
    VectorXd From(1);
    VectorXd v = VectorXd::Random(to.rows());
    MatrixXd Mult = MatrixXd::Random(to.rows(), to.rows());
    Weight w;
    TFun<VectorXd> custom;
    getFreeFun(custom);

    if(W == DIAGONAL)
      w.v = v;
    else
      w.s = s;

    double f = from;
    VectorXd t = to;

    VectorXd to2 = to;

    Ref<VectorXd> r1{to};
    Ref<const VectorXd> r2{v};
    Ref<const MatrixXd> r3{Mult};
    Ref<const VectorXd> r4{From};

    tvm::utils::override_is_malloc_allowed(false);
    auto ca = build<VectorXd, A, W, M, F>(to2, s, v, Mult, custom, From, cst);
    ca.run();
    tvm::utils::restore_is_malloc_allowed();
    assign(A, W, M, f, t, w, Mult);

    return t.isApprox(to2);
  }

  template<typename V, typename U>
  static void run(const U & from,
                  V & to,
                  typename std::enable_if<F == EXTERNAL || (V::ColsAtCompileTime == 1)>::type * = nullptr)
  {
    FAST_CHECK_UNARY(run_check(from, to));
  }
};

template<AssignType A>
struct TestNoFrom
{
  template<typename U>
  static void run(U & to)
  {
    typedef U MatrixType;
    enum
    {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };
    typedef typename MatrixType::Scalar Scalar;
    typedef Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, ColMajor, MaxRowsAtCompileTime, MaxColsAtCompileTime>
        Type;

    MatrixXd t = to;

    tvm::utils::override_is_malloc_allowed(false);
    auto ca = CompiledAssignmentWrapper<Type>::template make<A, NONE, IDENTITY, ZERO>(to);
    ca.run();
    tvm::utils::restore_is_malloc_allowed();
    assign(A, t);

    FAST_CHECK_UNARY(t.isApprox(to));
  }
};

template<Source F = EXTERNAL, typename U, typename V>
void testBatch(const U & from, V && to)
{
  Test<COPY, NONE, IDENTITY, F>::run(from, to);
  Test<ADD, NONE, IDENTITY, F>::run(from, to);
  Test<SUB, NONE, IDENTITY, F>::run(from, to);
  Test<MIN, NONE, IDENTITY, F>::run(from, to);
  Test<MAX, NONE, IDENTITY, F>::run(from, to);
  Test<COPY, MINUS, IDENTITY, F>::run(from, to);
  Test<SUB, MINUS, IDENTITY, F>::run(from, to);
  Test<ADD, MINUS, IDENTITY, F>::run(from, to);
  Test<MIN, MINUS, IDENTITY, F>::run(from, to);
  Test<MAX, MINUS, IDENTITY, F>::run(from, to);
  Test<COPY, SCALAR, IDENTITY, F>::run(from, to);
  Test<ADD, SCALAR, IDENTITY, F>::run(from, to);
  Test<SUB, SCALAR, IDENTITY, F>::run(from, to);
  Test<MIN, SCALAR, IDENTITY, F>::run(from, to);
  Test<MAX, SCALAR, IDENTITY, F>::run(from, to);
  Test<COPY, DIAGONAL, IDENTITY, F>::run(from, to);
  Test<ADD, DIAGONAL, IDENTITY, F>::run(from, to);
  Test<SUB, DIAGONAL, IDENTITY, F>::run(from, to);
  Test<MIN, DIAGONAL, IDENTITY, F>::run(from, to);
  Test<MAX, DIAGONAL, IDENTITY, F>::run(from, to);
  Test<COPY, NONE, GENERAL, F>::run(from, to);
  Test<ADD, NONE, GENERAL, F>::run(from, to);
  Test<SUB, NONE, GENERAL, F>::run(from, to);
  Test<MIN, NONE, GENERAL, F>::run(from, to);
  Test<MAX, NONE, GENERAL, F>::run(from, to);
  Test<COPY, MINUS, GENERAL, F>::run(from, to);
  Test<SUB, MINUS, GENERAL, F>::run(from, to);
  Test<ADD, MINUS, GENERAL, F>::run(from, to);
  Test<MIN, MINUS, GENERAL, F>::run(from, to);
  Test<MAX, MINUS, GENERAL, F>::run(from, to);
  Test<COPY, SCALAR, GENERAL, F>::run(from, to);
  Test<ADD, SCALAR, GENERAL, F>::run(from, to);
  Test<SUB, SCALAR, GENERAL, F>::run(from, to);
  Test<MIN, SCALAR, GENERAL, F>::run(from, to);
  Test<MAX, SCALAR, GENERAL, F>::run(from, to);
  Test<COPY, DIAGONAL, GENERAL, F>::run(from, to);
  Test<ADD, DIAGONAL, GENERAL, F>::run(from, to);
  Test<SUB, DIAGONAL, GENERAL, F>::run(from, to);
  Test<MIN, DIAGONAL, GENERAL, F>::run(from, to);
  Test<MAX, DIAGONAL, GENERAL, F>::run(from, to);
  Test<COPY, NONE, INVERSE_DIAGONAL, F>::run(from, to);
  Test<ADD, NONE, INVERSE_DIAGONAL, F>::run(from, to);
  Test<SUB, NONE, INVERSE_DIAGONAL, F>::run(from, to);
  Test<MIN, NONE, INVERSE_DIAGONAL, F>::run(from, to);
  Test<MAX, NONE, INVERSE_DIAGONAL, F>::run(from, to);
  Test<COPY, MINUS, INVERSE_DIAGONAL, F>::run(from, to);
  Test<SUB, MINUS, INVERSE_DIAGONAL, F>::run(from, to);
  Test<ADD, MINUS, INVERSE_DIAGONAL, F>::run(from, to);
  Test<MIN, MINUS, INVERSE_DIAGONAL, F>::run(from, to);
  Test<MAX, MINUS, INVERSE_DIAGONAL, F>::run(from, to);
  Test<COPY, SCALAR, INVERSE_DIAGONAL, F>::run(from, to);
  Test<ADD, SCALAR, INVERSE_DIAGONAL, F>::run(from, to);
  Test<SUB, SCALAR, INVERSE_DIAGONAL, F>::run(from, to);
  Test<MIN, SCALAR, INVERSE_DIAGONAL, F>::run(from, to);
  Test<MAX, SCALAR, INVERSE_DIAGONAL, F>::run(from, to);
  Test<COPY, DIAGONAL, INVERSE_DIAGONAL, F>::run(from, to);
  Test<ADD, DIAGONAL, INVERSE_DIAGONAL, F>::run(from, to);
  Test<SUB, DIAGONAL, INVERSE_DIAGONAL, F>::run(from, to);
  Test<MIN, DIAGONAL, INVERSE_DIAGONAL, F>::run(from, to);
  Test<MAX, DIAGONAL, INVERSE_DIAGONAL, F>::run(from, to);
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

TEST_CASE("Test compiled assignments")
{
  MatrixXd A = MatrixXd::Ones(5, 5);
  MatrixXd B = MatrixXd::Zero(5, 5);

  testBatch(A, B);

  testBatch(A.block(1, 1, 3, 3), B.topLeftCorner<3, 3>());

  VectorXd a = VectorXd::Ones(5);
  VectorXd b = VectorXd::Zero(5);
  testBatch(a, b);
  testBatch<CONSTANT>(3, b);
}

TEST_CASE("Test compiled assignments wrapper")
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
  a.push_back(MatrixAssignment::make<ADD, NONE, IDENTITY, EXTERNAL>(B.middleRows(0, 3), A1));
  a.push_back(MatrixAssignment::make<COPY, MINUS, IDENTITY, EXTERNAL>(B.middleRows(3, 2), A2));
  a.push_back(MatrixAssignment::make<COPY, DIAGONAL, IDENTITY, EXTERNAL>(B.middleRows(5, 3), A3, w));
  a.push_back(MatrixAssignment::make<COPY, SCALAR, IDENTITY, EXTERNAL>(B.middleRows(8, 4), A4, s));

  for(/*const*/ auto & assignment : a)
    assignment.run();

  MatrixXd C(3, 7);
  a[2].from(A1);
  a[2].to(C);
  a[2].run();

  FAST_CHECK_EQ(B.middleRows(0, 3), A1 + B_ref.middleRows(0, 3));
  FAST_CHECK_EQ(B.middleRows(3, 2), -A2);
  FAST_CHECK_EQ(B.middleRows(5, 3), w.asDiagonal() * A3);
  FAST_CHECK_EQ(B.middleRows(8, 4), s * A4);

  FAST_CHECK_EQ(C, w.asDiagonal() * A1);

  CHECK_THROWS(a[2].from(3));

  // default constructor and copy
  MatrixAssignment ma;
  ma = MatrixAssignment::make<ADD, NONE, IDENTITY, EXTERNAL>(B.middleRows(0, 3), A1);
  ma.run();
  FAST_CHECK_EQ(B.middleRows(0, 3), B_ref.middleRows(0, 3) + A1 + A1);
}
