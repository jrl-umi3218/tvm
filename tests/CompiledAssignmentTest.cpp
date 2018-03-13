#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#define AUTHORIZE_MALLOC_FOR_CACHE
#include <tvm/scheme/internal/CompiledAssignment.h>
//#include <tvm/scheme/internal/CompiledAssignmentWrapper.h>

#include <tuple>
#include <vector>

using namespace Eigen;
using namespace tvm::scheme::internal;

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

struct Weight
{
  double s;
  VectorXd v;
};

MatrixXd weightMult(WeightMult W, const Weight& w, const MatrixXd& M)
{
  switch (W)
  {
  case NONE: return M; break;
  case MINUS: return -M; break;
  case SCALAR: return w.s*M; break;
  case DIAGONAL: return w.v.asDiagonal() * M; break;
  case INVERSE_DIAGONAL: return w.v.cwiseInverse().asDiagonal() * M; break;
  default: return{};
  }
}

MatrixXd matrixMult(MatrixMult M, const MatrixXd& A, const MatrixXd& Mult)
{
  switch (M)
  {
  case IDENTITY: return A; break;
    //the test A.cols() is not accurate. We would like to test the number of columns
    //at compile time, but we can't. For these tests, we assume that we don't test
    //matrices with one column
  case GENERAL: if (A.cols()==1) return Mult*A; else return A*Mult; break;
  default: return{};
  }
}


void assign(AssignType A, WeightMult W, MatrixMult M,
  const Ref<const MatrixXd>& from, Ref<MatrixXd> to, 
  const Weight& w, const MatrixXd& Mult = MatrixXd())
{
  MatrixXd tmp1 = matrixMult(M, from, Mult);
  MatrixXd tmp2 = weightMult(W, w, tmp1);
  MatrixXd tmp3 = runOperator(A, tmp2, to);
  to = tmp3;
}

//s*M*F or s*F*M
void assign(AssignType A, WeightMult W, MatrixMult M,
  double from, Ref<MatrixXd> to, const Weight& w, const MatrixXd& Mult = MatrixXd())
{
  Eigen::DenseIndex r, c;
  if (to.cols() == 1)
  {
    c = to.cols();
    if (M == GENERAL)
      r = Mult.cols();
    else
      r = to.rows();
  }
  else
  {
    r = to.rows();
    if (M == GENERAL)
      c = Mult.rows();
    else
      c = to.cols();
  }

  MatrixXd tmp1 = matrixMult(M, MatrixXd::Constant(r,c,from), Mult);
  MatrixXd tmp2 = weightMult(W, w, tmp1);
  MatrixXd tmp3 = runOperator(A, tmp2, to);
  to = tmp3;
}

void assign(AssignType A, Ref<MatrixXd> to)
{
  if (A == COPY)
    to.setZero();
}

// an exemple of free fonction. Here, we reverse the columns of in
void freePostMult(Ref<MatrixXd> out, const Ref<const MatrixXd>& in)
{
  DenseIndex ci = 0;
  for (DenseIndex c = out.cols() - 1; c >= 0; --c)
  {
    out.col(c) = in.col(ci);
    ci = (ci + 1) % in.cols();
  }
}

void freePreMult(Ref<VectorXd> out, const Ref<const VectorXd>& in)
{
  DenseIndex ri = 0;
  for (DenseIndex r = out.rows() - 1; r >= 0; --r)
  {
    out.row(r) = in.row(ri);
    ri = (ri + 1) % in.rows();
  }
}

template<typename T>
using TFun = void(*)(Ref<T>, const Ref<const T>&);

void getFreeFun(TFun<MatrixXd>& f)
{
  f = &freePostMult;
}

void getFreeFun(TFun<VectorXd>& f)
{
  f = &freePreMult;
}

//detail implentation for the call function under
namespace detail
{
  template <typename F, typename Tuple, bool Done, int Total, int... N>
  struct call_impl
  {
    static F call(Tuple && t)
    {
      return call_impl<F, Tuple, Total == 1 + sizeof...(N), Total, N..., sizeof...(N)>::call(std::forward<Tuple>(t));
    }
  };

  template <typename F, typename Tuple, int Total, int... N>
  struct call_impl<F, Tuple, true, Total, N...>
  {
    static F call(Tuple && t)
    {
      return F(std::get<N>(std::forward<Tuple>(t))...);
    }
  };
}

// This function construct an object of type F with a constructor F(Args&&... args)
// Its main purpose is to unpack the tuple passed as argument into the variadic
// form of args
template <typename F, typename Tuple>
F call(Tuple && t)
{
  typedef typename std::decay<Tuple>::type ttype;
  return detail::call_impl<F, Tuple, 0 == std::tuple_size<ttype>::value, std::tuple_size<ttype>::value>::call(std::forward<Tuple>(t));
}

template<WeightMult W> struct WArg { static std::tuple<> get(double s, const Ref<const VectorXd>& w) { return {}; } };
template<> struct WArg<SCALAR> { static std::tuple<double> get(double s, const Ref<const VectorXd>& w) { return s; } };
template<> struct WArg<DIAGONAL> { static std::tuple<const Ref<const VectorXd>&> get(double s, const Ref<const VectorXd>& w) { return w; } };
template<> struct WArg<INVERSE_DIAGONAL> { static std::tuple<const Ref<const VectorXd>&> get(double s, const Ref<const VectorXd>& w) { return w; } };

template<MatrixMult M, typename MatrixType>
struct MArg { static std::tuple<> get(const Ref<const MatrixXd>& Mult, TFun<MatrixType> f) { return {}; } };
template<typename MatrixType>
struct MArg<GENERAL, MatrixType> { static std::tuple<const Ref<const MatrixXd>&> get(const Ref<const MatrixXd>& Mult, TFun<MatrixType> f) { return Mult; } };
template<typename MatrixType>
struct MArg<CUSTOM, MatrixType> { static std::tuple<TFun<MatrixType>> get(const Ref<const MatrixXd>& Mult, TFun<MatrixType> f) { return f; } };

template<Source F, typename MatrixType>
struct SArg { static std::tuple<const Ref<const MatrixType>&> get(const Ref<const MatrixType>& from, double constant) { return from; } };
template<typename MatrixType>
struct SArg<CONSTANT, MatrixType> { static std::tuple<double> get(const Ref<const MatrixType>& from, double constant) { return constant; } };
template<typename MatrixType>
struct SArg<ZERO, MatrixType> { static std::tuple<> get(const Ref<const MatrixType>& from, double constant) { return {}; } };

/** This function build a CompiledAssignement by picking the correct arguments
  * to pass to the constructor. The main work is to create a tuple containing
  * exactly the arguments needed byt the constructor.
  */
template<typename MatrixType, AssignType A, WeightMult W, MatrixMult M, Source F>
CompiledAssignment<MatrixType, A, W, M, F> build(Ref<MatrixType> to, double s, const Ref<const VectorXd>& w,
  const Ref<const MatrixXd>& Mult, TFun<MatrixType> f,
  const Ref<const MatrixType>& from, double constant)
{
  auto args = std::tuple_cat(std::make_tuple(to), SArg<F, MatrixType>::get(from, constant), WArg<W>::get(s, w), MArg<M, MatrixType>::get(Mult, f));
  using ReturnType = CompiledAssignment<MatrixType, A, W, M, F>;
  return call<ReturnType>(args);
}

template<AssignType A, WeightMult W, MatrixMult M, Source F>
struct Test
{
  template<typename Derived, typename U>
  static void run_check(const MatrixBase<Derived>& from, U& to)
  {
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
    double cst = 1;
    Eigen::VectorXd v;
    MatrixXd Mult;
    Weight w;
    TFun<Type> custom;
    if (ColsAtCompileTime == 1)
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

    if (W == DIAGONAL || W == INVERSE_DIAGONAL)
      w.v = v;
    else
      w.s = s;

    MatrixXd f = from;
    MatrixXd t = to;

    Eigen::internal::set_is_malloc_allowed(false);
    auto ca = build<Type, A, W, M, F>(to, s, v, Mult, custom, from, cst);
    //auto ca = CompiledAssignmentWrapper<Type>::template make<A, S, M, P>(from, to, s, &wOrM);
    ca.run();
    Eigen::internal::set_is_malloc_allowed(true);
    assign(A, W, M, f, t, w, Mult);

    FAST_CHECK_UNARY(t.isApprox(to));
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
    auto ca = CompiledAssignmentWrapper<VectorXd>::make<A, S, M, P>(from, to, s, &wOrM);
    ca.run();
    Eigen::internal::set_is_malloc_allowed(true);
    assign(A, S, M, P, f, t, s, wOrM);

    return t.isApprox(to);
  }

  //template<typename V, typename U>
  //static void run(const U& from, V& to, typename std::enable_if<F == EXTERNAL || (V::ColsAtCompileTime == 1 && P == PRE)>::type * = nullptr)
  //{
  //  FAST_CHECK_UNARY(run_check(from, to));
  //}

  //template<typename V, typename U>
  //static void run(const U&/*from*/, V& /*to*/, typename std::enable_if<!(F == EXTERNAL || (V::ColsAtCompileTime == 1 && P == PRE))>::type * = nullptr)
  //{
  //}
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
    auto ca = CompiledAssignmentWrapper<Type>::template make<A>(to);
    ca.run();
    Eigen::internal::set_is_malloc_allowed(true);
    assign(A, t);

    FAST_CHECK_UNARY(t.isApprox(to));
  }
};

template<Source F=EXTERNAL, typename U, typename V>
void testBatch(const U & from, V && to)
{
  Test<COPY, NONE,             IDENTITY, F>::run_check(from, to);
  Test<ADD,  NONE,             IDENTITY, F>::run_check(from, to);
  Test<SUB,  NONE,             IDENTITY, F>::run_check(from, to);
  Test<MIN,  NONE,             IDENTITY, F>::run_check(from, to);
  Test<MAX,  NONE,             IDENTITY, F>::run_check(from, to);
  Test<COPY, MINUS,            IDENTITY, F>::run_check(from, to);
  Test<SUB,  MINUS,            IDENTITY, F>::run_check(from, to);
  Test<ADD,  MINUS,            IDENTITY, F>::run_check(from, to);
  Test<MIN,  MINUS,            IDENTITY, F>::run_check(from, to);
  Test<MAX,  MINUS,            IDENTITY, F>::run_check(from, to);
  Test<COPY, SCALAR,           IDENTITY, F>::run_check(from, to);
  Test<ADD,  SCALAR,           IDENTITY, F>::run_check(from, to);
  Test<SUB,  SCALAR,           IDENTITY, F>::run_check(from, to);
  Test<MIN,  SCALAR,           IDENTITY, F>::run_check(from, to);
  Test<MAX,  SCALAR,           IDENTITY, F>::run_check(from, to);
  Test<COPY, DIAGONAL,         IDENTITY, F>::run_check(from, to);
  Test<ADD,  DIAGONAL,         IDENTITY, F>::run_check(from, to);
  Test<SUB,  DIAGONAL,         IDENTITY, F>::run_check(from, to);
  Test<MIN,  DIAGONAL,         IDENTITY, F>::run_check(from, to);
  Test<MAX,  DIAGONAL,         IDENTITY, F>::run_check(from, to);
  Test<COPY, INVERSE_DIAGONAL, IDENTITY, F>::run_check(from, to);
  Test<ADD,  INVERSE_DIAGONAL, IDENTITY, F>::run_check(from, to);
  Test<SUB,  INVERSE_DIAGONAL, IDENTITY, F>::run_check(from, to);
  Test<MIN,  INVERSE_DIAGONAL, IDENTITY, F>::run_check(from, to);
  Test<MAX,  INVERSE_DIAGONAL, IDENTITY, F>::run_check(from, to);
  Test<COPY, NONE,             GENERAL,  F>::run_check(from, to);
  Test<ADD,  NONE,             GENERAL,  F>::run_check(from, to);
  Test<SUB,  NONE,             GENERAL,  F>::run_check(from, to);
  Test<MIN,  NONE,             GENERAL,  F>::run_check(from, to);
  Test<MAX,  NONE,             GENERAL,  F>::run_check(from, to);
  Test<COPY, MINUS,            GENERAL,  F>::run_check(from, to);
  Test<SUB,  MINUS,            GENERAL,  F>::run_check(from, to);
  Test<ADD,  MINUS,            GENERAL,  F>::run_check(from, to);
  Test<MIN,  MINUS,            GENERAL,  F>::run_check(from, to);
  Test<MAX,  MINUS,            GENERAL,  F>::run_check(from, to);
  Test<COPY, SCALAR,           GENERAL,  F>::run_check(from, to);
  Test<ADD,  SCALAR,           GENERAL,  F>::run_check(from, to);
  Test<SUB,  SCALAR,           GENERAL,  F>::run_check(from, to);
  Test<MIN,  SCALAR,           GENERAL,  F>::run_check(from, to);
  Test<MAX,  SCALAR,           GENERAL,  F>::run_check(from, to);
  Test<COPY, DIAGONAL,         GENERAL,  F>::run_check(from, to);
  Test<ADD,  DIAGONAL,         GENERAL,  F>::run_check(from, to);
  Test<SUB,  DIAGONAL,         GENERAL,  F>::run_check(from, to);
  Test<MIN,  DIAGONAL,         GENERAL,  F>::run_check(from, to);
  Test<MAX,  DIAGONAL,         GENERAL,  F>::run_check(from, to);
  Test<COPY, INVERSE_DIAGONAL, GENERAL,  F>::run_check(from, to);
  Test<ADD,  INVERSE_DIAGONAL, GENERAL,  F>::run_check(from, to);
  Test<SUB,  INVERSE_DIAGONAL, GENERAL,  F>::run_check(from, to);
  Test<MIN,  INVERSE_DIAGONAL, GENERAL,  F>::run_check(from, to);
  Test<MAX,  INVERSE_DIAGONAL, GENERAL,  F>::run_check(from, to);
//  TestNoFrom<COPY>::run(to);
//  to.setRandom();
//  TestNoFrom<ADD>::run(to);
//  to.setRandom();
//  TestNoFrom<SUB>::run(to);
//  to.setRandom();
//  TestNoFrom<MIN>::run(to);
//  to.setRandom();
//  TestNoFrom<MAX>::run(to);
}



TEST_CASE("Test compiled assignments")
{
  MatrixXd A = MatrixXd::Ones(5, 5);
  MatrixXd B = MatrixXd::Zero(5, 5);
  testBatch(A, B);

  //testBatch(A.block(1, 1, 3, 2), B.topLeftCorner<3, 2>());

  //VectorXd a = VectorXd::Ones(5);
  //VectorXd b = VectorXd::Zero(5);
  //testBatch(a, b);
  //testBatch<CONSTANT>(3, b);
}

//TEST_CASE("Test compiled assignments wrapper")
//{
//  typedef CompiledAssignmentWrapper<MatrixXd> MatrixAssignment;
//  MatrixXd A1 = MatrixXd::Constant(3, 7, 1);
//  MatrixXd A2 = MatrixXd::Constant(2, 7, 2);
//  MatrixXd A3 = MatrixXd::Constant(3, 7, 3);
//  MatrixXd A4 = MatrixXd::Constant(4, 7, 4);
//  MatrixXd B = MatrixXd::Ones(12, 7);
//  MatrixXd B_ref = B;
//  double s = 2;
//  VectorXd w = Vector3d(1, 2, 3);
//
//  std::vector<MatrixAssignment> a;
//  a.push_back(MatrixAssignment::make<ADD, NONE, IDENTITY, PRE>(A1, B.middleRows(0, 3)));
//  a.push_back(MatrixAssignment::make<COPY, MINUS, IDENTITY, PRE>(A2, B.middleRows(3, 2)));
//  a.push_back(MatrixAssignment::make<COPY, NONE, DIAGONAL, PRE>(A3, B.middleRows(5, 3), 1, &w));
//  a.push_back(MatrixAssignment::make<COPY, SCALAR, IDENTITY, PRE>(A4, B.middleRows(8, 4), s));
//
//  for (const auto& assignment : a)
//    assignment.run();
//
//  MatrixXd C(3, 7);
//  a[2].from(A1);
//  a[2].to(C);
//  a[2].run();
//
//  FAST_CHECK_EQ(B.middleRows(0, 3), A1 + B_ref.middleRows(0,3));
//  FAST_CHECK_EQ(B.middleRows(3, 2), -A2);
//  FAST_CHECK_EQ(B.middleRows(5, 3), w.asDiagonal() * A3);
//  FAST_CHECK_EQ(B.middleRows(8, 4), s * A4);
//
//  FAST_CHECK_EQ(C, w.asDiagonal() * A1);
//}
