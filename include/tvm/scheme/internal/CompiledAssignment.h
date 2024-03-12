/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/utils/memoryChecks.h>

#include <Eigen/Core>

#include <type_traits>

namespace tvm
{

namespace scheme
{

namespace internal
{

/** Specify in which way \a from is assigned to \a to.*/
enum AssignType
{
  /** to =  from */ COPY,
  /** to += from */ ADD,
  /** to -= from */ SUB,
  /** to = min(to, from) */ MIN,
  /** to = max(to, from) */ MAX
};

/** Specify whether \a from is to be multiplied by 1, -1 or a user
 * specified scalar.
 */
enum WeightMult
{
  /** from */ NONE,
  /** -from */ MINUS,
  /** s*from */ SCALAR,
  /** diag(d) * from */ DIAGONAL
};

/** Specify if from is to be multiplied a matrix, and if so, what is the
 * type of the matrix.
 */
enum MatrixMult
{
  /** from */
  IDENTITY,
  /** M*from (vector case) or from*M (matrix case) where M is a matrix.*/
  GENERAL,
  /** inv(M) * from or from*inv(M) where M is diagonal*/
  INVERSE_DIAGONAL,
  /** use a custom multiplier */
  CUSTOM
};

/** The type of the source.
 * Note there are two considerations for explicitly introducing the
 * CONSTANT case (versus requiring the user to give a Ref to a constant
 * vector):
 *  - we can deal with it more efficiently
 *  - we take care automatically of any change of size of to or of the
 *    multiplier matrix.
 */
enum Source
{
  /** source is an external vector or matrix (main use-case) */
  EXTERNAL,
  /** source is zero */
  ZERO,
  /** source is a (non-zero) constant */
  CONSTANT
};

/** trait-like definition to detect if an Eigen expression \p MatrixType is describing
 * a vector.
 */
template<typename MatrixType>
using isVector = typename std::conditional<MatrixType::ColsAtCompileTime == 1, std::true_type, std::false_type>::type;

/* trait-like definition to detect if an Eigen expression \p MatrixType is describing
 * a matrix.
 */
template<typename MatrixType>
using isMatrix = typename std::conditional<MatrixType::ColsAtCompileTime != 1, std::true_type, std::false_type>::type;

/** Dummy type for constructors with no arguments*/
class NoArg
{};

/** Helper structure to get the N-th argument in a function call.
 * \sa ParseArg
 */
template<int N>
class ParseArg_
{
public:
  template<typename... Args>
  static typename std::tuple_element<N, std::tuple<Args...>>::type get(Args &&... args)
  {
    return std::get<N>(std::forward_as_tuple(args...));
  }
};

/** Helper structure to get no argument in a function call.
 * \sa ParseArg
 */
class ParseNoArg_
{
public:
  template<typename... Args>
  static NoArg get(Args &&...)
  {
    return {};
  }
};

/** A helper structure with a static \p get method to retrieve the N-th
 * argument in a function call. We use it to dispatch arguments to base
 * classes in the constructor of an aggregated derived class.
 * If N>=0, returns the corresponding argument, otherwise returns an instance
 * of NoArg.
 */
template<int N>
class ParseArg : public std::conditional<(N >= 0), ParseArg_<N>, ParseNoArg_>::type
{};

/** A dummy helper function to build hasNoArgCtor*/
template<typename T>
std::true_type hasNoArgCtorDummy(const T &);
/** Helper function that exists only if \p T has a constructor accepting NoArg.
 * We use hasNoArgCtorDummy as a mean to enable SFINAE.
 */
template<typename T>
decltype(hasNoArgCtorDummy(T(NoArg()))) hasNoArgCtor_(int);
/** Overload that always exists. It will only be chosen if the other overload
 * does not exist.
 */
template<typename>
std::false_type hasNoArgCtor_(...);

/** Traits-like class to detect if class \p T has a constructor accepting an
 * instance of \p NoArg as argument.
 * The class derives from std::true_type if this is the case and
 * std::false_type otherwise.
 */
template<typename T>
class hasNoArgCtor : public decltype(hasNoArgCtor_<T>(0))
{};

/** Given a list of types, count how many of then have a constructor
 * accepting NoArg
 */
template<typename T, typename... Args>
class ArgCount
{
public:
  static constexpr int count = ArgCount<Args...>::count + (hasNoArgCtor<T>::value ? 0 : 1);
};

/** End of recursion for ArgCount*/
template<typename T>
class ArgCount<T>
{
public:
  static constexpr int count = hasNoArgCtor<T>::value ? 0 : 1;
};

/** A cache for holding temporary value in the evaluation of an assignment.
 * By default, it does not do anything but forward an expression.
 */
template<typename MatrixType, bool Cache>
class CachedResult
{
public:
  CachedResult(const Eigen::Ref<MatrixType> &) {}

  template<typename T>
  const T & cache(const T & M)
  {
    return M;
  }
};

/** Specialization for when a cache is needed.*/
template<typename MatrixType>
class CachedResult<MatrixType, true>
{
public:
  CachedResult(const Eigen::Ref<MatrixType> & to) { TVM_TEMPORARY_ALLOW_EIGEN_MALLOC(cache_.resizeLike(to)); }

  template<typename T>
  const MatrixType & cache(const T & M)
  {
    using ConstType = decltype(std::declval<Eigen::Ref<const Eigen::MatrixXd>>() * Eigen::VectorXd::Constant(1, 1));
    if constexpr(std::is_same_v<const T, ConstType>)
    {
      // For the case where M = matrix * Constant, Eigen create a temporary to store  Vector:Constant before evaluating
      // the product. We avoid that here, by doing the product by hand. This could be further optimized by taking into
      // account the WeightMult at once, and possibly without using a cache.
      // The product A * v where v = c * 1 with c a scalar and 1 the vector of ones is equal to c * A * 1. We have that
      // A * 1 is the sum of column of A, what we leverage in the following computations:
      cache_ = M.lhs().rowwise().sum(); // TODO compare to a handmade loop summing the columns
      cache_ *= M.rhs().functor().m_other;
    }
    else
    {
      cache_.noalias() = M;
    }
    return cache_;
  }

  MatrixType & cache() { return cache_; }
  const MatrixType & cache() const { return cache_; }

private:
  MatrixType cache_;
};

/** Traits for deciding whether or not to use a cache before the assign step.
 * By default, no cache is used.
 */
template<typename MatrixType, AssignType A, WeightMult W, MatrixMult M, Source F>
class use_assign_cache : public std::false_type
{};

/** Specialization for min/max with general matrix product. In this case, we
 * use the cache
 */
template<typename MatrixType, WeightMult W>
class use_assign_cache<MatrixType, MIN, W, GENERAL, EXTERNAL> : public std::true_type
{};
template<typename MatrixType, WeightMult W>
class use_assign_cache<MatrixType, MAX, W, GENERAL, EXTERNAL> : public std::true_type
{};

/** Specialization for GENERAL*CONSTANT. This should not be necessary, but
 * the product needs a temporary. Maybe it's not the case anymore with Eigen 3.3.
 */
template<typename MatrixType, AssignType A, WeightMult W>
class use_assign_cache<MatrixType, A, W, GENERAL, CONSTANT> : public std::true_type
{};

/** Traits for deciding whether or not to use a cache for the product.
 * By default, no cache is used.
 */
template<typename MatrixType, AssignType A, WeightMult W, MatrixMult M, Source F>
class use_product_cache : public std::false_type
{};

/** Specialization for GENERAL product with diagonal weight
 *
 * FIXME : this should not be needed for MatrixType = VectorXd
 */
template<typename MatrixType, AssignType A>
class use_product_cache<MatrixType, A, DIAGONAL, GENERAL, EXTERNAL> : public std::true_type
{};
template<typename MatrixType, AssignType A>
class use_product_cache<MatrixType, A, DIAGONAL, GENERAL, CONSTANT> : public std::true_type
{};

/** Base class for the assignation */
template<AssignType A>
class AssignBase
{};

template<>
class AssignBase<COPY>
{
public:
  template<typename T, typename U>
  void assign(U & out, const T & in)
  {
    out.noalias() = in;
  }

  template<typename U>
  void assign(U & out, double in)
  {
    out.setConstant(in);
  }
};

template<>
class AssignBase<ADD>
{
public:
  template<typename T, typename U>
  void assign(U & out, const T & in)
  {
    out.noalias() += in;
  }

  template<typename U>
  void assign(U & out, double in)
  {
    out.array() += in;
  }
};

template<>
class AssignBase<SUB>
{
public:
  template<typename T, typename U>
  void assign(U & out, const T & in)
  {
    out.noalias() -= in;
  }

  template<typename U>
  void assign(U & out, double in)
  {
    out.array() -= in;
  }
};

template<>
class AssignBase<MIN>
{
public:
  template<typename T, typename U>
  void assign(U & out, const T & in)
  {
    out.array() = out.array().min(in.array());
  }

  template<typename U>
  void assign(U & out, double in)
  {
    out.array() = out.array().min(in);
  }
};

template<>
class AssignBase<MAX>
{
public:
  template<typename T, typename U>
  void assign(U & out, const T & in)
  {
    out.array() = out.array().max(in.array());
  }

  template<typename U>
  void assign(U & out, double in)
  {
    out.array() = out.array().max(in);
  }
};

/** Base class for the multiplication by a scalar*/
template<WeightMult W>
class WeightMultBase
{};

/** Specialization for NONE */
template<>
class WeightMultBase<NONE>
{
public:
  static const bool useArg = false;

  WeightMultBase(NoArg){};

  template<typename T>
  const T & applyWeightMult(const T & M)
  {
    return M;
  }
};

/** Specialization for MINUS */
template<>
class WeightMultBase<MINUS>
{
public:
  static const bool useArg = false;

  WeightMultBase(NoArg){};

  double applyWeightMult(const double & M) { return -M; }

  template<typename Derived>
  decltype(-std::declval<Eigen::MatrixBase<Derived>>()) applyWeightMult(const Eigen::MatrixBase<Derived> & M)
  {
    return -M;
  }

  /** We need this specialization because, oddly, -(A*B) relies on a temporary evaluation while (-A)*B does not*/
#if EIGEN_VERSION_AT_LEAST(3, 2, 90)
  template<typename Lhs, typename Rhs, int Option>
  decltype(-(std::declval<Lhs>().lazyProduct(std::declval<Rhs>()))) applyWeightMult(
      const Eigen::Product<Lhs, Rhs, Option> & P)
  {
    return -(P.lhs().lazyProduct(P.rhs()));
  }
#else
  template<typename Derived, typename Lhs, typename Rhs>
  decltype((-std::declval<Lhs>()) * std::declval<Rhs>()) applyWeightMult(const Eigen::ProductBase<Derived, Lhs, Rhs> & P)
  {
    return (-P.lhs()) * P.rhs();
  }
#endif
};

/** Specialization for SCALAR */
template<>
class WeightMultBase<SCALAR>
{
public:
  WeightMultBase(const double & s) : s_(s){};

  template<typename T>
  decltype(double() * std::declval<T>()) applyWeightMult(const T & M)
  {
    return s_ * M;
  }

#if EIGEN_VERSION_AT_LEAST(3, 2, 90)
  template<typename Lhs, typename Rhs, int Option>
  decltype(double() * (std::declval<Lhs>().lazyProduct(std::declval<Rhs>()))) applyWeightMult(
      const Eigen::Product<Lhs, Rhs, Option> & P)
  {
    return s_ * (P.lhs().lazyProduct(P.rhs()));
  }
#endif

private:
  const double & s_;
};

/** Specialization for DIAGONAL */
template<>
class WeightMultBase<DIAGONAL>
{
public:
  WeightMultBase(const Eigen::Ref<const Eigen::VectorXd> & d) : d_(d) {}

  template<typename T>
  using ReturnType = decltype(std::declval<Eigen::Ref<const Eigen::VectorXd>>().asDiagonal() * std::declval<T>());
  template<typename T>
  ReturnType<T> applyWeightMult(const T & M)
  {
    return d_.asDiagonal() * M;
  }

  /** Diagonal * constant vector case*/
  decltype(double() * std::declval<Eigen::Ref<const Eigen::VectorXd>>()) applyWeightMult(const double & d)
  {
    return d * d_;
  }

private:
  Eigen::Ref<const Eigen::VectorXd> d_;
};

/** Base class for the multiplication by a matrix*/
template<typename MatrixType, MatrixMult M>
class MatrixMultBase
{};

/** Partial specialization for IDENTITY*/
template<typename MatrixType>
class MatrixMultBase<MatrixType, IDENTITY>
{
public:
  MatrixMultBase(NoArg) {}

  template<typename T>
  const T & applyMatrixMult(const T & M)
  {
    return M;
  }
};

/** Partial specialization for GENERAL*/
template<typename MatrixType>
class MatrixMultBase<MatrixType, GENERAL>
{
public:
  MatrixMultBase(const Eigen::Ref<const Eigen::MatrixXd> & M) : M_(M) {}

  /** Return type of Matrix*T */
  template<typename T>
  using PreType = decltype(std::declval<Eigen::Ref<const Eigen::MatrixXd>>() * std::declval<T>());
  /** Return type of T*Matrix */
  template<typename T>
  using PostType = decltype(std::declval<T>() * std::declval<Eigen::Ref<const Eigen::MatrixXd>>());
  /** Return type of Matrix * ConstantVector */
  using ConstType = decltype(std::declval<Eigen::Ref<const Eigen::MatrixXd>>() * Eigen::VectorXd::Constant(1, 1));

  template<typename T>
  typename std::enable_if<isVector<MatrixType>::value, PreType<T>>::type applyMatrixMult(const T & M)
  {
    return M_ * M;
  }

  template<typename T>
  typename std::enable_if<!isVector<MatrixType>::value, PostType<T>>::type applyMatrixMult(const T & M)
  {
    return M * M_;
  }

  template<typename U = MatrixType>
  typename std::enable_if<isVector<U>::value, ConstType>::type applyMatrixMult(const double & d)
  {
    return M_ * Eigen::VectorXd::Constant(M_.cols(), d);
  }

  /** Cached version of applyMatrixMult*/
  template<typename T>
  void applyMatrixMultCached(MatrixType & cache, const T & M)
  {
    const auto p = applyMatrixMult(M);
    using ConstType = decltype(std::declval<Eigen::Ref<const Eigen::MatrixXd>>() * Eigen::VectorXd::Constant(1, 1));
    if constexpr(std::is_same_v<decltype(p), ConstType>)
    {
      // For the case where M = matrix * Constant, Eigen create a temporary to store  Vector:Constant before evaluating
      // the product. We avoid that here, by doing the product by hand. This could be further optimized by taking into
      // account the WeightMult at once, and possibly without using a cache.
      // The product A * v where v = c * 1 with c a scalar and 1 the vector of ones is equal to c * A * 1. We have that
      // A * 1 is the sum of column of A, what we leverage in the following computations:
      cache = p.lhs().rowwise().sum(); // TODO compare to a handmade loop summing the columns
      cache *= p.rhs().functor().m_other;
    }
    else
    {
      cache.noalias() = p;
    }
  }

private:
  Eigen::Ref<const Eigen::MatrixXd> M_;
};

/** Partial specialization for INVERSE_DIAGONAL*/
template<typename MatrixType>
class MatrixMultBase<MatrixType, INVERSE_DIAGONAL>
{
public:
  MatrixMultBase(const Eigen::Ref<const Eigen::MatrixXd> & M) : M_(M) {}

  /** Type of the diagonal inverse*/
  using InvDiagType =
      decltype(std::declval<Eigen::Ref<const Eigen::MatrixXd>>().diagonal().cwiseInverse().asDiagonal());
  /** Return type of Matrix*T */
  template<typename T>
  using PreType = decltype(std::declval<InvDiagType>() * std::declval<T>());
  /** Return type of T*Matrix */
  template<typename T>
  using PostType = decltype(std::declval<T>() * std::declval<InvDiagType>());
  /** Return type of Matrix * ConstantVector */
  using ConstType = decltype(std::declval<InvDiagType>() * Eigen::VectorXd::Constant(1, 1));

  template<typename T>
  typename std::enable_if<isVector<MatrixType>::value, PreType<T>>::type applyMatrixMult(const T & M)
  {
    return M_.diagonal().cwiseInverse().asDiagonal() * M;
  }

  template<typename T>
  typename std::enable_if<!isVector<MatrixType>::value, PostType<T>>::type applyMatrixMult(const T & M)
  {
    return M * M_.diagonal().cwiseInverse().asDiagonal();
  }

  template<typename U = MatrixType>
  typename std::enable_if<isVector<U>::value, ConstType>::type applyMatrixMult(const double & d)
  {
    return M_.diagonal().cwiseInverse().asDiagonal() * Eigen::VectorXd::Constant(M_.cols(), d);
  }

  /** Cached version of applyMatrixMult*/
  template<typename T>
  void applyMatrixMultCached(MatrixType & cache, const T & M)
  {
    cache.noalias() = applyMatrixMult(M);
  }

private:
  Eigen::Ref<const Eigen::MatrixXd> M_;
};

/** Partial specialization for CUSTOM*/
template<typename MatrixType>
class MatrixMultBase<MatrixType, CUSTOM>
{
public:
  MatrixMultBase(void (*mult)(Eigen::Ref<MatrixType> out, const Eigen::Ref<const MatrixType> & in)) : mult_(mult) {}

  template<typename T>
  void applyMatrixMultCached(MatrixType & cache, const T & M)
  {
    mult_(cache, M);
  }

private:
  void (*mult_)(Eigen::Ref<MatrixType> out, const Eigen::Ref<const MatrixType> & in);
};

/** Base class for managing the source*/
template<typename MatrixType, Source F>
class SourceBase
{
public:
  using SourceType = typename std::conditional<F == CONSTANT, double, Eigen::Ref<const MatrixType>>::type;

  SourceBase(const SourceType & from) : from_(from) {}

  const SourceType & from() const { return from_; }

  void from(const SourceType & from)
  {
    // We want to do from_ = from but there is no operator= for Eigen::Ref,
    // so we need to use a placement new.
    new(&from_) SourceType(from);
  }

private:
  SourceType from_;
};

/** Partial specialization for ZERO*/
template<typename MatrixType>
class SourceBase<MatrixType, ZERO>
{};

/** The main class. Its run method performs the assignment t = op(t, w*M*f)
 * (if f is a vector) or t = op(t, w*f*M) (if f is a matrix)
 * where
 *  - t is the target matrix/vector
 *  - f is the source matrix/vector
 *  - op is described by A
 *  - w is a scalar, user supplied if W is SCALAR, +/-1 if W is NONE or
 *    MINUS, and a vector if W is DIAGONAL or INVERSE_DIAGONAL
 *  - M is a matrix, either the identity or user-supplied, depending on
 *    the template parameter M (see MatrixMult)
 * If F=EXTERNAl f is a user supplied Eigen::Ref<MatrixType>, if F=ZERO,
 * f=0 and if F=CONSTANT, f is a constant vector (vector only).
 *
 * This class is meant to be a helper class and should not live on its own,
 * but be create by a higher-level class ensuring its data are valid.
 *
 * \internal TODO:
 * - use an object for custom multiplications
 * - resize of cache at initialization
 * - reference on all inputs? (not the case now for inputs that are double)
 */
template<typename MatrixType, AssignType A, WeightMult W, MatrixMult M, Source F = EXTERNAL>
class CompiledAssignment : public CachedResult<MatrixType,
                                               use_assign_cache<MatrixType, A, W, M, F>::value
                                                   || use_product_cache<MatrixType, A, W, M, F>::value>,
                           public AssignBase<A>,
                           public WeightMultBase<W>,
                           public MatrixMultBase<MatrixType, M>,
                           public SourceBase<MatrixType, F>
{
private:
  using CBase =
      CachedResult<MatrixType,
                   use_assign_cache<MatrixType, A, W, M, F>::value || use_product_cache<MatrixType, A, W, M, F>::value>;
  using WBase = WeightMultBase<W>;
  using MBase = MatrixMultBase<MatrixType, M>;
  using SBase = SourceBase<MatrixType, F>;
  using SParse =
      typename std::conditional<hasNoArgCtor<SBase>::value, ParseArg<-1>, ParseArg<ArgCount<SBase>::count - 1>>::type;
  using WParse = typename std::
      conditional<hasNoArgCtor<WBase>::value, ParseArg<-1>, ParseArg<ArgCount<SBase, WBase>::count - 1>>::type;
  using MParse = typename std::
      conditional<hasNoArgCtor<MBase>::value, ParseArg<-1>, ParseArg<ArgCount<SBase, WBase, MBase>::count - 1>>::type;

  /** Constructor. Arguments are given as follows:
   * \param to output matrix/vector.
   * \param from input matrix/vector (only for F = EXTERNAL or F = CONSTANT).
   * \param w weight. (only for W = SCALAR, DIAGONAL or INVERSE_DIAGONAL. It
   * is a scalar for W = SCALAR, and a vector for W = DIAGONAL or
   * W = INVERSE_DIAGONAL.
   * \param M matrix used in the multiplication (only for M = GENERAL or
   * M = CUSTOM)
   */
  template<typename... Args>
  CompiledAssignment(const Eigen::Ref<MatrixType> & to, Args &&... args)
  : CBase(to), WBase(WParse::get(std::forward<Args>(args)...)), MBase(MParse::get(std::forward<Args>(args)...)),
    SBase(SParse::get(std::forward<Args>(args)...)), to_(to)
  {
    static_assert(!(isMatrix<MatrixType>::value && F == CONSTANT), "Constant source is only for vectors.");
  }

public:
  template<typename U = MatrixType>
  typename std::enable_if<!use_product_cache<U, A, W, M, F>::value>::type run()
  {
    // There is room for speed improvement by switching at runtime in function of the
    // matrices sizes, in particular for M = GENERAL, it seems that lazy product is
    // faster for small matrices (but slower for bigger ones)
    this->assign(to_, this->cache(this->applyWeightMult(this->applyMatrixMult(this->from()))));
  }

  template<typename U = MatrixType>
  typename std::enable_if<use_product_cache<U, A, W, M, F>::value && !use_assign_cache<U, A, W, M, F>::value>::type run()
  {
    this->applyMatrixMultCached(this->cache(), this->from());
    this->assign(to_, this->applyWeightMult(this->cache()));
  }

  template<typename U = MatrixType>
  typename std::enable_if<use_product_cache<U, A, W, M, F>::value && use_assign_cache<U, A, W, M, F>::value>::type run()
  {
    this->applyMatrixMultCached(this->cache(), this->from());
    this->cache() = this->applyWeightMult(this->cache());
    this->assign(to_, this->cache());
  }

  void to(const Eigen::Ref<MatrixType> & to)
  {
    // We want to do to_ = to but there is no operator= for Eigen::Ref,
    // so we need to use a placement new.
    new(&to_) Eigen::Ref<MatrixType>(to);
  }

private:
  /** Warning: it is the user responsibility to ensure that the matrix/vector
   * pointed to by from_, to_ and, if applicable, M_ stay alive.*/
  Eigen::Ref<MatrixType> to_;

  template<typename MatrixType_>
  friend class CompiledAssignmentWrapper;
};

/** Specialization for F=0. The class does nothing in the general case.*/
template<typename MatrixType, AssignType A, WeightMult W, MatrixMult M>
class CompiledAssignment<MatrixType, A, W, M, ZERO>
{
public:
  using SourceType = Eigen::Ref<const MatrixType>;

private:
  CompiledAssignment(const Eigen::Ref<MatrixType> & to) : to_(to) {}

public:
  void run() { /* Do nothing */ }
  void from(const Eigen::Ref<const MatrixType> &) { /* Do nothing */ }
  void to(const Eigen::Ref<MatrixType> & to)
  {
    // We want to do to_ = to but there is no operator= for Eigen::Ref,
    // so we need to use a placement new.
    new(&to_) Eigen::Ref<MatrixType>(to);
  }

private:
  Eigen::Ref<MatrixType> to_;

  template<typename MatrixType_>
  friend class CompiledAssignmentWrapper;
};

/** Specialization for F=0 and A=COPY.*/
template<typename MatrixType, WeightMult W, MatrixMult M>
class CompiledAssignment<MatrixType, COPY, W, M, ZERO>
{
public:
  using SourceType = Eigen::Ref<const MatrixType>;

private:
  CompiledAssignment(const Eigen::Ref<MatrixType> & to) : to_(to) {}

public:
  void run() { to_.setZero(); }
  void from(const Eigen::Ref<const MatrixType> &) { /* Do nothing */ }
  void to(const Eigen::Ref<MatrixType> & to)
  {
    // We want to do to_ = to but there is no operator= for Eigen::Ref,
    // so we need to use a placement new.
    new(&to_) Eigen::Ref<MatrixType>(to);
  }

private:
  Eigen::Ref<MatrixType> to_;

  template<typename MatrixType_>
  friend class CompiledAssignmentWrapper;
};

/** Specialization for F=0 and A=MIN.*/
template<typename MatrixType, WeightMult W, MatrixMult M>
class CompiledAssignment<MatrixType, MIN, W, M, ZERO>
{
public:
  using SourceType = Eigen::Ref<const MatrixType>;

private:
  CompiledAssignment(const Eigen::Ref<MatrixType> & to) : to_(to) {}

public:
  void run() { to_.array() = to_.array().min(0); }
  void from(const Eigen::Ref<const MatrixType> &) { /* Do nothing */ }
  void to(const Eigen::Ref<MatrixType> & to) { new(&to_) Eigen::Ref<MatrixType>(to); }

private:
  Eigen::Ref<MatrixType> to_;

  template<typename MatrixType_>
  friend class CompiledAssignmentWrapper;
};

/** Specialization for F=0 and A=MAX.*/
template<typename MatrixType, WeightMult W, MatrixMult M>
class CompiledAssignment<MatrixType, MAX, W, M, ZERO>
{
public:
  using SourceType = Eigen::Ref<const MatrixType>;

private:
  CompiledAssignment(const Eigen::Ref<MatrixType> & to) : to_(to) {}

public:
  void run() { to_.array() = to_.array().max(0); }
  void from(const Eigen::Ref<const MatrixType> &) { /* Do nothing */ }
  void to(const Eigen::Ref<MatrixType> & to) { new(&to_) Eigen::Ref<MatrixType>(to); }

private:
  Eigen::Ref<MatrixType> to_;

  template<typename MatrixType_>
  friend class CompiledAssignmentWrapper;
};

} // namespace internal

} // namespace scheme

} // namespace tvm
