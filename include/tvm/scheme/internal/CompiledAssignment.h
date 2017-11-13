#pragma once

/* Copyright 2017 CNRS-UM LIRMM, CNRS-AIST JRL
 *
 * This file is part of TVM.
 *
 * TVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with TVM.  If not, see <http://www.gnu.org/licenses/>.
 */

#define EIGEN_RUNTIME_NO_MALLOC
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
    COPY,         // to =  from
    ADD,          // to += from
    SUB,          // to -= from
    MIN,          // to = min(to, from)
    MAX           // to = max(to, from)
  };

  /** Specify whether \a from is to be multiplied by 1, -1 or a user
    * specified scalar.
    */
  enum ScalarMult
  {
    NONE,         // from
    MINUS,        // -from
    SCALAR        // s*from
  };

  /** Specify if from is to be multiplied a matrix, and if so, what is the
    * type of the matrix.
    */
  enum MatrixMult
  {
    IDENTITY,     // from
    DIAGONAL,     // diag(w)*from or from*diag(w)
    GENERAL,      // M*from or from*M
    TBD
  };

  /** Specify if from is to be left- or right- multiplied by the matrix.*/
  enum MultPos
  {
    PRE,          // M*from
    POST          // from*M
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
    EXTERNAL,     // source is an external vector or matrix (main use-case)
    ZERO,         // source is zero
    CONSTANT      // source is a (non-zero) constant
  };

  template<typename MatrixType, bool Cache>
  class CachedResult
  {
  public:
    template<typename T>
    const T& cache(const T& M) { return M; }
  };

  template<typename MatrixType>
  class CachedResult<MatrixType, true>
  {
  public:
    template<typename T>
    const MatrixType& cache(const T& M)
    {
#ifdef AUTHORIZE_MALLOC_FOR_CACHE
      if (!Eigen::internal::is_malloc_allowed())
      {
        Eigen::internal::set_is_malloc_allowed(true);
        cache_ = M; return cache_;
        Eigen::internal::set_is_malloc_allowed(false);
      }
      else
#endif
        cache_ = M;
      return cache_;
    }

  private:
    MatrixType cache_;
  };


  /** Traits for deciding whether or not to use a cache. By default, no cache is used.*/
  template<typename MatrixType, AssignType A, ScalarMult S, MatrixMult M, MultPos P, Source F>
  class cache_traits : public std::false_type {};

  /** Specialization for min/max with general matrix product. In this case, we use the cache*/
  template<typename MatrixType, ScalarMult S, MultPos P>
  class cache_traits<MatrixType, MIN, S, GENERAL, P, EXTERNAL> : public std::true_type {};
  template<typename MatrixType, ScalarMult S, MultPos P>
  class cache_traits<MatrixType, MAX, S, GENERAL, P, EXTERNAL> : public std::true_type {};
  /** Specialization for GENERAL*CONSTANT. This should not be necessary, but
    * the product needs a temporary. Maybe it's not the case anymore with Eigen 3.3*/
  template<typename MatrixType, AssignType A, ScalarMult S, MultPos P>
  class cache_traits<MatrixType, A, S, GENERAL, P, CONSTANT> : public std::true_type {};

  /** Base class for the assignation */
  template<AssignType A>  class AssignBase {};

  template<>
  class AssignBase<COPY>
  {
  public:
    template <typename T, typename U>
    void assign(const T& in, U& out) { out.noalias() = in; }

    template <typename U>
    void assign(double in, U& out) { out.setConstant(in); }
  };

  template<>
  class AssignBase<ADD>
  {
  public:
    template <typename T, typename U>
    void assign(const T& in, U& out) { out.noalias() += in; }

    template <typename U>
    void assign(double in, U& out) { out.array() += in; }
  };

  template<>
  class AssignBase<SUB>
  {
  public:
    template <typename T, typename U>
    void assign(const T& in, U& out) { out.noalias() -= in; }

    template <typename U>
    void assign(double in, U& out) { out.array() -= in; }
  };

  template<>
  class AssignBase<MIN>
  {
  public:
    template <typename T, typename U>
    void assign(const T& in, U& out) { out.array() = out.array().min(in.array()); }

    template <typename U>
    void assign(double in, U& out) { out.array() = out.array().min(in); }
  };

  template<>
  class AssignBase<MAX>
  {
  public:
    template <typename T, typename U>
    void assign(const T& in, U& out) { out.array() = out.array().max(in.array()); }

    template <typename U>
    void assign(double in, U& out) { out.array() = out.array().max(in); }
  };


  /** Base class for the multiplication by a scalar*/
  template<ScalarMult S> class ScalarMultBase {};

  /** Specialization for NONE */
  template<>
  class ScalarMultBase<NONE>
  {
  public:
    ScalarMultBase(double) {};

    template<typename T>
    const T& applyScalarMult(const T& M) { return M; }
  };

  /** Specialization for MINUS */
  template<>
  class ScalarMultBase<MINUS>
  {
  public:
    ScalarMultBase(double) {};

    double applyScalarMult(const double& M) { return -M; }

    template<typename Derived>
    decltype(-std::declval<Eigen::MatrixBase<Derived> >()) applyScalarMult(const Eigen::MatrixBase<Derived>& M) { return -M; }

    /** We need this specialization because, odly, -(A*B) relies on a temporary evaluation while (-A)*B does not*/
    template<typename Derived, typename Lhs, typename Rhs>
    decltype((-std::declval<Lhs>())*std::declval<Rhs>()) applyScalarMult(const Eigen::ProductBase<Derived, Lhs, Rhs>& P)
    {
      return (-P.lhs())*P.rhs();
    }
  };

  /** Specialization for SCALAR */
  template<>
  class ScalarMultBase<SCALAR>
  {
  public:
    ScalarMultBase(double s) : s_(s) {};

    template<typename T>
    decltype(double()*std::declval<T>()) applyScalarMult(const T& M) { return s_*M; }

  private:
    double s_;
  };


  /** Base class for the multiplication by a matrix*/
  template<MatrixMult M, MultPos P> class MatrixMultBase {};

  /** Partial specialization for IDENTITY*/
  template<MultPos P>
  class MatrixMultBase<IDENTITY, P>
  {
  public:
    typedef void MultType;
    MatrixMultBase(const MultType* const) {}

    template<typename T>
    const T& applyMatrixMult(const T& M) { return M; }
  };

  /** Partial specialization for DIAGONAL*/
  template<MultPos P>
  class MatrixMultBase<DIAGONAL, P>
  {
  public:
    typedef Eigen::VectorXd MultType;
    MatrixMultBase(const MultType* const w) : w_(*w) { assert(w != nullptr); }

    /** Return type of VectorXd.asDiagonal()*T */
    template<typename T>
    using PreType = decltype(Eigen::VectorXd().asDiagonal()*std::declval<T>());
    /** Return type of T*VectorXd.asDiagonal() */
    template<typename T>
    using PostType = decltype(std::declval<T>()*Eigen::VectorXd().asDiagonal());
    template<typename T>
    using ReturnType = typename std::conditional<P == PRE, PreType<T>, PostType<T>>::type;

    template<typename T>
    ReturnType<T> applyMatrixMult(const T& M);

    /** overload for T=double, i.e. we emulate w_.asDiagonal()*VectorXd::Constant(size,cst)*/
    decltype(double()*Eigen::VectorXd()) applyMatrixMult(const double& d)
    {
      return d*w_;
    }

  private:
    const Eigen::VectorXd& w_;
  };

  /** Partial specialization for GENERAL*/
  template<MultPos P>
  class MatrixMultBase<GENERAL, P>
  {
  public:
    typedef Eigen::MatrixXd MultType;
    MatrixMultBase(const MultType* const M) : M_(*M) { assert(M != nullptr); }

    /** Return type of MatrixXd*T */
    template<typename T>
    using PreType = decltype(Eigen::MatrixXd()*std::declval<T>());
    /** Return type of T*MatrixXd */
    template<typename T>
    using PostType = decltype(std::declval<T>()*Eigen::MatrixXd());
    template<typename T>
    using ReturnType = typename std::conditional<P == PRE, PreType<T>, PostType<T>>::type;

    template<typename T>
    ReturnType<T> applyMatrixMult(const T& M);

    decltype(Eigen::MatrixXd()*Eigen::VectorXd::Constant(1, 1)) applyMatrixMult(const double& d)
    {
      return M_*Eigen::VectorXd::Constant(M_.cols(), d);
    }

  private:
    const Eigen::MatrixXd& M_;
  };


  /** Base class for managing the source*/
  template<typename MatrixType, Source F>
  class SourceBase
  {
  public:
    using SourceType = typename std::conditional<F==CONSTANT, double, Eigen::Ref<const MatrixType>>::type;

    SourceBase(const SourceType& from) : from_(from) {}

    const SourceType& from() const
    {
      return from_;
    }

    void from(const SourceType& from)
    {
      // We want to do from_ = from but there is no operator= for Eigen::Ref,
      // so we need to use a placement new.
      new (&from_) SourceType(from);
    }

  private:
    SourceType from_;
  };

  /** Partial specialization for ZERO*/
  template<typename MatrixType>
  class SourceBase<MatrixType, ZERO> {};



  /** The main class. Its run method perfoms the assignment t = op(t, s*M*f)
    * (for P=PRE) or t = op(t, s*f*M) (for P=POST)
    * where
    *  - t is the target matrix/vector
    *  - f is the source matrix/vector
    *  - op is described by A
    *  - s is a scalar, user supplied if S is SCALAR, +/-1 otherwise
    *  - M is a matrix, either the identity or user-supplied, depending on
    *    the template parameter M (see MatrixMult)
    * If F=EXTERNAl f is a user supplied Eigen::Ref<MatrixType>, if F=ZERO,
    * f=0 and if F=CONSTANT, f is a constant vector (vector only).
    *
    * This class is meant to be a helper class and should not live on its own,
    * but be create by a higher-level class ensuring its data are valid.
    *
    * FIXME get rid of PRE/POST and have the assignment t = op(t, s*M1*f*M2)
    * instead (needed for substitution in constraints with anistropic weight.
    */
  template<typename MatrixType, AssignType A, ScalarMult S, MatrixMult M, MultPos P, Source F=EXTERNAL>
  class CompiledAssignment
    : public CachedResult<MatrixType, cache_traits<MatrixType, A, S, M, P, F>::value>
    , public AssignBase<A>
    , public ScalarMultBase<S>
    , public MatrixMultBase<M, P>
    , public SourceBase<MatrixType, F>
  {
  private:
    CompiledAssignment(const typename SourceBase<MatrixType, F>::SourceType& from, const Eigen::Ref<MatrixType>& to, double s = 1, const typename MatrixMultBase<M, P>::MultType* const m = nullptr)
      : ScalarMultBase<S>(s)
      , MatrixMultBase<M, P>(m)
      , SourceBase<MatrixType,F>(from)
      , to_(to)
    {
      static_assert(F == EXTERNAL || (MatrixType::ColsAtCompileTime == 1 && P == PRE), "For F=CONSTANT, only vector type and pre-multiplications are allowed");
    }

  public:
    //FIXME shall we use the operator() instead of run, and make this class a functor ?
    void run()
    {
      this->assign(this->cache(this->applyScalarMult(this->applyMatrixMult(this->from()))), to_);
    }

    void to(const Eigen::Ref<MatrixType>& to)
    {
      // We want to do to_ = to but there is no operator= for Eigen::Ref,
      // so we need to use a placement new.
      new (&to_) Eigen::Ref<MatrixType>(to);
    }

  private:
    /** Warning: it is the user responsability to ensure that the matrix/vector
      * pointed to by from_, to_ and, if applicable, M_ stay alive.*/
    Eigen::Ref<MatrixType> to_;

    template<typename MatrixType_>
    friend class CompiledAssignmentWrapper;
  };


  /** Specialization for F=0. The class does nothing in the general case.*/
  template<typename MatrixType, AssignType A, ScalarMult S, MatrixMult M, MultPos P>
  class CompiledAssignment<MatrixType, A, S, M, P, ZERO>
  {
  private:
    CompiledAssignment(const Eigen::Ref<MatrixType>& to) : to_(to) {}

  public:
    void run() {/* Do nothing */ }
    void from(const Eigen::Ref<const MatrixType>&) {/* Do nothing */}
    void to(const Eigen::Ref<MatrixType>& to)
    {
      // We want to do to_ = to but there is no operator= for Eigen::Ref,
      // so we need to use a placement new.
      new (&to_) Eigen::Ref<MatrixType>(to);
    }

  private:
    Eigen::Ref<MatrixType> to_;

    template<typename MatrixType_>
    friend class CompiledAssignmentWrapper;
  };


  /** Specialization for F=0 and A=COPY.*/
  template<typename MatrixType, ScalarMult S, MatrixMult M, MultPos P>
  class CompiledAssignment<MatrixType, COPY, S, M, P, ZERO>
  {
  private:
    CompiledAssignment(const Eigen::Ref<MatrixType>& to): to_(to) {}

  public:
    void run() { to_.setZero(); }
    void from(const Eigen::Ref<const MatrixType>&) {/* Do nothing */ }
    void to(const Eigen::Ref<MatrixType>& to)
    {
      // We want to do to_ = to but there is no operator= for Eigen::Ref,
      // so we need to use a placement new.
      new (&to_) Eigen::Ref<MatrixType>(to);
    }

  private:
    Eigen::Ref<MatrixType> to_;

    template<typename MatrixType_>
    friend class CompiledAssignmentWrapper;
  };




  template<>
  template<typename T>
  inline MatrixMultBase<DIAGONAL, PRE>::ReturnType<T> MatrixMultBase<DIAGONAL, PRE>::applyMatrixMult(const T& M)
  {
    return w_.asDiagonal()*M;
  }

  template<>
  template<typename T>
  inline MatrixMultBase<DIAGONAL, POST>::ReturnType<T> MatrixMultBase<DIAGONAL, POST>::applyMatrixMult(const T& M)
  {
    return M*w_.asDiagonal();
  }

  template<>
  template<typename T>
  inline MatrixMultBase<GENERAL, PRE>::ReturnType<T> MatrixMultBase<GENERAL, PRE>::applyMatrixMult(const T& M)
  {
    return M_*M;
  }

  template<>
  template<typename T>
  inline MatrixMultBase<GENERAL, POST>::ReturnType<T> MatrixMultBase<GENERAL, POST>::applyMatrixMult(const T& M)
  {
    return M*M_;
  }

}  // namespace internal

}  // namespace scheme

}  // namespace tvm
