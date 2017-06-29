#pragma once

#include <type_traits>

#define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Core>

namespace taskvm
{
  namespace utils
  {
    enum AssignType
    {
      REPLACE,      // to =  from
      ADD,          // to += from
      SUB,          // to -= from
      MIN,          // to = min(to, from)
      MAX           // to = max(to, from)
    };

    enum ScalarMult
    {
      NONE,         // from
      MINUS,        // -from
      SCALAR        // alpha*from
    };

    enum MatrixMult
    {
      IDENTITY,     // from
      DIAGONAL,     // diag(w)*from or from*diag(w)
      GENERAL,      // M*from or from*M
      TBD
    };

    enum MultPos
    {
      PRE,          // M*from
      POST          // from*M
    };


    template<typename MatrixType, bool Cache>
    struct CachedResult
    {
      template<typename T>
      const T& cache(const T& M) { return M; }
    };

    template<typename MatrixType>
    struct CachedResult<MatrixType, true>
    {
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
    template<typename MatrixType, AssignType A, ScalarMult S, MatrixMult M, MultPos P>
    struct cache_traits : public std::false_type {};

    /** Specialization for min/max with general matrix product. In this case, we use the cache*/
    template<typename MatrixType, ScalarMult S, MultPos P>
    struct cache_traits<MatrixType, MIN, S, GENERAL, P> : public std::true_type {};
    template<typename MatrixType, ScalarMult S, MultPos P>
    struct cache_traits<MatrixType, MAX, S, GENERAL, P> : public std::true_type {};


    /** Base struct for the assignation */
    template<AssignType A>  struct AssignBase {};

    template<>
    struct AssignBase<REPLACE>
    {
      template <typename T, typename U>
      void assign(const T& in, U& out) { out.noalias() = in; }
    };

    template<>
    struct AssignBase<ADD>
    {
      template <typename T, typename U>
      void assign(const T& in, U& out) { out.noalias() += in; }
    };

    template<>
    struct AssignBase<SUB>
    {
      template <typename T, typename U>
      void assign(const T& in, U& out) { out.noalias() -= in; }
    };

    template<>
    struct AssignBase<MIN>
    {
      template <typename T, typename U>
      void assign(const T& in, U& out) { out.array().min(in.array()); }
    };

    template<>
    struct AssignBase<MAX>
    {
      template <typename T, typename U>
      void assign(const T& in, U& out) { out.array().max(in.array()); }
    };


    /** Base struct for the multiplication by a scalar*/
    template<ScalarMult S> struct ScalarMultBase {};

    /** Specialization for NONE */
    template<>
    struct ScalarMultBase<NONE>
    {
      ScalarMultBase(double) {};

      template<typename T>
      const T& applyScalarMult(const T& M) { return M; }
    };

    /** Specialization for MINUS */
    template<>
    struct ScalarMultBase<MINUS>
    {
      ScalarMultBase(double) {};

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
    struct ScalarMultBase<SCALAR>
    {
      ScalarMultBase(double s) : s_(s) {};

      template<typename T>
      decltype(double()*std::declval<T>()) applyScalarMult(const T& M) { return s_*M; }

    private:
      double s_;
    };


    /** Base struct for the multiplication by a matrix*/
    template<MatrixMult M, MultPos P> struct MatrixMultBase {};

    /** Partial specialization for IDENTITY*/
    template<MultPos P>
    struct MatrixMultBase<IDENTITY, P>
    {
      typedef void MultType;
      MatrixMultBase(const MultType* const) {}

      template<typename T>
      const T& applyMatrixMult(const T& M) { return M; }
    };

    /** Partial specialization for DIAGONAL*/
    template<MultPos P>
    struct MatrixMultBase<DIAGONAL, P>
    {
      typedef Eigen::VectorXd MultType;
      MatrixMultBase(const MultType* const w) : w_(*w) { assert(w != nullptr); }

      /** Return type of VectorXd.asDiagonal()*T */
      template<typename T>
      using PreType = decltype(Eigen::VectorXd().asDiagonal()*std::declval<T>());
      /** Return type of T*VectorXd.asDiagonal() */
      template<typename T>
      using PostType = decltype(std::declval<T>()*Eigen::VectorXd().asDiagonal());
      template<typename T, MultPos P>
      using ReturnType = typename std::conditional<P == PRE, PreType<T>, PostType<T>>::type;

      template<typename T>
      ReturnType<T, P> applyMatrixMult(const T& M);

    private:
      const Eigen::VectorXd& w_;
    };

    /** Partial specialization for GENERAL*/
    template<MultPos P>
    struct MatrixMultBase<GENERAL, P>
    {
      typedef Eigen::MatrixXd MultType;
      MatrixMultBase(const MultType* const M) : M_(*M) { assert(M != nullptr); }

      /** Return type of MatrixXd*T */
      template<typename T>
      using PreType = decltype(Eigen::MatrixXd()*std::declval<T>());
      /** Return type of T*MatrixXd */
      template<typename T>
      using PostType = decltype(std::declval<T>()*Eigen::MatrixXd());
      template<typename T, MultPos P>
      using ReturnType = typename std::conditional<P == PRE, PreType<T>, PostType<T>>::type;

      template<typename T>
      ReturnType<T, P> applyMatrixMult(const T& M);

    private:
      const Eigen::MatrixXd& M_;
    };



    /** The main class */
    template<typename MatrixType, AssignType A, ScalarMult S, MatrixMult M, MultPos P>
    struct CompiledAssignment
      : public CachedResult<MatrixType, cache_traits<MatrixType, A, S, M, P>::value>
      , public AssignBase<A>
      , public ScalarMultBase<S>
      , public MatrixMultBase<M, P>
    {
      CompiledAssignment(const Eigen::Ref<const MatrixType>& from, Eigen::Ref<MatrixType> to, double s = 0, const typename MatrixMultBase<M, P>::MultType* const m = nullptr)
        : ScalarMultBase<S>(s)
        , MatrixMultBase<M, P>(m)
        , from_(from)
        , to_(to) {}

      void run()
      {
        assign(cache(applyScalarMult(applyMatrixMult(from_))), to_);
      }

    private:
      /** Warning: it is the user responsability to ensure that the matrix/vector
      * pointed to by from_, to_ and, if applicable, M_ stay alive.*/
      Eigen::Ref<const MatrixType> from_;
      Eigen::Ref<MatrixType> to_;
    };




    template<>
    template<typename T>
    inline MatrixMultBase<DIAGONAL, PRE>::ReturnType<T, PRE> MatrixMultBase<DIAGONAL, PRE>::applyMatrixMult(const T& M)
    {
      return w_.asDiagonal()*M;
    }

    template<>
    template<typename T>
    inline MatrixMultBase<DIAGONAL, POST>::ReturnType<T, POST> MatrixMultBase<DIAGONAL, POST>::applyMatrixMult(const T& M)
    {
      return M*w_.asDiagonal();
    }

    template<>
    template<typename T>
    inline MatrixMultBase<GENERAL, PRE>::ReturnType<T, PRE> MatrixMultBase<GENERAL, PRE>::applyMatrixMult(const T& M)
    {
      return M_*M;
    }

    template<>
    template<typename T>
    inline MatrixMultBase<GENERAL, PRE>::ReturnType<T, POST> MatrixMultBase<GENERAL, POST>::applyMatrixMult(const T& M)
    {
      return M*M_;
    }
  }
}
