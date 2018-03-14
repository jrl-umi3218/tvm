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

#include <tvm/scheme/internal/CompiledAssignment.h>

#include <Eigen/Core>

#include <memory>

namespace tvm
{

namespace scheme
{

namespace internal
{
#if 0
  class Assignment;

  namespace helper
  {
    template<Source F>
    class from_FunctionBuilder;
  }


  /** This class wraps a CompiledAssignment so as to hide the template
    * machinery. The three member functions of CompiledAssignment are
    * exposed: run, from, to.
    */
  template<typename MatrixType>
  class CompiledAssignmentWrapper
  {
  public:
    CompiledAssignmentWrapper(const CompiledAssignmentWrapper<MatrixType>& other);
    CompiledAssignmentWrapper(CompiledAssignmentWrapper<MatrixType>&& other);
    CompiledAssignmentWrapper<MatrixType>& operator=(CompiledAssignmentWrapper<MatrixType> other);
    ~CompiledAssignmentWrapper();

    template<AssignType A, ScalarMult S, MatrixMult M, MultPos P, typename T>
    static CompiledAssignmentWrapper make(const T& from, const Eigen::Ref<MatrixType>& to,
      double s = 1, const typename MatrixMultBase<M, P>::MultType* const m = nullptr);

    template<AssignType A>
    static CompiledAssignmentWrapper make(const Eigen::Ref<MatrixType>& to);

    std::function<void()> run;
    std::function<void(const Eigen::Ref<MatrixType>&)> to;

    void from(double from);
    void from(const Eigen::Ref<const MatrixType>& from);

  private:
    CompiledAssignmentWrapper() = default;

    template<typename T>
    friend void pseudoSwap(CompiledAssignmentWrapper<T>&, CompiledAssignmentWrapper<T>&);
    friend class tvm::scheme::internal::Assignment;

    template<AssignType A, ScalarMult S, MatrixMult M, MultPos P>
    void construct(const Eigen::Ref<const MatrixType>& from, const Eigen::Ref<MatrixType>& to,
      double s, const typename MatrixMultBase<M, P>::MultType* const m);

    template<AssignType A, ScalarMult S, MatrixMult M, MultPos P>
    void construct(double from, const Eigen::Ref<MatrixType>& to,
      double s, const typename MatrixMultBase<M, P>::MultType* const m);

    template<AssignType A, ScalarMult S, MatrixMult M, MultPos P>
    void construct(const Eigen::Ref<MatrixType>& to);

    template<AssignType A, ScalarMult S, MatrixMult M, MultPos P, Source F>
    void constructFunctions();

    /** We store the wrapped CompiledAssignment as a void*. This allows us to
      * have a non templated wrapper.
      * We of course need to keep trace of the real type. This is done through
      * the various std::function member of the class.
      */
    void* compiledAssignment_ = nullptr;

    std::function<void()> delete_;
    std::function<void(CompiledAssignmentWrapper*)> copy_;
    std::function<void(CompiledAssignmentWrapper*)> build_;
    std::function<void(const Eigen::Ref<const MatrixType>&)> from_;

    template<Source F>
    friend class helper::from_FunctionBuilder;
  };

  namespace helper
  {
    template<Source F>
    class from_FunctionBuilder
    {
    public:
      template<typename CA, typename MatrixType>
      static std::function<void(const Eigen::Ref<const MatrixType>&)> construct(CompiledAssignmentWrapper<MatrixType>* wrapper)
      {
        return [wrapper](const Eigen::Ref<const MatrixType>& from)
        {
          static_cast<CA*>(wrapper->compiledAssignment_)->from(from);
        };
      }
    };

    template<>
    class from_FunctionBuilder<Source::ZERO>
    {
    public:
      template<typename CA, typename MatrixType>
      static std::function<void(const Eigen::Ref<const MatrixType>&)> construct(CompiledAssignmentWrapper<MatrixType>*)
      {
        return [](const Eigen::Ref<const MatrixType>&)
        {
          //do nothing
        };
      }
    };

    template<>
    class from_FunctionBuilder<Source::CONSTANT>
    {
    public:
      template<typename CA, typename MatrixType>
      static std::function<void(const Eigen::Ref<const MatrixType>&)> construct(CompiledAssignmentWrapper<MatrixType>* wrapper)
      {
        return [wrapper](const Eigen::Ref<const MatrixType>& from)
        {
          static_cast<CA*>(wrapper->compiledAssignment_)->from(from[0]);
        };
      }
    };
  }  // namespace helper


  template<typename MatrixType>
  inline CompiledAssignmentWrapper<MatrixType>::CompiledAssignmentWrapper(const CompiledAssignmentWrapper<MatrixType>& other)
  {
    other.copy_(this);
    this->build_ = other.build_;
    build_(this);
  }

  template<typename MatrixType>
  inline CompiledAssignmentWrapper<MatrixType>::CompiledAssignmentWrapper(CompiledAssignmentWrapper<MatrixType>&& other)
  {
    pseudoSwap(*this, other);
    build_(this);
  }

  template<typename MatrixType>
  inline CompiledAssignmentWrapper<MatrixType>& CompiledAssignmentWrapper<MatrixType>::operator=(CompiledAssignmentWrapper<MatrixType> other)
  {
    pseudoSwap(*this, other);
    build_(this);
    return *this;
  }

  template<typename MatrixType>
  inline CompiledAssignmentWrapper<MatrixType>::~CompiledAssignmentWrapper()
  {
    delete_();
  }

  template<typename MatrixType>
  inline void CompiledAssignmentWrapper<MatrixType>::from(double fromVal)
  {
    from(Eigen::Matrix<double, 1, 1>::Constant(fromVal));
  }

  template<typename MatrixType>
  inline void CompiledAssignmentWrapper<MatrixType>::from(const Eigen::Ref<const MatrixType>& fromVal)
  {
    from_(fromVal);
  }

  template<typename MatrixType>
  template<AssignType A, ScalarMult S, MatrixMult M, MultPos P, typename T>
  inline CompiledAssignmentWrapper<MatrixType> CompiledAssignmentWrapper<MatrixType>::make(const T& from, const Eigen::Ref<MatrixType>& to, double s, const typename MatrixMultBase<M, P>::MultType * const m)
  {
    CompiledAssignmentWrapper<MatrixType> wrapper;
    wrapper.construct<A, S, M, P>(from, to, s, m);
    return wrapper;
  }

  template<typename MatrixType>
  template<AssignType A>
  inline CompiledAssignmentWrapper<MatrixType> CompiledAssignmentWrapper<MatrixType>::make(const Eigen::Ref<MatrixType>& to)
  {
    CompiledAssignmentWrapper<MatrixType> wrapper;
    wrapper.construct<A, NONE, IDENTITY, PRE>(to);
    return wrapper;
  }

  template<typename MatrixType>
  template<AssignType A, ScalarMult S, MatrixMult M, MultPos P>
  inline void CompiledAssignmentWrapper<MatrixType>::construct(const Eigen::Ref<const MatrixType>& from, const Eigen::Ref<MatrixType>& to,
    double s, const typename MatrixMultBase<M, P>::MultType* const m)
  {
    compiledAssignment_ = new CompiledAssignment<MatrixType, A, S, M, P, EXTERNAL>(from, to, s, m);
    constructFunctions<A, S, M, P, EXTERNAL>();
    build_(this);
  }

  template<typename MatrixType>
  template<AssignType A, ScalarMult S, MatrixMult M, MultPos P>
  inline void CompiledAssignmentWrapper<MatrixType>::construct(double from, const Eigen::Ref<MatrixType>& to,
    double s, const typename MatrixMultBase<M, P>::MultType* const m)
  {
    compiledAssignment_ = new CompiledAssignment<MatrixType, A, S, M, P, CONSTANT>(from, to, s, m);
    constructFunctions<A, S, M, P, CONSTANT>();
    build_(this);
  }

  template<typename MatrixType>
  template<AssignType A, ScalarMult S, MatrixMult M, MultPos P>
  inline void CompiledAssignmentWrapper<MatrixType>::construct(const Eigen::Ref<MatrixType>& to)
  {
    compiledAssignment_ = new CompiledAssignment<MatrixType, A, S, M, P, ZERO>(to);
    constructFunctions<A, S, M, P, ZERO>();
    build_(this);
  }

  template<typename MatrixType>
  template<AssignType A, ScalarMult S, MatrixMult M, MultPos P, Source F>
  inline void CompiledAssignmentWrapper<MatrixType>::constructFunctions()
  {
    using CA = CompiledAssignment<MatrixType, A, S, M, P, F>;
    using Wrapper = CompiledAssignmentWrapper<MatrixType>;

    build_ = [](Wrapper* wrapper)
    {
      wrapper->delete_ = [wrapper]()
      {
        delete static_cast<CA*>(wrapper->compiledAssignment_);
      };
      wrapper->copy_ = [wrapper](Wrapper* other)
      {
        other->compiledAssignment_ = static_cast<void*>(new CA(*(static_cast<CA*>(wrapper->compiledAssignment_))));
      };

      wrapper->run = [wrapper]()
      {
        static_cast<CA*>(wrapper->compiledAssignment_)->run();
      };

      wrapper->from_ = helper::from_FunctionBuilder<F>::template construct<CA>(wrapper);

      wrapper->to = [wrapper](const Eigen::Ref<MatrixType>& to)
      {
        static_cast<CA*>(wrapper->compiledAssignment_)->to(to);
      };
    };
  }

  template<typename MatrixType>
  inline void pseudoSwap(CompiledAssignmentWrapper<MatrixType>& first, CompiledAssignmentWrapper<MatrixType>& second)
  {
    using std::swap;
    swap(first.compiledAssignment_, second.compiledAssignment_);
    swap(first.build_, second.build_);

    /* if we want to make it a real swap, we need to ad this:
    if (first.build_) first.build_(first);
    if (second.build_) second.build_(second);

    However, this adds some costs for something we don't really need.
    */
  }

#endif

  /** This class wraps a CompiledAssignment so as to hide the template
    * machinery. The three member functions of CompiledAssignment are
    * exposed: run, from, to.
    */
  template<typename MatrixType>
  class CompiledAssignmentWrapper
  {
  public:
    template<typename T> static void srun(void* ca);
    template<typename T> static void sdelete(void* ca);
    template<typename T> static void sfrom(void* ca, const typename T::SourceType& f);
    template<typename T> static void sto(void* ca, const Eigen::Ref<MatrixType>& from);

    CompiledAssignmentWrapper(const CompiledAssignmentWrapper&) = delete;
    CompiledAssignmentWrapper(CompiledAssignmentWrapper&&) = default;
    ~CompiledAssignmentWrapper() = default;

    /** Run the assignment*/
    void run();
    /** Change the source (for assignement built with Source == CONSTANT)
      * \throw std::runtime_error if used for Source != CONSTANT
      */
    void from(double);
    /** Change the source (for assignement built with Source != CONSTANT)
      * \throw std::runtime_error if used for Source == CONSTANT
      */
    void from(const Eigen::Ref<const MatrixType>& from);
    /** Change the destination of the assignment*/
    void to(const Eigen::Ref<MatrixType>&);

    /** Create an assignement and its wrapper.
      * \param args to [, from] [, weight] [, multiplier] where
      * -\p to is the destination matrix or vector
      * -\p from is the source (nothing for F = ZERO, a double for F = CONSTANT,
      *  and a matrix or vector for F = EXTERNAL)
      * -\p weight is the weight to apply (nothing for W = NONE and W = MINUS, a
      *  double for W = SCALAR, and a vector for W = DIAGONAL or 
      *  W = INVERSE_DIAGONAL)
      * -\p multiplier is the matrix multiplier (nothing for M = IDENTITY, a
      *  matrix for M = GENERAL, and a custom object for M = CUSTOM)
      */
    template<AssignType A, WeightMult W, MatrixMult M, Source F = EXTERNAL, typename... Args>
    static CompiledAssignmentWrapper make(Args&&... args);

  private:
    CompiledAssignmentWrapper(void(*deleter)(void*));

    std::unique_ptr<void, void(*)(void*)> ca_;
    void(*run_)(void*);
    void(*fromd_)(void*, double);
    void(*fromm_)(void*, const Eigen::Ref<const MatrixType>&);
    void(*to_)(void*, const Eigen::Ref<MatrixType>&);
  };


  template<typename MatrixType>
  template<typename T>
  inline void CompiledAssignmentWrapper<MatrixType>::srun(void* ca)
  {
    reinterpret_cast<T*>(ca)->run();
  }

  template<typename MatrixType>
  template<typename T>
  inline void CompiledAssignmentWrapper<MatrixType>::sdelete(void* ca)
  {
    reinterpret_cast<T*>(ca)->~T();
  }

  template<typename MatrixType>
  template<typename T>
  inline void CompiledAssignmentWrapper<MatrixType>::sfrom(void* ca, const typename T::SourceType& f)
  {
    reinterpret_cast<T*>(ca)->from(f);
  }

  template<typename MatrixType>
  template<typename T>
  inline void CompiledAssignmentWrapper<MatrixType>::sto(void* ca, const Eigen::Ref<MatrixType>& t)
  {
    reinterpret_cast<T*>(ca)->to(t);
  }

  template<typename MatrixType>
  template<AssignType A, WeightMult W, MatrixMult M, Source F, typename ...Args>
  inline CompiledAssignmentWrapper<MatrixType> CompiledAssignmentWrapper<MatrixType>::make(Args &&... args)
  {
    using CA = CompiledAssignment<MatrixType, A, W, M, F>;
    CompiledAssignmentWrapper<MatrixType> w(sdelete<CA>);
    w.run_ = srun<CA>;
    w.fromd_ = F==CONSTANT?reinterpret_cast<void(*)(void*, double)>(&sfrom<CA>):nullptr;
    w.fromm_ = F!=CONSTANT?reinterpret_cast<void(*)(void*, const Eigen::Ref<const MatrixType>&)>(&sfrom<CA>) :nullptr;
    w.to_ = sto<CA>;
    w.ca_.reset(new CA(std::forward<Args>(args)...));
    return w;
  }

  template<typename MatrixType>
  inline CompiledAssignmentWrapper<MatrixType>::CompiledAssignmentWrapper(void(*deleter)(void*))
    : ca_(nullptr, deleter)
    , run_(nullptr)
    , fromd_(nullptr)
    , fromm_(nullptr)
    , to_(nullptr)
  {
  }

  template<typename MatrixType>
  inline void CompiledAssignmentWrapper<MatrixType>::run()
  {
    run_(ca_.get());
  }

  template<typename MatrixType>
  inline void CompiledAssignmentWrapper<MatrixType>::from(double d)
  {
    if (fromd_)
    {
      fromd_(ca_.get(), d);
    }
    else
    {
      throw std::runtime_error("Method from(double) is invalid for this assignment, try from(const Eigen::Ref<const MatrixType>&) instead");
    }
  }

  template<typename MatrixType>
  inline void CompiledAssignmentWrapper<MatrixType>::from(const Eigen::Ref<const MatrixType>& f)
  {
    if (fromm_)
    {
      fromm_(ca_.get(), f);
    }
    else
    {
      throw std::runtime_error("Method from(const Eigen::Ref<const MatrixType>&) is invalid for this assignment, try from(double) instead");
    }
  }

  template<typename MatrixType>
  inline void CompiledAssignmentWrapper<MatrixType>::to(const Eigen::Ref<MatrixType>& t)
  {
    to_(ca_.get(), t);
  }

}  // namespace internal

}  // namespace scheme

}  // namespace tvm
