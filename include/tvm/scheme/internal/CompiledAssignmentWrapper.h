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

#include <functional>

namespace tvm
{

namespace scheme
{

namespace internal
{

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
    typedef CompiledAssignment<MatrixType, A, S, M, P, F> CA;
    typedef CompiledAssignmentWrapper<MatrixType> Wrapper;

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

}  // namespace internal

}  // namespace scheme

}  // namespace tvm
