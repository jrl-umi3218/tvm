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
  /** This class wraps a CompiledAssignment so as to hide the template
    * machinery. The three member functions of CompiledAssignment are
    * exposed: run, from, to.
    *
    * \internal Internally we hold a void* pointer to the CompiledAssignment.
    * We also hold pointer to functions that are templated by the actual type of
    * the assignment. Those functions allow casting back the void* to the true
    * type of the assignment.
    */
  template<typename MatrixType>
  class CompiledAssignmentWrapper
  {
  public:
    CompiledAssignmentWrapper(const CompiledAssignmentWrapper&);
    CompiledAssignmentWrapper(CompiledAssignmentWrapper&&) = default;
    ~CompiledAssignmentWrapper() = default;

    CompiledAssignmentWrapper& operator=(const CompiledAssignmentWrapper&);
    CompiledAssignmentWrapper& operator=(CompiledAssignmentWrapper&&) = default;

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
      * - \p to is the destination matrix or vector
      * - \p from is the source (nothing for F = ZERO, a double for F = CONSTANT,
      *   and a matrix or vector for F = EXTERNAL)
      * - \p weight is the weight to apply (nothing for W = NONE and W = MINUS, a
      *   double for W = SCALAR, and a vector for W = DIAGONAL or 
      *   W = INVERSE_DIAGONAL)
      * - \p multiplier is the matrix multiplier (nothing for M = IDENTITY, a
      *   matrix for M = GENERAL, and a custom object for M = CUSTOM)
      */
    template<AssignType A, WeightMult W, MatrixMult M, Source F = EXTERNAL, typename... Args>
    static CompiledAssignmentWrapper make(Args&&... args);

  private:
    CompiledAssignmentWrapper(void(*deleter)(void*));

    /** call ca->run*/
    template<typename T> static void srun(void* ca);
    /** call the destructor of ca*/
    template<typename T> static void sdelete(void* ca);
    /** call ca->from*/
    template<typename T> static void sfrom(void* ca, const typename T::SourceType& f);
    /** call ca->to*/
    template<typename T> static void sto(void* ca, const Eigen::Ref<MatrixType>& from);
    /** return a copy of ca*/
    template<typename T> static void* sclone(void* ca);

    /** A pointer to CompiledAssignment whose type has been erased.*/
    std::unique_ptr<void, void(*)(void*)> ca_;
    /** Pointer to srun<T> where T is the actual type of the CompiledAssignment.*/
    void(*run_)(void*);
    /** Pointer to sfrom<T> if the assignment has a CONSTANT Source, a null
      * pointer otherwise. T is the actual type of the CompileAssignment.
      */
    void(*fromd_)(void*, const double&);
    /** Pointer to sfrom<T> if the assignment has not a CONSTANT Source, a null
      * pointer otherwise. T is the actual type of the CompileAssignment.
      */
    void(*fromm_)(void*, const Eigen::Ref<const MatrixType>&);
    /** Pointer to srun<T> where T is the actual type of the CompiledAssignment.*/
    void(*to_)(void*, const Eigen::Ref<MatrixType>&);
    /** Pointer to sclone<T> where T is the actual type of the CompiledAssignment.*/
    void*(*clone_)(void*);
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
  template<typename T>
  inline void* CompiledAssignmentWrapper<MatrixType>::sclone(void* ca)
  {
    return new T(*reinterpret_cast<T*>(ca));
  }

  template<typename MatrixType>
  template<AssignType A, WeightMult W, MatrixMult M, Source F, typename ...Args>
  inline CompiledAssignmentWrapper<MatrixType> CompiledAssignmentWrapper<MatrixType>::make(Args &&... args)
  {
    using CA = CompiledAssignment<MatrixType, A, W, M, F>;
    CompiledAssignmentWrapper<MatrixType> w(sdelete<CA>);
    w.run_ = srun<CA>;
    w.fromd_ = F==CONSTANT?reinterpret_cast<void(*)(void*, const double&)>(&sfrom<CA>):nullptr;
    w.fromm_ = F!=CONSTANT?reinterpret_cast<void(*)(void*, const Eigen::Ref<const MatrixType>&)>(&sfrom<CA>) :nullptr;
    w.to_ = sto<CA>;
    w.clone_ = sclone<CA>;
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
    , clone_(nullptr)
  {
  }

  template<typename MatrixType>
  inline CompiledAssignmentWrapper<MatrixType>::CompiledAssignmentWrapper(const CompiledAssignmentWrapper& other)
    : ca_(other.clone_(other.ca_.get()), other.ca_.get_deleter())
    , run_(other.run_)
    , fromd_(other.fromd_)
    , fromm_(other.fromm_)
    , to_(other.to_)
    , clone_(other.clone_)
  {
  }

  template<typename MatrixType>
  inline CompiledAssignmentWrapper<MatrixType>& CompiledAssignmentWrapper<MatrixType>::operator=(const CompiledAssignmentWrapper& other)
  {
    ca_.reset(other.clone_(other.ca_.get()));
    ca_.get_deleter() = other.ca_.get_deleter();
    run_ = other.run_;
    fromd_ = other.fromd_;
    fromm_ = other.fromm_;
    to_ = other.to_;
    clone_ = other.clone_;

    return *this;
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
