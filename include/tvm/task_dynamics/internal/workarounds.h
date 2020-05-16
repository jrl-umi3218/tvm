/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <Eigen/Core>

// Forward declarations
namespace tvm::task_dynamics::internal
{
  class WorkaroundDoubleVectorMatrixVariant;
}

namespace std
{
  template<typename T>
  const T& get(const tvm::task_dynamics::internal::WorkaroundDoubleVectorMatrixVariant& g);
}

namespace tvm::task_dynamics::internal
{
  /** A minimal workaround for std::variant<double, VectorXd, MatrixXd>. */
  class WorkaroundDoubleVectorMatrixVariant
  {
  public:
    WorkaroundDoubleVectorMatrixVariant(double d) : index_(0), d_(d), v_(0), m_(0, 0) {}
    WorkaroundDoubleVectorMatrixVariant(std::in_place_index_t<0>, double d) : index_(0), d_(d), v_(0), m_(0, 0) {}
    WorkaroundDoubleVectorMatrixVariant(const Eigen::VectorXd& v) : index_(1), d_(0), v_(v), m_(0, 0) {}
    WorkaroundDoubleVectorMatrixVariant(std::in_place_index_t<1>, const Eigen::VectorXd& v) : index_(1), d_(0), v_(v), m_(0, 0) {}
    WorkaroundDoubleVectorMatrixVariant(const Eigen::MatrixXd& m) : index_(2), d_(0), v_(0), m_(m) {}
    WorkaroundDoubleVectorMatrixVariant(std::in_place_index_t<2>, const Eigen::MatrixXd& m) : index_(2), d_(0), v_(0), m_(m) {}

    /** Return the index of the active type. 0: double, 1: VectorXd, 2: MatrixXd*/
    int index() const { return index_; }

    template<int Index, typename T>
    void emplace(const T& e)
    {
      static_assert(Index >= 0 && Index <= 2);
      if constexpr (Index == 0) d_ = e;
      else if constexpr (Index == 1) v_ = e;
      else m_ = e;
      index_ = Index;
    }

  private:
    int index_;
    double d_;
    Eigen::VectorXd v_;
    Eigen::MatrixXd m_;

    template<typename T>
    friend const T& ::std::get(const WorkaroundDoubleVectorMatrixVariant& g);
  };
}

namespace std
{
  template<> 
  inline const double& get<double>(const tvm::task_dynamics::internal::WorkaroundDoubleVectorMatrixVariant& g) 
  {
    assert(g.index() == 0);
    return g.d_; 
  }

  template<> 
  inline const Eigen::VectorXd& get<Eigen::VectorXd>(const tvm::task_dynamics::internal::WorkaroundDoubleVectorMatrixVariant& g) 
  { 
    assert(g.index() == 1);
    return g.v_;
  }

  template<> 
  inline const Eigen::MatrixXd& get<Eigen::MatrixXd>(const tvm::task_dynamics::internal::WorkaroundDoubleVectorMatrixVariant& g) 
  { 
    assert(g.index() == 2);
    return g.m_;
  }
}