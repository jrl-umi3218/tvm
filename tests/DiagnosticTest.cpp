/** Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/diagnostic/matrix.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace tvm::diagnostic;
using namespace Eigen;

// test that \c isInMatrix applied to S returns the correct value for all elements of M.
template<typename T, typename U>
void testIsInMatrix(T & M, U & S)
{
  M.setZero();
  S.setOnes();
  for(int i = 0; i < M.rows(); ++i)
  {
    for(int j = 0; j < M.cols(); ++j)
    {
      FAST_CHECK_EQ(isInMatrix(M, i, j, S), M(i, j) == 1);
    }
  }
}

TEST_CASE("isInMatrix")
{
  int m = 12, n = 10;
  MatrixXd M(m, n);

  auto B = M.block(2, 3, 7, 6);
  testIsInMatrix(M, B);

  auto C = B.block(1, 1, 3, 3);
  testIsInMatrix(B, C);

  auto D = Map<MatrixXd, 0, Stride<-1, -1>>(M.data() + 1, 4, 5, Stride<-1, -1>(24, 3));
  testIsInMatrix(M, D);

  Matrix3d N;
  testIsInMatrix(M, N);
}