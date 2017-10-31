#include "MatrixProperties.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace tvm;

TEST_CASE("Test shape properties")
{
  MatrixProperties p0;
  FAST_CHECK_EQ(p0.shape(), MatrixProperties::GENERAL);
  FAST_CHECK_EQ(p0.positiveness(), MatrixProperties::NA);
  FAST_CHECK_UNARY_FALSE(p0.isConstant());
  FAST_CHECK_UNARY_FALSE(p0.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p0.isIdentity());
  FAST_CHECK_UNARY_FALSE(p0.isInvertible());
  FAST_CHECK_UNARY_FALSE(p0.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p0.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p0.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p0.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p0.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p0.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p0.isPositiveSemiDefinite());
  FAST_CHECK_UNARY_FALSE(p0.isSymmetric());
  FAST_CHECK_UNARY_FALSE(p0.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p0.isZero());
  FAST_CHECK_UNARY_FALSE(p0.isTriangular());
  FAST_CHECK_UNARY_FALSE(p0.isLowerTriangular());
  FAST_CHECK_UNARY_FALSE(p0.isUpperTriangular());

  MatrixProperties p1(MatrixProperties::LOWER_TRIANGULAR);
  FAST_CHECK_EQ(p1.shape(), MatrixProperties::LOWER_TRIANGULAR);
  FAST_CHECK_EQ(p1.positiveness(), MatrixProperties::NA);
  FAST_CHECK_UNARY_FALSE(p1.isConstant());
  FAST_CHECK_UNARY_FALSE(p1.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p1.isIdentity());
  FAST_CHECK_UNARY_FALSE(p1.isInvertible());
  FAST_CHECK_UNARY_FALSE(p1.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p1.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p1.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p1.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p1.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p1.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p1.isPositiveSemiDefinite());
  FAST_CHECK_UNARY_FALSE(p1.isSymmetric());
  FAST_CHECK_UNARY_FALSE(p1.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p1.isZero());
  FAST_CHECK_UNARY(p1.isTriangular());
  FAST_CHECK_UNARY(p1.isLowerTriangular());
  FAST_CHECK_UNARY_FALSE(p1.isUpperTriangular());

  MatrixProperties p2(MatrixProperties::UPPER_TRIANGULAR);
  FAST_CHECK_EQ(p2.shape(), MatrixProperties::UPPER_TRIANGULAR);
  FAST_CHECK_EQ(p2.positiveness(), MatrixProperties::NA);
  FAST_CHECK_UNARY_FALSE(p2.isConstant());
  FAST_CHECK_UNARY_FALSE(p2.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p2.isIdentity());
  FAST_CHECK_UNARY_FALSE(p2.isInvertible());
  FAST_CHECK_UNARY_FALSE(p2.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p2.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p2.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p2.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p2.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p2.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p2.isPositiveSemiDefinite());
  FAST_CHECK_UNARY_FALSE(p2.isSymmetric());
  FAST_CHECK_UNARY_FALSE(p2.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p2.isZero());
  FAST_CHECK_UNARY(p2.isTriangular());
  FAST_CHECK_UNARY_FALSE(p2.isLowerTriangular());
  FAST_CHECK_UNARY(p2.isUpperTriangular());

  MatrixProperties p3(MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p3.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p3.positiveness(), MatrixProperties::INDEFINITE);
  FAST_CHECK_UNARY_FALSE(p3.isConstant());
  FAST_CHECK_UNARY(p3.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p3.isIdentity());
  FAST_CHECK_UNARY_FALSE(p3.isInvertible());
  FAST_CHECK_UNARY_FALSE(p3.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p3.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p3.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p3.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p3.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p3.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p3.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p3.isSymmetric());
  FAST_CHECK_UNARY(p3.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p3.isZero());
  FAST_CHECK_UNARY(p3.isTriangular());
  FAST_CHECK_UNARY(p3.isLowerTriangular());
  FAST_CHECK_UNARY(p3.isUpperTriangular());

  MatrixProperties p4(MatrixProperties::MULTIPLE_OF_IDENTITY);
  FAST_CHECK_EQ(p4.shape(), MatrixProperties::MULTIPLE_OF_IDENTITY);
  FAST_CHECK_EQ(p4.positiveness(), MatrixProperties::INDEFINITE);
  FAST_CHECK_UNARY_FALSE(p4.isConstant());
  FAST_CHECK_UNARY(p4.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p4.isIdentity());
  FAST_CHECK_UNARY_FALSE(p4.isInvertible());
  FAST_CHECK_UNARY_FALSE(p4.isMinusIdentity());
  FAST_CHECK_UNARY(p4.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p4.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p4.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p4.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p4.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p4.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p4.isSymmetric());
  FAST_CHECK_UNARY(p4.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p4.isZero());
  FAST_CHECK_UNARY(p4.isTriangular());
  FAST_CHECK_UNARY(p4.isLowerTriangular());
  FAST_CHECK_UNARY(p4.isUpperTriangular());

  MatrixProperties p5(MatrixProperties::IDENTITY);
  FAST_CHECK_EQ(p5.shape(), MatrixProperties::IDENTITY);
  FAST_CHECK_EQ(p5.positiveness(), MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_UNARY(p5.isConstant());
  FAST_CHECK_UNARY(p5.isDiagonal());
  FAST_CHECK_UNARY(p5.isIdentity());
  FAST_CHECK_UNARY(p5.isInvertible());
  FAST_CHECK_UNARY_FALSE(p5.isMinusIdentity());
  FAST_CHECK_UNARY(p5.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p5.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p5.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p5.isNonZeroIndefinite());
  FAST_CHECK_UNARY(p5.isPositiveDefinite());
  FAST_CHECK_UNARY(p5.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p5.isSymmetric());
  FAST_CHECK_UNARY(p5.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p5.isZero());
  FAST_CHECK_UNARY(p5.isTriangular());
  FAST_CHECK_UNARY(p5.isLowerTriangular());
  FAST_CHECK_UNARY(p5.isUpperTriangular());

  MatrixProperties p6(MatrixProperties::MINUS_IDENTITY);
  FAST_CHECK_EQ(p6.shape(), MatrixProperties::MINUS_IDENTITY);
  FAST_CHECK_EQ(p6.positiveness(), MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_UNARY(p6.isConstant());
  FAST_CHECK_UNARY(p6.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p6.isIdentity());
  FAST_CHECK_UNARY(p6.isInvertible());
  FAST_CHECK_UNARY(p6.isMinusIdentity());
  FAST_CHECK_UNARY(p6.isMultipleOfIdentity());
  FAST_CHECK_UNARY(p6.isNegativeDefinite());
  FAST_CHECK_UNARY(p6.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p6.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p6.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p6.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p6.isSymmetric());
  FAST_CHECK_UNARY(p6.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p6.isZero());
  FAST_CHECK_UNARY(p6.isTriangular());
  FAST_CHECK_UNARY(p6.isLowerTriangular());
  FAST_CHECK_UNARY(p6.isUpperTriangular());

  MatrixProperties p7(MatrixProperties::ZERO);
  FAST_CHECK_EQ(p7.shape(), MatrixProperties::ZERO);
  FAST_CHECK_EQ(p7.positiveness(), MatrixProperties::INDEFINITE);
  FAST_CHECK_UNARY(p7.isConstant());
  FAST_CHECK_UNARY(p7.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p7.isIdentity());
  FAST_CHECK_UNARY_FALSE(p7.isInvertible());
  FAST_CHECK_UNARY_FALSE(p7.isMinusIdentity());
  FAST_CHECK_UNARY(p7.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p7.isNegativeDefinite());
  FAST_CHECK_UNARY(p7.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p7.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p7.isPositiveDefinite());
  FAST_CHECK_UNARY(p7.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p7.isSymmetric());
  FAST_CHECK_UNARY(p7.isIndefinite());
  FAST_CHECK_UNARY(p7.isZero());
  FAST_CHECK_UNARY(p7.isTriangular());
  FAST_CHECK_UNARY(p7.isLowerTriangular());
  FAST_CHECK_UNARY(p7.isUpperTriangular());
}

TEST_CASE("Test properties deductions")
{
  MatrixProperties p01(MatrixProperties::GENERAL, MatrixProperties::POSITIVE_SEMIDEFINITE);
  FAST_CHECK_EQ(p01.shape(), MatrixProperties::GENERAL);
  FAST_CHECK_EQ(p01.positiveness(), MatrixProperties::POSITIVE_SEMIDEFINITE);
  FAST_CHECK_UNARY_FALSE(p01.isConstant());
  FAST_CHECK_UNARY_FALSE(p01.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p01.isIdentity());
  FAST_CHECK_UNARY_FALSE(p01.isInvertible());
  FAST_CHECK_UNARY_FALSE(p01.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p01.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p01.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p01.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p01.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p01.isPositiveDefinite());
  FAST_CHECK_UNARY(p01.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p01.isSymmetric());
  FAST_CHECK_UNARY(p01.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p01.isZero());
  FAST_CHECK_UNARY_FALSE(p01.isTriangular());
  FAST_CHECK_UNARY_FALSE(p01.isLowerTriangular());
  FAST_CHECK_UNARY_FALSE(p01.isUpperTriangular());

  MatrixProperties p02(MatrixProperties::GENERAL, MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_EQ(p02.shape(), MatrixProperties::GENERAL);
  FAST_CHECK_EQ(p02.positiveness(), MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_UNARY_FALSE(p02.isConstant());
  FAST_CHECK_UNARY_FALSE(p02.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p02.isIdentity());
  FAST_CHECK_UNARY(p02.isInvertible());
  FAST_CHECK_UNARY_FALSE(p02.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p02.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p02.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p02.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p02.isNonZeroIndefinite());
  FAST_CHECK_UNARY(p02.isPositiveDefinite());
  FAST_CHECK_UNARY(p02.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p02.isSymmetric());
  FAST_CHECK_UNARY(p02.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p02.isZero());
  FAST_CHECK_UNARY_FALSE(p02.isTriangular());
  FAST_CHECK_UNARY_FALSE(p02.isLowerTriangular());
  FAST_CHECK_UNARY_FALSE(p02.isUpperTriangular());

  MatrixProperties p03(MatrixProperties::GENERAL, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  FAST_CHECK_EQ(p03.shape(), MatrixProperties::GENERAL);
  FAST_CHECK_EQ(p03.positiveness(), MatrixProperties::NEGATIVE_SEMIDEFINITE);
  FAST_CHECK_UNARY_FALSE(p03.isConstant());
  FAST_CHECK_UNARY_FALSE(p03.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p03.isIdentity());
  FAST_CHECK_UNARY_FALSE(p03.isInvertible());
  FAST_CHECK_UNARY_FALSE(p03.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p03.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p03.isNegativeDefinite());
  FAST_CHECK_UNARY(p03.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p03.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p03.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p03.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p03.isSymmetric());
  FAST_CHECK_UNARY(p03.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p03.isZero());
  FAST_CHECK_UNARY_FALSE(p03.isTriangular());
  FAST_CHECK_UNARY_FALSE(p03.isLowerTriangular());
  FAST_CHECK_UNARY_FALSE(p03.isUpperTriangular());

  MatrixProperties p04(MatrixProperties::GENERAL, MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_EQ(p04.shape(), MatrixProperties::GENERAL);
  FAST_CHECK_EQ(p04.positiveness(), MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_UNARY_FALSE(p04.isConstant());
  FAST_CHECK_UNARY_FALSE(p04.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p04.isIdentity());
  FAST_CHECK_UNARY(p04.isInvertible());
  FAST_CHECK_UNARY_FALSE(p04.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p04.isMultipleOfIdentity());
  FAST_CHECK_UNARY(p04.isNegativeDefinite());
  FAST_CHECK_UNARY(p04.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p04.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p04.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p04.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p04.isSymmetric());
  FAST_CHECK_UNARY(p04.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p04.isZero());
  FAST_CHECK_UNARY_FALSE(p04.isTriangular());
  FAST_CHECK_UNARY_FALSE(p04.isLowerTriangular());
  FAST_CHECK_UNARY_FALSE(p04.isUpperTriangular());

  MatrixProperties p05(MatrixProperties::GENERAL, MatrixProperties::INDEFINITE);
  FAST_CHECK_EQ(p05.shape(), MatrixProperties::GENERAL);
  FAST_CHECK_EQ(p05.positiveness(), MatrixProperties::INDEFINITE);
  FAST_CHECK_UNARY_FALSE(p05.isConstant());
  FAST_CHECK_UNARY_FALSE(p05.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p05.isIdentity());
  FAST_CHECK_UNARY_FALSE(p05.isInvertible());
  FAST_CHECK_UNARY_FALSE(p05.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p05.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p05.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p05.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p05.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p05.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p05.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p05.isSymmetric());
  FAST_CHECK_UNARY(p05.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p05.isZero());
  FAST_CHECK_UNARY_FALSE(p05.isTriangular());
  FAST_CHECK_UNARY_FALSE(p05.isLowerTriangular());
  FAST_CHECK_UNARY_FALSE(p05.isUpperTriangular());

  MatrixProperties p06(MatrixProperties::GENERAL, MatrixProperties::NON_ZERO_INDEFINITE);
  FAST_CHECK_EQ(p06.shape(), MatrixProperties::GENERAL);
  FAST_CHECK_EQ(p06.positiveness(), MatrixProperties::NON_ZERO_INDEFINITE);
  FAST_CHECK_UNARY_FALSE(p06.isConstant());
  FAST_CHECK_UNARY_FALSE(p06.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p06.isIdentity());
  FAST_CHECK_UNARY(p06.isInvertible());
  FAST_CHECK_UNARY_FALSE(p06.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p06.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p06.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p06.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p06.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p06.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p06.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p06.isSymmetric());
  FAST_CHECK_UNARY(p06.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p06.isZero());
  FAST_CHECK_UNARY_FALSE(p06.isTriangular());
  FAST_CHECK_UNARY_FALSE(p06.isLowerTriangular());
  FAST_CHECK_UNARY_FALSE(p06.isUpperTriangular());

  MatrixProperties p11(MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::POSITIVE_SEMIDEFINITE);
  FAST_CHECK_EQ(p11.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p11.positiveness(), MatrixProperties::POSITIVE_SEMIDEFINITE);
  FAST_CHECK_UNARY_FALSE(p11.isConstant());
  FAST_CHECK_UNARY(p11.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p11.isIdentity());
  FAST_CHECK_UNARY_FALSE(p11.isInvertible());
  FAST_CHECK_UNARY_FALSE(p11.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p11.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p11.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p11.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p11.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p11.isPositiveDefinite());
  FAST_CHECK_UNARY(p11.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p11.isSymmetric());
  FAST_CHECK_UNARY(p11.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p11.isZero());
  FAST_CHECK_UNARY(p11.isTriangular());
  FAST_CHECK_UNARY(p11.isLowerTriangular());
  FAST_CHECK_UNARY(p11.isUpperTriangular());

  MatrixProperties p12(MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_EQ(p12.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p12.positiveness(), MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_UNARY_FALSE(p12.isConstant());
  FAST_CHECK_UNARY(p12.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p12.isIdentity());
  FAST_CHECK_UNARY(p12.isInvertible());
  FAST_CHECK_UNARY_FALSE(p12.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p12.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p12.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p12.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p12.isNonZeroIndefinite());
  FAST_CHECK_UNARY(p12.isPositiveDefinite());
  FAST_CHECK_UNARY(p12.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p12.isSymmetric());
  FAST_CHECK_UNARY(p12.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p12.isZero());
  FAST_CHECK_UNARY(p12.isTriangular());
  FAST_CHECK_UNARY(p12.isLowerTriangular());
  FAST_CHECK_UNARY(p12.isUpperTriangular());

  MatrixProperties p13(MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  FAST_CHECK_EQ(p13.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p13.positiveness(), MatrixProperties::NEGATIVE_SEMIDEFINITE);
  FAST_CHECK_UNARY_FALSE(p13.isConstant());
  FAST_CHECK_UNARY(p13.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p13.isIdentity());
  FAST_CHECK_UNARY_FALSE(p13.isInvertible());
  FAST_CHECK_UNARY_FALSE(p13.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p13.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p13.isNegativeDefinite());
  FAST_CHECK_UNARY(p13.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p13.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p13.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p13.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p13.isSymmetric());
  FAST_CHECK_UNARY(p13.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p13.isZero());
  FAST_CHECK_UNARY(p13.isTriangular());
  FAST_CHECK_UNARY(p13.isLowerTriangular());
  FAST_CHECK_UNARY(p13.isUpperTriangular());

  MatrixProperties p14(MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_EQ(p14.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p14.positiveness(), MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_UNARY_FALSE(p14.isConstant());
  FAST_CHECK_UNARY(p14.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p14.isIdentity());
  FAST_CHECK_UNARY(p14.isInvertible());
  FAST_CHECK_UNARY_FALSE(p14.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p14.isMultipleOfIdentity());
  FAST_CHECK_UNARY(p14.isNegativeDefinite());
  FAST_CHECK_UNARY(p14.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p14.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p14.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p14.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p14.isSymmetric());
  FAST_CHECK_UNARY(p14.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p14.isZero());
  FAST_CHECK_UNARY(p14.isTriangular());
  FAST_CHECK_UNARY(p14.isLowerTriangular());
  FAST_CHECK_UNARY(p14.isUpperTriangular());

  MatrixProperties p15(MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::INDEFINITE);
  FAST_CHECK_EQ(p15.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p15.positiveness(), MatrixProperties::INDEFINITE);
  FAST_CHECK_UNARY_FALSE(p15.isConstant());
  FAST_CHECK_UNARY(p15.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p15.isIdentity());
  FAST_CHECK_UNARY_FALSE(p15.isInvertible());
  FAST_CHECK_UNARY_FALSE(p15.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p15.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p15.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p15.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p15.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p15.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p15.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p15.isSymmetric());
  FAST_CHECK_UNARY(p15.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p15.isZero());
  FAST_CHECK_UNARY(p15.isTriangular());
  FAST_CHECK_UNARY(p15.isLowerTriangular());
  FAST_CHECK_UNARY(p15.isUpperTriangular());

  MatrixProperties p16(MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NON_ZERO_INDEFINITE);
  FAST_CHECK_EQ(p16.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p16.positiveness(), MatrixProperties::NON_ZERO_INDEFINITE);
  FAST_CHECK_UNARY_FALSE(p16.isConstant());
  FAST_CHECK_UNARY(p16.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p16.isIdentity());
  FAST_CHECK_UNARY(p16.isInvertible());
  FAST_CHECK_UNARY_FALSE(p16.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p16.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p16.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p16.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p16.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p16.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p16.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p16.isSymmetric());
  FAST_CHECK_UNARY(p16.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p16.isZero());
  FAST_CHECK_UNARY(p16.isTriangular());
  FAST_CHECK_UNARY(p16.isLowerTriangular());
  FAST_CHECK_UNARY(p16.isUpperTriangular());


  MatrixProperties p21(MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::POSITIVE_SEMIDEFINITE);
  FAST_CHECK_EQ(p21.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p21.positiveness(), MatrixProperties::POSITIVE_SEMIDEFINITE);
  FAST_CHECK_UNARY_FALSE(p21.isConstant());
  FAST_CHECK_UNARY(p21.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p21.isIdentity());
  FAST_CHECK_UNARY_FALSE(p21.isInvertible());
  FAST_CHECK_UNARY_FALSE(p21.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p21.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p21.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p21.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p21.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p21.isPositiveDefinite());
  FAST_CHECK_UNARY(p21.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p21.isSymmetric());
  FAST_CHECK_UNARY(p21.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p21.isZero());
  FAST_CHECK_UNARY(p21.isTriangular());
  FAST_CHECK_UNARY(p21.isLowerTriangular());
  FAST_CHECK_UNARY(p21.isUpperTriangular());

  MatrixProperties p22(MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_EQ(p22.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p22.positiveness(), MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_UNARY_FALSE(p22.isConstant());
  FAST_CHECK_UNARY(p22.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p22.isIdentity());
  FAST_CHECK_UNARY(p22.isInvertible());
  FAST_CHECK_UNARY_FALSE(p22.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p22.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p22.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p22.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p22.isNonZeroIndefinite());
  FAST_CHECK_UNARY(p22.isPositiveDefinite());
  FAST_CHECK_UNARY(p22.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p22.isSymmetric());
  FAST_CHECK_UNARY(p22.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p22.isZero());
  FAST_CHECK_UNARY(p22.isTriangular());
  FAST_CHECK_UNARY(p22.isLowerTriangular());
  FAST_CHECK_UNARY(p22.isUpperTriangular());

  MatrixProperties p23(MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  FAST_CHECK_EQ(p23.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p23.positiveness(), MatrixProperties::NEGATIVE_SEMIDEFINITE);
  FAST_CHECK_UNARY_FALSE(p23.isConstant());
  FAST_CHECK_UNARY(p23.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p23.isIdentity());
  FAST_CHECK_UNARY_FALSE(p23.isInvertible());
  FAST_CHECK_UNARY_FALSE(p23.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p23.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p23.isNegativeDefinite());
  FAST_CHECK_UNARY(p23.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p23.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p23.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p23.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p23.isSymmetric());
  FAST_CHECK_UNARY(p23.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p23.isZero());
  FAST_CHECK_UNARY(p23.isTriangular());
  FAST_CHECK_UNARY(p23.isLowerTriangular());
  FAST_CHECK_UNARY(p23.isUpperTriangular());

  MatrixProperties p24(MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_EQ(p24.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p24.positiveness(), MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_UNARY_FALSE(p24.isConstant());
  FAST_CHECK_UNARY(p24.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p24.isIdentity());
  FAST_CHECK_UNARY(p24.isInvertible());
  FAST_CHECK_UNARY_FALSE(p24.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p24.isMultipleOfIdentity());
  FAST_CHECK_UNARY(p24.isNegativeDefinite());
  FAST_CHECK_UNARY(p24.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p24.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p24.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p24.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p24.isSymmetric());
  FAST_CHECK_UNARY(p24.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p24.isZero());
  FAST_CHECK_UNARY(p24.isTriangular());
  FAST_CHECK_UNARY(p24.isLowerTriangular());
  FAST_CHECK_UNARY(p24.isUpperTriangular());

  MatrixProperties p25(MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::INDEFINITE);
  FAST_CHECK_EQ(p25.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p25.positiveness(), MatrixProperties::INDEFINITE);
  FAST_CHECK_UNARY_FALSE(p25.isConstant());
  FAST_CHECK_UNARY(p25.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p25.isIdentity());
  FAST_CHECK_UNARY_FALSE(p25.isInvertible());
  FAST_CHECK_UNARY_FALSE(p25.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p25.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p25.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p25.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p25.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p25.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p25.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p25.isSymmetric());
  FAST_CHECK_UNARY(p25.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p25.isZero());
  FAST_CHECK_UNARY(p25.isTriangular());
  FAST_CHECK_UNARY(p25.isLowerTriangular());
  FAST_CHECK_UNARY(p25.isUpperTriangular());

  MatrixProperties p26(MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NON_ZERO_INDEFINITE);
  FAST_CHECK_EQ(p26.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p26.positiveness(), MatrixProperties::NON_ZERO_INDEFINITE);
  FAST_CHECK_UNARY_FALSE(p26.isConstant());
  FAST_CHECK_UNARY(p26.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p26.isIdentity());
  FAST_CHECK_UNARY(p26.isInvertible());
  FAST_CHECK_UNARY_FALSE(p26.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p26.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p26.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p26.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p26.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p26.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p26.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p26.isSymmetric());
  FAST_CHECK_UNARY(p26.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p26.isZero());
  FAST_CHECK_UNARY(p26.isTriangular());
  FAST_CHECK_UNARY(p26.isLowerTriangular());
  FAST_CHECK_UNARY(p26.isUpperTriangular());


  MatrixProperties p31(MatrixProperties::DIAGONAL, MatrixProperties::POSITIVE_SEMIDEFINITE);
  FAST_CHECK_EQ(p31.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p31.positiveness(), MatrixProperties::POSITIVE_SEMIDEFINITE);
  FAST_CHECK_UNARY_FALSE(p31.isConstant());
  FAST_CHECK_UNARY(p31.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p31.isIdentity());
  FAST_CHECK_UNARY_FALSE(p31.isInvertible());
  FAST_CHECK_UNARY_FALSE(p31.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p31.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p31.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p31.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p31.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p31.isPositiveDefinite());
  FAST_CHECK_UNARY(p31.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p31.isSymmetric());
  FAST_CHECK_UNARY(p31.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p31.isZero());
  FAST_CHECK_UNARY(p31.isTriangular());
  FAST_CHECK_UNARY(p31.isLowerTriangular());
  FAST_CHECK_UNARY(p31.isUpperTriangular());

  MatrixProperties p32(MatrixProperties::DIAGONAL, MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_EQ(p32.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p32.positiveness(), MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_UNARY_FALSE(p32.isConstant());
  FAST_CHECK_UNARY(p32.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p32.isIdentity());
  FAST_CHECK_UNARY(p32.isInvertible());
  FAST_CHECK_UNARY_FALSE(p32.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p32.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p32.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p32.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p32.isNonZeroIndefinite());
  FAST_CHECK_UNARY(p32.isPositiveDefinite());
  FAST_CHECK_UNARY(p32.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p32.isSymmetric());
  FAST_CHECK_UNARY(p32.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p32.isZero());
  FAST_CHECK_UNARY(p32.isTriangular());
  FAST_CHECK_UNARY(p32.isLowerTriangular());
  FAST_CHECK_UNARY(p32.isUpperTriangular());

  MatrixProperties p33(MatrixProperties::DIAGONAL, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  FAST_CHECK_EQ(p33.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p33.positiveness(), MatrixProperties::NEGATIVE_SEMIDEFINITE);
  FAST_CHECK_UNARY_FALSE(p33.isConstant());
  FAST_CHECK_UNARY(p33.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p33.isIdentity());
  FAST_CHECK_UNARY_FALSE(p33.isInvertible());
  FAST_CHECK_UNARY_FALSE(p33.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p33.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p33.isNegativeDefinite());
  FAST_CHECK_UNARY(p33.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p33.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p33.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p33.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p33.isSymmetric());
  FAST_CHECK_UNARY(p33.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p33.isZero());
  FAST_CHECK_UNARY(p33.isTriangular());
  FAST_CHECK_UNARY(p33.isLowerTriangular());
  FAST_CHECK_UNARY(p33.isUpperTriangular());

  MatrixProperties p34(MatrixProperties::DIAGONAL, MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_EQ(p34.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p34.positiveness(), MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_UNARY_FALSE(p34.isConstant());
  FAST_CHECK_UNARY(p34.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p34.isIdentity());
  FAST_CHECK_UNARY(p34.isInvertible());
  FAST_CHECK_UNARY_FALSE(p34.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p34.isMultipleOfIdentity());
  FAST_CHECK_UNARY(p34.isNegativeDefinite());
  FAST_CHECK_UNARY(p34.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p34.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p34.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p34.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p34.isSymmetric());
  FAST_CHECK_UNARY(p34.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p34.isZero());
  FAST_CHECK_UNARY(p34.isTriangular());
  FAST_CHECK_UNARY(p34.isLowerTriangular());
  FAST_CHECK_UNARY(p34.isUpperTriangular());

  MatrixProperties p35(MatrixProperties::DIAGONAL, MatrixProperties::INDEFINITE);
  FAST_CHECK_EQ(p35.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p35.positiveness(), MatrixProperties::INDEFINITE);
  FAST_CHECK_UNARY_FALSE(p35.isConstant());
  FAST_CHECK_UNARY(p35.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p35.isIdentity());
  FAST_CHECK_UNARY_FALSE(p35.isInvertible());
  FAST_CHECK_UNARY_FALSE(p35.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p35.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p35.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p35.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p35.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p35.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p35.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p35.isSymmetric());
  FAST_CHECK_UNARY(p35.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p35.isZero());
  FAST_CHECK_UNARY(p35.isTriangular());
  FAST_CHECK_UNARY(p35.isLowerTriangular());
  FAST_CHECK_UNARY(p35.isUpperTriangular());

  MatrixProperties p36(MatrixProperties::DIAGONAL, MatrixProperties::NON_ZERO_INDEFINITE);
  FAST_CHECK_EQ(p36.shape(), MatrixProperties::DIAGONAL);
  FAST_CHECK_EQ(p36.positiveness(), MatrixProperties::NON_ZERO_INDEFINITE);
  FAST_CHECK_UNARY_FALSE(p36.isConstant());
  FAST_CHECK_UNARY(p36.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p36.isIdentity());
  FAST_CHECK_UNARY(p36.isInvertible());
  FAST_CHECK_UNARY_FALSE(p36.isMinusIdentity());
  FAST_CHECK_UNARY_FALSE(p36.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p36.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p36.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p36.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p36.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p36.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p36.isSymmetric());
  FAST_CHECK_UNARY(p36.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p36.isZero());
  FAST_CHECK_UNARY(p36.isTriangular());
  FAST_CHECK_UNARY(p36.isLowerTriangular());
  FAST_CHECK_UNARY(p36.isUpperTriangular());
  

  MatrixProperties p41(MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::POSITIVE_SEMIDEFINITE);
  FAST_CHECK_EQ(p41.shape(), MatrixProperties::MULTIPLE_OF_IDENTITY);
  FAST_CHECK_EQ(p41.positiveness(), MatrixProperties::POSITIVE_SEMIDEFINITE);
  FAST_CHECK_UNARY_FALSE(p41.isConstant());
  FAST_CHECK_UNARY(p41.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p41.isIdentity());
  FAST_CHECK_UNARY_FALSE(p41.isInvertible());
  FAST_CHECK_UNARY_FALSE(p41.isMinusIdentity());
  FAST_CHECK_UNARY(p41.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p41.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p41.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p41.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p41.isPositiveDefinite());
  FAST_CHECK_UNARY(p41.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p41.isSymmetric());
  FAST_CHECK_UNARY(p41.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p41.isZero());
  FAST_CHECK_UNARY(p41.isTriangular());
  FAST_CHECK_UNARY(p41.isLowerTriangular());
  FAST_CHECK_UNARY(p41.isUpperTriangular());

  MatrixProperties p42(MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_EQ(p42.shape(), MatrixProperties::MULTIPLE_OF_IDENTITY);
  FAST_CHECK_EQ(p42.positiveness(), MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_UNARY_FALSE(p42.isConstant());
  FAST_CHECK_UNARY(p42.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p42.isIdentity());
  FAST_CHECK_UNARY(p42.isInvertible());
  FAST_CHECK_UNARY_FALSE(p42.isMinusIdentity());
  FAST_CHECK_UNARY(p42.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p42.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p42.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p42.isNonZeroIndefinite());
  FAST_CHECK_UNARY(p42.isPositiveDefinite());
  FAST_CHECK_UNARY(p42.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p42.isSymmetric());
  FAST_CHECK_UNARY(p42.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p42.isZero());
  FAST_CHECK_UNARY(p42.isTriangular());
  FAST_CHECK_UNARY(p42.isLowerTriangular());
  FAST_CHECK_UNARY(p42.isUpperTriangular());

  MatrixProperties p43(MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  FAST_CHECK_EQ(p43.shape(), MatrixProperties::MULTIPLE_OF_IDENTITY);
  FAST_CHECK_EQ(p43.positiveness(), MatrixProperties::NEGATIVE_SEMIDEFINITE);
  FAST_CHECK_UNARY_FALSE(p43.isConstant());
  FAST_CHECK_UNARY(p43.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p43.isIdentity());
  FAST_CHECK_UNARY_FALSE(p43.isInvertible());
  FAST_CHECK_UNARY_FALSE(p43.isMinusIdentity());
  FAST_CHECK_UNARY(p43.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p43.isNegativeDefinite());
  FAST_CHECK_UNARY(p43.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p43.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p43.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p43.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p43.isSymmetric());
  FAST_CHECK_UNARY(p43.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p43.isZero());
  FAST_CHECK_UNARY(p43.isTriangular());
  FAST_CHECK_UNARY(p43.isLowerTriangular());
  FAST_CHECK_UNARY(p43.isUpperTriangular());

  MatrixProperties p44(MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_EQ(p44.shape(), MatrixProperties::MULTIPLE_OF_IDENTITY);
  FAST_CHECK_EQ(p44.positiveness(), MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_UNARY_FALSE(p44.isConstant());
  FAST_CHECK_UNARY(p44.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p44.isIdentity());
  FAST_CHECK_UNARY(p44.isInvertible());
  FAST_CHECK_UNARY_FALSE(p44.isMinusIdentity());
  FAST_CHECK_UNARY(p44.isMultipleOfIdentity());
  FAST_CHECK_UNARY(p44.isNegativeDefinite());
  FAST_CHECK_UNARY(p44.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p44.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p44.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p44.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p44.isSymmetric());
  FAST_CHECK_UNARY(p44.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p44.isZero());
  FAST_CHECK_UNARY(p44.isTriangular());
  FAST_CHECK_UNARY(p44.isLowerTriangular());
  FAST_CHECK_UNARY(p44.isUpperTriangular());

  MatrixProperties p45(MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::INDEFINITE);
  FAST_CHECK_EQ(p45.shape(), MatrixProperties::MULTIPLE_OF_IDENTITY);
  FAST_CHECK_EQ(p45.positiveness(), MatrixProperties::INDEFINITE);
  FAST_CHECK_UNARY_FALSE(p45.isConstant());
  FAST_CHECK_UNARY(p45.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p45.isIdentity());
  FAST_CHECK_UNARY_FALSE(p45.isInvertible());
  FAST_CHECK_UNARY_FALSE(p45.isMinusIdentity());
  FAST_CHECK_UNARY(p45.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p45.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p45.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p45.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p45.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p45.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p45.isSymmetric());
  FAST_CHECK_UNARY(p45.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p45.isZero());
  FAST_CHECK_UNARY(p45.isTriangular());
  FAST_CHECK_UNARY(p45.isLowerTriangular());
  FAST_CHECK_UNARY(p45.isUpperTriangular());

  MatrixProperties p46(MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NON_ZERO_INDEFINITE);
  FAST_CHECK_EQ(p46.shape(), MatrixProperties::MULTIPLE_OF_IDENTITY);
  FAST_CHECK_EQ(p46.positiveness(), MatrixProperties::NON_ZERO_INDEFINITE);
  FAST_CHECK_UNARY_FALSE(p46.isConstant());
  FAST_CHECK_UNARY(p46.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p46.isIdentity());
  FAST_CHECK_UNARY(p46.isInvertible());
  FAST_CHECK_UNARY_FALSE(p46.isMinusIdentity());
  FAST_CHECK_UNARY(p46.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p46.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p46.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p46.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p46.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p46.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p46.isSymmetric());
  FAST_CHECK_UNARY(p46.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p46.isZero());
  FAST_CHECK_UNARY(p46.isTriangular());
  FAST_CHECK_UNARY(p46.isLowerTriangular());
  FAST_CHECK_UNARY(p46.isUpperTriangular());

  MatrixProperties p51(MatrixProperties::IDENTITY, MatrixProperties::POSITIVE_SEMIDEFINITE);
  FAST_CHECK_EQ(p51.shape(), MatrixProperties::IDENTITY);
  FAST_CHECK_EQ(p51.positiveness(), MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_UNARY(p51.isConstant());
  FAST_CHECK_UNARY(p51.isDiagonal());
  FAST_CHECK_UNARY(p51.isIdentity());
  FAST_CHECK_UNARY(p51.isInvertible());
  FAST_CHECK_UNARY_FALSE(p51.isMinusIdentity());
  FAST_CHECK_UNARY(p51.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p51.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p51.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p51.isNonZeroIndefinite());
  FAST_CHECK_UNARY(p51.isPositiveDefinite());
  FAST_CHECK_UNARY(p51.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p51.isSymmetric());
  FAST_CHECK_UNARY(p51.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p51.isZero());
  FAST_CHECK_UNARY(p51.isTriangular());
  FAST_CHECK_UNARY(p51.isLowerTriangular());
  FAST_CHECK_UNARY(p51.isUpperTriangular());

  MatrixProperties p52(MatrixProperties::IDENTITY, MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_EQ(p52.shape(), MatrixProperties::IDENTITY);
  FAST_CHECK_EQ(p52.positiveness(), MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_UNARY(p52.isConstant());
  FAST_CHECK_UNARY(p52.isDiagonal());
  FAST_CHECK_UNARY(p52.isIdentity());
  FAST_CHECK_UNARY(p52.isInvertible());
  FAST_CHECK_UNARY_FALSE(p52.isMinusIdentity());
  FAST_CHECK_UNARY(p52.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p52.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p52.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p52.isNonZeroIndefinite());
  FAST_CHECK_UNARY(p52.isPositiveDefinite());
  FAST_CHECK_UNARY(p52.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p52.isSymmetric());
  FAST_CHECK_UNARY(p52.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p52.isZero());
  FAST_CHECK_UNARY(p52.isTriangular());
  FAST_CHECK_UNARY(p52.isLowerTriangular());
  FAST_CHECK_UNARY(p52.isUpperTriangular());

  CHECK_THROWS_AS(
    MatrixProperties p53(MatrixProperties::IDENTITY, MatrixProperties::NEGATIVE_SEMIDEFINITE)
    , std::runtime_error);
  CHECK_THROWS_AS(
    MatrixProperties p54(MatrixProperties::IDENTITY, MatrixProperties::NEGATIVE_DEFINITE)
    , std::runtime_error);

  MatrixProperties p55(MatrixProperties::IDENTITY, MatrixProperties::INDEFINITE);
  FAST_CHECK_EQ(p55.shape(), MatrixProperties::IDENTITY);
  FAST_CHECK_EQ(p55.positiveness(), MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_UNARY(p55.isConstant());
  FAST_CHECK_UNARY(p55.isDiagonal());
  FAST_CHECK_UNARY(p55.isIdentity());
  FAST_CHECK_UNARY(p55.isInvertible());
  FAST_CHECK_UNARY_FALSE(p55.isMinusIdentity());
  FAST_CHECK_UNARY(p55.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p55.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p55.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p55.isNonZeroIndefinite());
  FAST_CHECK_UNARY(p55.isPositiveDefinite());
  FAST_CHECK_UNARY(p55.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p55.isSymmetric());
  FAST_CHECK_UNARY(p55.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p55.isZero());
  FAST_CHECK_UNARY(p55.isTriangular());
  FAST_CHECK_UNARY(p55.isLowerTriangular());
  FAST_CHECK_UNARY(p55.isUpperTriangular());

  MatrixProperties p56(MatrixProperties::IDENTITY, MatrixProperties::NON_ZERO_INDEFINITE);
  FAST_CHECK_EQ(p56.shape(), MatrixProperties::IDENTITY);
  FAST_CHECK_EQ(p56.positiveness(), MatrixProperties::POSITIVE_DEFINITE);
  FAST_CHECK_UNARY(p56.isConstant());
  FAST_CHECK_UNARY(p56.isDiagonal());
  FAST_CHECK_UNARY(p56.isIdentity());
  FAST_CHECK_UNARY(p56.isInvertible());
  FAST_CHECK_UNARY_FALSE(p56.isMinusIdentity());
  FAST_CHECK_UNARY(p56.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p56.isNegativeDefinite());
  FAST_CHECK_UNARY_FALSE(p56.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p56.isNonZeroIndefinite());
  FAST_CHECK_UNARY(p56.isPositiveDefinite());
  FAST_CHECK_UNARY(p56.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p56.isSymmetric());
  FAST_CHECK_UNARY(p56.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p56.isZero());
  FAST_CHECK_UNARY(p56.isTriangular());
  FAST_CHECK_UNARY(p56.isLowerTriangular());
  FAST_CHECK_UNARY(p56.isUpperTriangular());

  CHECK_THROWS_AS(
    MatrixProperties p61(MatrixProperties::MINUS_IDENTITY, MatrixProperties::POSITIVE_SEMIDEFINITE)
    , std::runtime_error);
  CHECK_THROWS_AS(
    MatrixProperties p62(MatrixProperties::MINUS_IDENTITY, MatrixProperties::POSITIVE_DEFINITE)
    , std::runtime_error);
  
  
  MatrixProperties p63(MatrixProperties::MINUS_IDENTITY, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  FAST_CHECK_EQ(p63.shape(), MatrixProperties::MINUS_IDENTITY);
  FAST_CHECK_EQ(p63.positiveness(), MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_UNARY(p63.isConstant());
  FAST_CHECK_UNARY(p63.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p63.isIdentity());
  FAST_CHECK_UNARY(p63.isInvertible());
  FAST_CHECK_UNARY(p63.isMinusIdentity());
  FAST_CHECK_UNARY(p63.isMultipleOfIdentity());
  FAST_CHECK_UNARY(p63.isNegativeDefinite());
  FAST_CHECK_UNARY(p63.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p63.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p63.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p63.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p63.isSymmetric());
  FAST_CHECK_UNARY(p63.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p63.isZero());
  FAST_CHECK_UNARY(p63.isTriangular());
  FAST_CHECK_UNARY(p63.isLowerTriangular());
  FAST_CHECK_UNARY(p63.isUpperTriangular());

  MatrixProperties p64(MatrixProperties::MINUS_IDENTITY, MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_EQ(p64.shape(), MatrixProperties::MINUS_IDENTITY);
  FAST_CHECK_EQ(p64.positiveness(), MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_UNARY(p64.isConstant());
  FAST_CHECK_UNARY(p64.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p64.isIdentity());
  FAST_CHECK_UNARY(p64.isInvertible());
  FAST_CHECK_UNARY(p64.isMinusIdentity());
  FAST_CHECK_UNARY(p64.isMultipleOfIdentity());
  FAST_CHECK_UNARY(p64.isNegativeDefinite());
  FAST_CHECK_UNARY(p64.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p64.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p64.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p64.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p64.isSymmetric());
  FAST_CHECK_UNARY(p64.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p64.isZero());
  FAST_CHECK_UNARY(p64.isTriangular());
  FAST_CHECK_UNARY(p64.isLowerTriangular());
  FAST_CHECK_UNARY(p64.isUpperTriangular());

  MatrixProperties p65(MatrixProperties::MINUS_IDENTITY, MatrixProperties::INDEFINITE);
  FAST_CHECK_EQ(p65.shape(), MatrixProperties::MINUS_IDENTITY);
  FAST_CHECK_EQ(p65.positiveness(), MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_UNARY(p65.isConstant());
  FAST_CHECK_UNARY(p65.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p65.isIdentity());
  FAST_CHECK_UNARY(p65.isInvertible());
  FAST_CHECK_UNARY(p65.isMinusIdentity());
  FAST_CHECK_UNARY(p65.isMultipleOfIdentity());
  FAST_CHECK_UNARY(p65.isNegativeDefinite());
  FAST_CHECK_UNARY(p65.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p65.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p65.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p65.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p65.isSymmetric());
  FAST_CHECK_UNARY(p65.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p65.isZero());
  FAST_CHECK_UNARY(p65.isTriangular());
  FAST_CHECK_UNARY(p65.isLowerTriangular());
  FAST_CHECK_UNARY(p65.isUpperTriangular());

  MatrixProperties p66(MatrixProperties::MINUS_IDENTITY, MatrixProperties::NON_ZERO_INDEFINITE);
  FAST_CHECK_EQ(p66.shape(), MatrixProperties::MINUS_IDENTITY);
  FAST_CHECK_EQ(p66.positiveness(), MatrixProperties::NEGATIVE_DEFINITE);
  FAST_CHECK_UNARY(p66.isConstant());
  FAST_CHECK_UNARY(p66.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p66.isIdentity());
  FAST_CHECK_UNARY(p66.isInvertible());
  FAST_CHECK_UNARY(p66.isMinusIdentity());
  FAST_CHECK_UNARY(p66.isMultipleOfIdentity());
  FAST_CHECK_UNARY(p66.isNegativeDefinite());
  FAST_CHECK_UNARY(p66.isNegativeSemidefinite());
  FAST_CHECK_UNARY(p66.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p66.isPositiveDefinite());
  FAST_CHECK_UNARY_FALSE(p66.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p66.isSymmetric());
  FAST_CHECK_UNARY(p66.isIndefinite());
  FAST_CHECK_UNARY_FALSE(p66.isZero());
  FAST_CHECK_UNARY(p66.isTriangular());
  FAST_CHECK_UNARY(p66.isLowerTriangular());
  FAST_CHECK_UNARY(p66.isUpperTriangular());
  
  MatrixProperties p71(MatrixProperties::ZERO, MatrixProperties::POSITIVE_SEMIDEFINITE);
  FAST_CHECK_EQ(p71.shape(), MatrixProperties::ZERO);
  FAST_CHECK_EQ(p71.positiveness(), MatrixProperties::POSITIVE_SEMIDEFINITE);
  FAST_CHECK_UNARY(p71.isConstant());
  FAST_CHECK_UNARY(p71.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p71.isIdentity());
  FAST_CHECK_UNARY_FALSE(p71.isInvertible());
  FAST_CHECK_UNARY_FALSE(p71.isMinusIdentity());
  FAST_CHECK_UNARY(p71.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p71.isNegativeDefinite());
  FAST_CHECK_UNARY(p71.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p71.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p71.isPositiveDefinite());
  FAST_CHECK_UNARY(p71.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p71.isSymmetric());
  FAST_CHECK_UNARY(p71.isIndefinite());
  FAST_CHECK_UNARY(p71.isZero());
  FAST_CHECK_UNARY(p71.isTriangular());
  FAST_CHECK_UNARY(p71.isLowerTriangular());
  FAST_CHECK_UNARY(p71.isUpperTriangular());

  
  CHECK_THROWS_AS(
    MatrixProperties p72(MatrixProperties::ZERO, MatrixProperties::POSITIVE_DEFINITE)
    , std::runtime_error);
  
  MatrixProperties p73(MatrixProperties::ZERO, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  FAST_CHECK_EQ(p73.shape(), MatrixProperties::ZERO);
  FAST_CHECK_EQ(p73.positiveness(), MatrixProperties::NEGATIVE_SEMIDEFINITE);
  FAST_CHECK_UNARY(p73.isConstant());
  FAST_CHECK_UNARY(p73.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p73.isIdentity());
  FAST_CHECK_UNARY_FALSE(p73.isInvertible());
  FAST_CHECK_UNARY_FALSE(p73.isMinusIdentity());
  FAST_CHECK_UNARY(p73.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p73.isNegativeDefinite());
  FAST_CHECK_UNARY(p73.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p73.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p73.isPositiveDefinite());
  FAST_CHECK_UNARY(p73.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p73.isSymmetric());
  FAST_CHECK_UNARY(p73.isIndefinite());
  FAST_CHECK_UNARY(p73.isZero());
  FAST_CHECK_UNARY(p73.isTriangular());
  FAST_CHECK_UNARY(p73.isLowerTriangular());
  FAST_CHECK_UNARY(p73.isUpperTriangular());

  CHECK_THROWS_AS(
    MatrixProperties p74(MatrixProperties::ZERO, MatrixProperties::NEGATIVE_DEFINITE)
    , std::runtime_error);

  MatrixProperties p75(MatrixProperties::ZERO, MatrixProperties::INDEFINITE);
  FAST_CHECK_EQ(p75.shape(), MatrixProperties::ZERO);
  FAST_CHECK_EQ(p75.positiveness(), MatrixProperties::INDEFINITE);
  FAST_CHECK_UNARY(p75.isConstant());
  FAST_CHECK_UNARY(p75.isDiagonal());
  FAST_CHECK_UNARY_FALSE(p75.isIdentity());
  FAST_CHECK_UNARY_FALSE(p75.isInvertible());
  FAST_CHECK_UNARY_FALSE(p75.isMinusIdentity());
  FAST_CHECK_UNARY(p75.isMultipleOfIdentity());
  FAST_CHECK_UNARY_FALSE(p75.isNegativeDefinite());
  FAST_CHECK_UNARY(p75.isNegativeSemidefinite());
  FAST_CHECK_UNARY_FALSE(p75.isNonZeroIndefinite());
  FAST_CHECK_UNARY_FALSE(p75.isPositiveDefinite());
  FAST_CHECK_UNARY(p75.isPositiveSemiDefinite());
  FAST_CHECK_UNARY(p75.isSymmetric());
  FAST_CHECK_UNARY(p75.isIndefinite());
  FAST_CHECK_UNARY(p75.isZero());
  FAST_CHECK_UNARY(p75.isTriangular());
  FAST_CHECK_UNARY(p75.isLowerTriangular());
  FAST_CHECK_UNARY(p75.isUpperTriangular());

  CHECK_THROWS_AS(
    MatrixProperties p76(MatrixProperties::ZERO, MatrixProperties::NON_ZERO_INDEFINITE)
    , std::runtime_error);
}


#define buildAndCheck(shouldThrow, ... ) \
  if (shouldThrow) {\
    CHECK_THROWS_AS(MatrixProperties(__VA_ARGS__), std::runtime_error); \
  } else {\
    CHECK_NOTHROW(MatrixProperties(__VA_ARGS__)); }

TEST_CASE("Test constness compatibility")
{
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::GENERAL, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::GENERAL, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::GENERAL, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::GENERAL, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::GENERAL, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::GENERAL, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::GENERAL, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::DIAGONAL, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::DIAGONAL, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::DIAGONAL, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::DIAGONAL, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::DIAGONAL, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::DIAGONAL, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::DIAGONAL, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(false), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::IDENTITY, MatrixProperties::NA);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::IDENTITY, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::IDENTITY, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::IDENTITY, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::IDENTITY, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::IDENTITY, MatrixProperties::INDEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::IDENTITY, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NA);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::MINUS_IDENTITY, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::MINUS_IDENTITY, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::MINUS_IDENTITY, MatrixProperties::INDEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::ZERO, MatrixProperties::NA);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::ZERO, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::ZERO, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::ZERO, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::ZERO, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::ZERO, MatrixProperties::INDEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(false), MatrixProperties::ZERO, MatrixProperties::NON_ZERO_INDEFINITE);


  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::GENERAL, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::GENERAL, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::GENERAL, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::GENERAL, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::GENERAL, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::GENERAL, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::GENERAL, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::DIAGONAL, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::DIAGONAL, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::DIAGONAL, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::DIAGONAL, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::DIAGONAL, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::DIAGONAL, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::DIAGONAL, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::IDENTITY, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::IDENTITY, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::IDENTITY, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(true), MatrixProperties::IDENTITY, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(true), MatrixProperties::IDENTITY, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::IDENTITY, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::IDENTITY, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NA);
  buildAndCheck(true, MatrixProperties::Constness(true), MatrixProperties::MINUS_IDENTITY, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(true), MatrixProperties::MINUS_IDENTITY, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::MINUS_IDENTITY, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::ZERO, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::ZERO, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(true), MatrixProperties::ZERO, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::ZERO, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(true), MatrixProperties::ZERO, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Constness(true), MatrixProperties::ZERO, MatrixProperties::INDEFINITE);
  buildAndCheck(true, MatrixProperties::Constness(true), MatrixProperties::ZERO, MatrixProperties::NON_ZERO_INDEFINITE);
}

TEST_CASE("Test invertibility compatibility")
{
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::GENERAL, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::GENERAL, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::GENERAL, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::GENERAL, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::GENERAL, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::GENERAL, MatrixProperties::INDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::GENERAL, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::INDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::INDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::DIAGONAL, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::DIAGONAL, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::DIAGONAL, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::DIAGONAL, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::DIAGONAL, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::DIAGONAL, MatrixProperties::INDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::DIAGONAL, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::INDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::IDENTITY, MatrixProperties::NA);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::IDENTITY, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::IDENTITY, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::IDENTITY, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::IDENTITY, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::IDENTITY, MatrixProperties::INDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::IDENTITY, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NA);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::MINUS_IDENTITY, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::MINUS_IDENTITY, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::MINUS_IDENTITY, MatrixProperties::INDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::ZERO, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::ZERO, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::ZERO, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::ZERO, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::ZERO, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(false), MatrixProperties::ZERO, MatrixProperties::INDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(false), MatrixProperties::ZERO, MatrixProperties::NON_ZERO_INDEFINITE);


  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::GENERAL, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::GENERAL, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::GENERAL, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::GENERAL, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::GENERAL, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::GENERAL, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::GENERAL, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::LOWER_TRIANGULAR, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::UPPER_TRIANGULAR, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::DIAGONAL, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::DIAGONAL, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::DIAGONAL, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::DIAGONAL, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::DIAGONAL, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::DIAGONAL, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::DIAGONAL, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::MULTIPLE_OF_IDENTITY, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::IDENTITY, MatrixProperties::NA);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::IDENTITY, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::IDENTITY, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(true), MatrixProperties::IDENTITY, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(true), MatrixProperties::IDENTITY, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::IDENTITY, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::IDENTITY, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NA);
  buildAndCheck(true, MatrixProperties::Invertibility(true), MatrixProperties::MINUS_IDENTITY, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(true), MatrixProperties::MINUS_IDENTITY, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::MINUS_IDENTITY, MatrixProperties::INDEFINITE);
  buildAndCheck(false, MatrixProperties::Invertibility(true), MatrixProperties::MINUS_IDENTITY, MatrixProperties::NON_ZERO_INDEFINITE);

  buildAndCheck(true, MatrixProperties::Invertibility(true), MatrixProperties::ZERO, MatrixProperties::NA);
  buildAndCheck(true, MatrixProperties::Invertibility(true), MatrixProperties::ZERO, MatrixProperties::POSITIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(true), MatrixProperties::ZERO, MatrixProperties::POSITIVE_DEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(true), MatrixProperties::ZERO, MatrixProperties::NEGATIVE_SEMIDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(true), MatrixProperties::ZERO, MatrixProperties::NEGATIVE_DEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(true), MatrixProperties::ZERO, MatrixProperties::INDEFINITE);
  buildAndCheck(true, MatrixProperties::Invertibility(true), MatrixProperties::ZERO, MatrixProperties::NON_ZERO_INDEFINITE);
}

TEST_CASE("Test argument order and repetition")
{
  MatrixProperties::Shape s = MatrixProperties::GENERAL;
  MatrixProperties::Positiveness p = MatrixProperties::NA;
  MatrixProperties::Constness c;
  MatrixProperties::Invertibility i;

  buildAndCheck(false, s);
  buildAndCheck(false, s);
  buildAndCheck(false, c);
  buildAndCheck(false, i);


  buildAndCheck(false, s, p);
  buildAndCheck(false, s, c);
  buildAndCheck(false, s, i);

  buildAndCheck(false, p, s);
  buildAndCheck(false, p, c);
  buildAndCheck(false, p, i);

  buildAndCheck(false, c, s);
  buildAndCheck(false, c, p);
  buildAndCheck(false, c, i);

  buildAndCheck(false, i, s);
  buildAndCheck(false, i, p);
  buildAndCheck(false, i, c);


  buildAndCheck(false, s, p, c);
  buildAndCheck(false, s, p, i);
  buildAndCheck(false, s, c, p);
  buildAndCheck(false, s, c, i);
  buildAndCheck(false, s, i, p);
  buildAndCheck(false, s, i, c);

  buildAndCheck(false, p, s, c);
  buildAndCheck(false, p, s, i);
  buildAndCheck(false, p, c, s);
  buildAndCheck(false, p, c, i);
  buildAndCheck(false, p, i, s);
  buildAndCheck(false, p, i, c);

  buildAndCheck(false, c, s, p);
  buildAndCheck(false, c, s, i);
  buildAndCheck(false, c, p, s);
  buildAndCheck(false, c, p, i);
  buildAndCheck(false, c, i, s);
  buildAndCheck(false, c, i, p);

  buildAndCheck(false, i, s, p);
  buildAndCheck(false, i, s, c);
  buildAndCheck(false, i, p, s);
  buildAndCheck(false, i, p, c);
  buildAndCheck(false, i, c, s);
  buildAndCheck(false, i, c, p);


  buildAndCheck(false, s, p, c, i);
  buildAndCheck(false, s, p, i, c);
  buildAndCheck(false, s, c, p, i);
  buildAndCheck(false, s, c, i, p);
  buildAndCheck(false, s, i, p, c);
  buildAndCheck(false, s, i, c, p);

  buildAndCheck(false, p, s, c, i);
  buildAndCheck(false, p, s, i, c);
  buildAndCheck(false, p, c, s, i);
  buildAndCheck(false, p, c, i, s);
  buildAndCheck(false, p, i, s, c);
  buildAndCheck(false, p, i, c, s);

  buildAndCheck(false, c, s, p, i);
  buildAndCheck(false, c, s, i, p);
  buildAndCheck(false, c, p, s, i);
  buildAndCheck(false, c, p, i, s);
  buildAndCheck(false, c, i, s, p);
  buildAndCheck(false, c, i, p, s);

  buildAndCheck(false, i, s, p, c);
  buildAndCheck(false, i, s, c, p);
  buildAndCheck(false, i, p, s, c);
  buildAndCheck(false, i, p, c, s);
  buildAndCheck(false, i, c, s, p);
  buildAndCheck(false, i, c, p, s);
}
