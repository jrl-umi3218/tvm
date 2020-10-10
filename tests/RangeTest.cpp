/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Range.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace tvm;

TEST_CASE("Contains integer")
{
  Range r(3, 5);
  FAST_CHECK_UNARY_FALSE(r.contains(0));
  FAST_CHECK_UNARY_FALSE(r.contains(1));
  FAST_CHECK_UNARY_FALSE(r.contains(2));
  FAST_CHECK_UNARY(r.contains(3));
  FAST_CHECK_UNARY(r.contains(4));
  FAST_CHECK_UNARY(r.contains(5));
  FAST_CHECK_UNARY(r.contains(6));
  FAST_CHECK_UNARY(r.contains(7));
  FAST_CHECK_UNARY_FALSE(r.contains(8));
  FAST_CHECK_UNARY_FALSE(r.contains(9));
}

TEST_CASE("Contains range")
{
  Range r(3, 5); // 3 4 5 6 7
  FAST_CHECK_UNARY_FALSE(r.contains(Range(0, 0))); //
  FAST_CHECK_UNARY_FALSE(r.contains(Range(0, 2))); // 0 1
  FAST_CHECK_UNARY_FALSE(r.contains(Range(0, 3))); // 0 1 2 
  FAST_CHECK_UNARY_FALSE(r.contains(Range(0, 4))); // 0 1 2 3 
  FAST_CHECK_UNARY_FALSE(r.contains(Range(0, 6))); // 0 1 2 3 4 5 
  FAST_CHECK_UNARY_FALSE(r.contains(Range(0, 8))); // 0 1 2 3 4 5 6 7
  FAST_CHECK_UNARY_FALSE(r.contains(Range(0, 9))); // 0 1 2 3 4 5 6 7 8

  FAST_CHECK_UNARY_FALSE(r.contains(Range(2, 0))); // 
  FAST_CHECK_UNARY_FALSE(r.contains(Range(2, 1))); // 2
  FAST_CHECK_UNARY_FALSE(r.contains(Range(2, 2))); // 2 3
  FAST_CHECK_UNARY_FALSE(r.contains(Range(2, 4))); // 2 3 4 5
  FAST_CHECK_UNARY_FALSE(r.contains(Range(2, 6))); // 2 3 4 5 6 7
  FAST_CHECK_UNARY_FALSE(r.contains(Range(2, 7))); // 2 3 4 5 6 7 8

  FAST_CHECK_UNARY(r.contains(Range(3, 0)));       //
  FAST_CHECK_UNARY(r.contains(Range(3, 1)));       // 3
  FAST_CHECK_UNARY(r.contains(Range(3, 3)));       // 3 4 5
  FAST_CHECK_UNARY(r.contains(Range(3, 5)));       // 3 4 5 6 7
  FAST_CHECK_UNARY_FALSE(r.contains(Range(3, 6))); // 3 4 5 6 8

  FAST_CHECK_UNARY(r.contains(Range(4, 0)));       //
  FAST_CHECK_UNARY(r.contains(Range(4, 1)));       // 4
  FAST_CHECK_UNARY(r.contains(Range(4, 3)));       // 4 5 6
  FAST_CHECK_UNARY(r.contains(Range(4, 4)));       // 4 5 6 7
  FAST_CHECK_UNARY_FALSE(r.contains(Range(4, 5))); // 4 5 6 7 8

  FAST_CHECK_UNARY(r.contains(Range(6, 0)));       //
  FAST_CHECK_UNARY(r.contains(Range(6, 1)));       // 6
  FAST_CHECK_UNARY(r.contains(Range(6, 2)));       // 6 7
  FAST_CHECK_UNARY_FALSE(r.contains(Range(6, 4))); // 6 7 8 9

  FAST_CHECK_UNARY(r.contains(Range(7, 0)));       //
  FAST_CHECK_UNARY(r.contains(Range(7, 1)));       // 7
  FAST_CHECK_UNARY_FALSE(r.contains(Range(7, 2))); // 7 8
  FAST_CHECK_UNARY_FALSE(r.contains(Range(6, 3))); // 7 8 9

  FAST_CHECK_UNARY_FALSE(r.contains(Range(8, 0))); //
  FAST_CHECK_UNARY_FALSE(r.contains(Range(8, 1))); // 8
  FAST_CHECK_UNARY_FALSE(r.contains(Range(8, 2))); // 8 9

  FAST_CHECK_UNARY_FALSE(r.contains(Range(9, 1))); // 9
}

TEST_CASE("Intersects range")
{
  Range r(3, 5); // 3 4 5 6 7
  FAST_CHECK_UNARY_FALSE(r.intersects(Range(0, 0))); //
  FAST_CHECK_UNARY_FALSE(r.intersects(Range(0, 2))); // 0 1
  FAST_CHECK_UNARY_FALSE(r.intersects(Range(0, 3))); // 0 1 2
  FAST_CHECK_UNARY(r.intersects(Range(0, 4))); // 0 1 2 3
  FAST_CHECK_UNARY(r.intersects(Range(0, 6))); // 0 1 2 3 4 5
  FAST_CHECK_UNARY(r.intersects(Range(0, 8))); // 0 1 2 3 4 5 6 7
  FAST_CHECK_UNARY(r.intersects(Range(0, 9))); // 0 1 2 3 4 5 6 7 8

  FAST_CHECK_UNARY_FALSE(r.intersects(Range(2, 0))); //
  FAST_CHECK_UNARY_FALSE(r.intersects(Range(2, 1))); // 2
  FAST_CHECK_UNARY(r.intersects(Range(2, 2))); // 2 3
  FAST_CHECK_UNARY(r.intersects(Range(2, 4))); // 2 3 4 5
  FAST_CHECK_UNARY(r.intersects(Range(2, 6))); // 2 3 4 5 6 7
  FAST_CHECK_UNARY(r.intersects(Range(2, 7))); // 2 3 4 5 6 7 8

  FAST_CHECK_UNARY(r.intersects(Range(3, 0)));       //
  FAST_CHECK_UNARY(r.intersects(Range(3, 1)));       // 3
  FAST_CHECK_UNARY(r.intersects(Range(3, 3)));       // 3 4 5
  FAST_CHECK_UNARY(r.intersects(Range(3, 5)));       // 3 4 5 6 7
  FAST_CHECK_UNARY(r.intersects(Range(3, 6))); // 3 4 5 6 8

  FAST_CHECK_UNARY(r.intersects(Range(4, 0)));       //
  FAST_CHECK_UNARY(r.intersects(Range(4, 1)));       // 4
  FAST_CHECK_UNARY(r.intersects(Range(4, 3)));       // 4 5 6
  FAST_CHECK_UNARY(r.intersects(Range(4, 4)));       // 4 5 6 7
  FAST_CHECK_UNARY(r.intersects(Range(4, 5))); // 4 5 6 7 8

  FAST_CHECK_UNARY(r.intersects(Range(6, 0)));       //
  FAST_CHECK_UNARY(r.intersects(Range(6, 1)));       // 6
  FAST_CHECK_UNARY(r.intersects(Range(6, 2)));       // 6 7
  FAST_CHECK_UNARY(r.intersects(Range(6, 4))); // 6 7 8 9

  FAST_CHECK_UNARY(r.intersects(Range(7, 0)));       //
  FAST_CHECK_UNARY(r.intersects(Range(7, 1)));       // 7
  FAST_CHECK_UNARY(r.intersects(Range(7, 2))); // 7 8
  FAST_CHECK_UNARY(r.intersects(Range(6, 3))); // 7 8 9

  FAST_CHECK_UNARY_FALSE(r.intersects(Range(8, 0))); //
  FAST_CHECK_UNARY_FALSE(r.intersects(Range(8, 1))); // 8
  FAST_CHECK_UNARY_FALSE(r.intersects(Range(8, 2))); // 8 9

  FAST_CHECK_UNARY_FALSE(r.intersects(Range(9, 1))); // 9
}
