#include <gtest/gtest.h>

TEST(GTEST_TEST, CheckTrue)
{
  unsigned int i = 1;
  ++i;
  EXPECT_EQ((unsigned int)2, i);
}
