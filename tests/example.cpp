#include <gtest/gtest.h>
#include <iostream>

// A simple test to verify testing framework works
TEST(BasicTest, PrintTest) {
  std::cout << "Hello from test!" << std::endl;
  EXPECT_TRUE(true); // Simple assertion that will always pass
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  std::cout << "Starting test..." << std::endl;
  return RUN_ALL_TESTS();
}