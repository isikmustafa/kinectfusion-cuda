#include "pch.h"

// Include the .cpp file instead of just the header for the functions to test.
// Otherwise we have to explicitly link to each .obj file. 
// Alternatively we could add the .cpp files to the test project but this gives conflicts with precompiled headers.

// A test fixture is a class to collect multiple related tests.
class ExampleTestFixture : public ::testing::Test
{
protected:
    // This value is constant and available for all tests in this class.
    static const int example_constant = 42;

    // This value is not constant and could be different for each test.
    // Note that gtest creates a new instance of this class every time a test is executed.
    // Tests cannot affect each other.
    double some_value;

    // This function is called before each test.
    // Use it for common setup code.
    virtual void SetUp() 
    {
        some_value = 1.5;
    }

    // This function is called after each test.
    // Use it to free ressources, delete temporary files, etc.
    virtual void TearDown() 
    {

    }
};

// This is a unit test belonging to the test fixture (class), specified in the first parameter.
// Specify a descriptive name in the 2nd parameter.
TEST_F(ExampleTestFixture, ExampleTest)
{
    ASSERT_EQ(42, example_constant);
    ASSERT_DOUBLE_EQ(1.5, some_value);
}

TEST_F(ExampleTestFixture, TestGlmToArray)
{
    glm::mat3 true_rot_mat = glm::mat3(
        glm::vec3(0.0, 1.0, 0.0),
        glm::vec3(-1.0, 42.0, 0.0),
        glm::vec3(0.0, 0.0, 1.0));

    float* flat = &true_rot_mat[0][0];
    ASSERT_FLOAT_EQ(42.0, flat[4]);
}