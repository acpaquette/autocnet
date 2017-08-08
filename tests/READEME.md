## Testing

The project is slowly transitioning over to using pytest (from Nose) for all testing.  Traditionally, the conftest.py file is used to support session wide testing fixtures.  We are using the same model here, defining Mocks of our primary objects that are used throughout the testing suite.
