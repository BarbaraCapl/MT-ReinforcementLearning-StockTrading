# Content of test_utils.py
import pytest
from data_preprocessing.preprocessors import calculate_price_volume_WhartonData  

def test_function1():
    # Assume function1() should return True when it succeeds
    assert function1() == True

def test_function2():
    # Assume function2() takes a parameter and returns a string twice as long
    input_str = "hello"
    result = function2(input_str)
    assert len(result) == 2 * len(input_str)
    assert type(result) is str

# Add more tests for other functions
