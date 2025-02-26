import functools
import time
from typing import Any, Callable

from .logger import logger


def log_function_call(func: Callable) -> Callable:
    """Decorator to log function calls with timing"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        # Log function call
        logger.debug(f"Calling function '{func.__name__}' " f"with args: {args}, kwargs: {kwargs}")

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Log successful execution
            logger.debug(
                f"Function '{func.__name__}' completed successfully "
                f"in {execution_time:.2f} seconds"
            )

            return result

        except Exception as e:
            # Log error with full stack trace
            logger.exception(f"Error in function '{func.__name__}': {str(e)}")
            raise

    return wrapper


def log_class_methods(cls: Any) -> Any:
    """Decorator to log all methods of a class"""
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and not attr_name.startswith("__"):
            setattr(cls, attr_name, log_function_call(attr_value))
    return cls


# Example usage
@log_function_call
def example_function(x: int, y: int) -> int:
    return x + y


@log_class_methods
class ExampleClass:
    def method1(self):
        return "Hello"

    def method2(self, x):
        return x * 2
