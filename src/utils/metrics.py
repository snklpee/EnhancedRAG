import time
import functools
from typing import Callable, Any

def track_metrics(counter_fn: Callable[..., int]):
    """
    Decorator to measure execution time and record a count derived by counter_fn.

    counter_fn should accept exactly what the decorated function returns:
      - If func returns (a, b, c), counter_fn(a, b, c) -> int
      - If func returns x,       counter_fn(x)       -> int
    """
    def decorator(func: Callable[..., Any]):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start

            # Unpack only the values that func returned
            if isinstance(result, tuple):
                count = counter_fn(*result)
            else:
                count = counter_fn(result)

            print(f"[METRICS] {func.__name__}: time={duration:.2f}s, count={count}")
            return result
        return wrapper
    return decorator
