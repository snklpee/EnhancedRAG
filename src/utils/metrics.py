import time
import functools
from typing import Callable, Any

def track_metrics(counter_fn: Callable[..., int]):
    """
    Decorator to measure execution time and record count returned by counter_fn.
    counter_fn(fn_result) -> int
    """
    def decorator(func: Callable[..., Any]):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            count = counter_fn(result)
            print(f"[METRICS] {func.__name__}: time={duration:.2f}s, count={count}")
            return result
        return wrapper
    return decorator
