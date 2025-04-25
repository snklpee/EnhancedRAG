# /src/utils/metrics.py
import time
import functools
from typing import Callable, Any, Tuple, Union

def track_metrics(
    counter_fn: Callable[..., int],
    *,
    target: str = "outputs",
):
    """
    Decorator to measure execution time and record a count derived by counter_fn.

    Args:
      counter_fn: a function that returns an int when applied to either
        - the decorated function’s return values (default), or
        - the decorated function’s input arguments.
      target: "outputs" to apply counter_fn to the function’s return values;
              "inputs" to apply counter_fn to the function’s args + kwargs.

    Usage:
      # Count on outputs (default):
      @track_metrics(lambda chunks, tc: len(chunks))
      def foo(...) -> Tuple[list, int]: ...

      # Count on inputs:
      @track_metrics(lambda docs: len(docs), target="inputs")
      def foo(docs: List[Document], ...) -> ...
    """
    if target not in ("outputs", "inputs"):
        raise ValueError("track_metrics target must be 'inputs' or 'outputs'")

    def decorator(func: Callable[..., Any]):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start

            # Decide what to pass into counter_fn
            if target == "inputs":
                # Pass only positional args, ignore self if method
                # and ignore kwargs unless you want them included too
                count = counter_fn(*args, **kwargs)
            else:  # "outputs"
                # Unpack return values if tuple, else pass single value
                if isinstance(result, tuple):
                    count = counter_fn(*result)
                else:
                    count = counter_fn(result)

            print(f"[METRICS] {func.__name__}: time={duration:.2f}s, count={count}")
            return result

        return wrapper
    return decorator
