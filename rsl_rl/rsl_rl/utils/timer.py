import functools
import time
import torch
from typing import Callable


class TimerCumulative:
    """Timer class to measure the time taken by different parts of the code.

    Example:
    ```python

        @TIMER_CUMULATIVE
        def foo():
            # do something

        foo()
        TIMER_CUMULATIVE.start("bar")
        # do something
        TIMER_CUMULATIVE.stop("bar")

        with TIMER_CUMULATIVE("baz"):
            # do something

        print(TIMER_CUMULATIVE)


    ```
    """

    def __init__(self):
        self.timing_dict = {}
        self.cumulative_time_dict = {}
        self.global_start_time = time.time()

    def start(self, key: str):
        self.timing_dict[key] = time.time()

    def stop(self, key: str):
        torch.cuda.synchronize()
        if key not in self.cumulative_time_dict:
            self.cumulative_time_dict[key] = 0
        self.cumulative_time_dict[key] += time.time() - self.timing_dict.get(key, time.time())

    def __repr__(self) -> str:
        if self.cumulative_time_dict:
            max_key_len = max(len(k) for k in self.cumulative_time_dict.keys())
            max_val_len = max(len(f"{int(v)}") for v in self.cumulative_time_dict.values())
        else:
            max_key_len = 5
            max_val_len = 0
        max_val_len += max_val_len // 3
        print_str = "\nTimer in seconds\n"
        for k, v in self.cumulative_time_dict.items():
            print_str += f"{k:<{max_key_len}}: {v:>{max_val_len+4}_.3f}s\n"

        total_time = time.time() - self.global_start_time
        print_str += f"{'Total':<{max_key_len}}: {total_time:>{max_val_len+4}_.3f}s\n"
        return print_str

    def __call__(self, arg: Callable | str) -> Callable:
        """Allows the instance to be used as a decorator or a context manager."""
        if callable(arg):
            # Used as a decorator
            func = arg

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                key = func.__name__
                self.start(key)
                try:
                    return func(*args, **kwargs)
                finally:
                    self.stop(key)

            return wrapper
        else:
            # Assume arg is a key (string), return a context manager
            key = str(arg)

            class TimerContextManager:
                def __enter__(self_inner):
                    self.start(key)

                def __exit__(self_inner, exc_type, exc_value, traceback):
                    self.stop(key)

            return TimerContextManager()  # type: ignore


# Singleton instance of TimerCumulative
TIMER_CUMULATIVE = TimerCumulative()

# Example usage
if __name__ == "__main__":

    # time all calls of foo
    @TIMER_CUMULATIVE
    def foo(t=0.1):
        time.sleep(t)

    foo()

    # time a specific block of code
    TIMER_CUMULATIVE.start("bar")
    foo(0.2)
    time.sleep(0.1)
    TIMER_CUMULATIVE.stop("bar")

    # use the with statement to time a block of code
    with TIMER_CUMULATIVE("baz"):
        foo(0.3)
        time.sleep(0.1)

    time.sleep(0.1)  # not timed

    # nested timing
    with TIMER_CUMULATIVE("bar_loop"):
        for _ in range(10):
            with TIMER_CUMULATIVE("baz_loop"):
                foo(0.01)
                with TIMER_CUMULATIVE("baz_loop_inner"):
                    foo(0.001)

                TIMER_CUMULATIVE.start("bar_loop_inner")
                TIMER_CUMULATIVE.stop("bar_loop_outer")
                foo(0.001)
                TIMER_CUMULATIVE.stop("bar_loop_inner")
                TIMER_CUMULATIVE.start("bar_loop_outer")

    print(TIMER_CUMULATIVE)
