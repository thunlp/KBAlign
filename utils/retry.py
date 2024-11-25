import functools
import time


def retry(max_retries=3, sleep_time=5,allow_empty=False):
    def retry_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    result = func(*args, **kwargs)
                    if allow_empty==False and result == "":
                        raise ValueError("Result is null or empty")
                    return result
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"Attempt {retries + 1} on {func.__name__} failed: {e}")
                    retries += 1
                    time.sleep(sleep_time)
                    if retries == max_retries:
                        raise ValueError(
                            f"{func.__name__} failed after {max_retries} attempts"
                        )

        return wrapper

    return retry_decorator
