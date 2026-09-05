import os


def get_total_cpus() -> int:
    """
    Get the total number of CPUs available.
    """
    return os.cpu_count()
