"""Size the native thread pools to the CPUs this process may use, before numpy, LightGBM or XGBoost load.

OpenMP and OpenBLAS start one thread per host core. In a container limited to a fraction of a CPU (Render's
free plan) those threads exhaust the quota at once and the container is paused for most of every period,
so a run that takes seconds on a laptop never finishes. `N_JOBS` overrides the detected value."""
import os


def _cpu_quota() -> int | None:
    try:  # cgroup v2: "<quota> <period>" or "max <period>"
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        return None if quota == "max" else max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        return max(1, quota // period) if quota > 0 else None
    except (OSError, ValueError):
        return None


_budget = os.getenv("N_JOBS") or (str(q) if (q := _cpu_quota()) else None)
if _budget:
    for _var in ("N_JOBS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_var, _budget)
