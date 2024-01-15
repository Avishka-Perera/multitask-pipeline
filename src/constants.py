import os

log_levels = [0, 1, 2, 3]  # 0: notset, 1: info, 2: warn, 3: error
analysis_levels = [
    0,
    1,
    2,
]  # 0: no analysis; 1: break loss into parts; 2: break loss into parts and analyze gradients


cache_dir = os.path.join(os.path.expanduser("~"), ".cache/mt_pipe")
model_weights_dir = os.path.join(cache_dir, "models-weights")
