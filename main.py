from __future__ import annotations

from src.pipeline import Pipeline
from utils.setup_env import setup_project_env

if __name__ == '__main__':
    project_dir, config, set_log = setup_project_env()
    pipeline = Pipeline(config)
    pipeline.main()
