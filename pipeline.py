from __future__ import annotations

import logging

from utils.setup_env import setup_project_env


class Test:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def main(self):
        self.logger.info('This is a test')
        print('This is a test')


if __name__ == '__main__':
    project_dir, config, setup_logs = setup_project_env()
    test = Test()
    test.main()
