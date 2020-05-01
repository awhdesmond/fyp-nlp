import multiprocessing
import gunicorn.app.base
from dotenv import load_dotenv

import app
import cli
import conf

import log
logger = log.init_stream_logger(__name__)


class Application(gunicorn.app.base.BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super(Application, self).__init__()

    def load_config(self):
        config = dict(
            [
                (key, value)
                for key, value in self.options.items()
                if key in self.cfg.settings and value is not None
            ]
        )
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


if __name__ == "__main__":
    args = cli.parse_cli_args()
    config = conf.Config.load_config(args.env_path)
    falcon_app = app.create_app(config)

    default_workers = (multiprocessing.cpu_count() * 2) + 1
    opts = {
        "bind": args.bind,
        "proc_name": args.app_name,
        "worker_class": args.worker_class,
        "workers": args.workers or default_workers,
    }

    logger.info(f"Server listening on: {args.bind}")
    Application(falcon_app, opts).run()
