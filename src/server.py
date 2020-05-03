import multiprocessing
from waitress import serve

import app
import cli
import conf

import log
logger = log.init_stream_logger(__name__)

if __name__ == "__main__":
    args = cli.parse_cli_args()
    config = conf.Config.load_config(args.env_path)
    falcon_app = app.create_app(config)

    default_workers = (multiprocessing.cpu_count() * 2) + 1

    logger.info(f"Server listening on: {args.bind}")
    serve(falcon_app, listen=args.bind)
