import argparse


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Runs the App")

    # General
    g = parser.add_argument_group("General")
    g.add_argument(
        "--env-name",
        type=str,
        default="LOCAL",
        choices=["LOCAL", "DEV", "TEST", "STAGING", "PROD"],
        dest="env_name",
        help="Environment to run as (default %(default)s)",
    )
    g.add_argument(
        "--env-path",
        type=str,
        default="config/.env",
        dest="env_path",
        help="Environment to run as (default %(default)s)",
    )

    # Gunicorn
    gu = parser.add_argument_group("Gunicorn")
    gu.add_argument(
        "--bind",
        type=str,
        default="0.0.0.0:5000",
        dest="bind",
        help="The socket to bind. (default %(default)s)",
    )
    gu.add_argument(
        "--workers",
        type=int,
        default=0,
        dest="workers",
        help="The number of worker processes for handling requests. "
        "0 means using the following formula: CPU cores*2+1. "
        "(default %(default)s)",
    )

    return parser.parse_args()
