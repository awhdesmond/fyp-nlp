# Pinocchio API Server
Backend server for pinocchio.

## Usage
```
usage: server.py [-h] [--app-name APP_NAME]
                 [--env-name {LOCAL,DEV,TEST,STAGING,PROD}]
                 [--env-path ENV_PATH] [--bind BIND]
                 [--worker-class WORKER_CLASS] [--workers WORKERS]
                 [--access-control-allow-origin ACCESS_CONTROL_ALLOW_ORIGIN]
                 [--access-control-allow-methods ACCESS_CONTROL_ALLOW_METHODS]
                 [--access-control-allow-credentials ACCESS_CONTROL_ALLOW_CREDENTIALS]
                 [--access-control-allow-headers ACCESS_CONTROL_ALLOW_HEADERS]

Runs the App

optional arguments:
  -h, --help            show this help message and exit

General:
  --app-name APP_NAME   Application and process name (default app)
  --env-name {LOCAL,DEV,TEST,STAGING,PROD}
                        Environment to run as (default LOCAL)
  --env-path ENV_PATH   Environment to run as (default config/.env)

Gunicorn:
  --bind BIND           The socket to bind. (default 0.0.0.0:5000)
  --worker-class WORKER_CLASS
                        The type of workers to use. (default
                        egg:meinheld#gunicorn_worker)
  --workers WORKERS     The number of worker processes for handling requests.
                        0 means using the following formula: CPU cores*2+1.
                        (default 0)

Middleware:
  --access-control-allow-origin ACCESS_CONTROL_ALLOW_ORIGIN
                        (default *)
  --access-control-allow-methods ACCESS_CONTROL_ALLOW_METHODS
                        (default GET, PUT, POST, DELETE, HEAD, PATCH)
  --access-control-allow-credentials ACCESS_CONTROL_ALLOW_CREDENTIALS
                        (default true)
  --access-control-allow-headers ACCESS_CONTROL_ALLOW_HEADERS
                        (default Origin, Authorization, Content-Type,
                        X-Requested-With)
```