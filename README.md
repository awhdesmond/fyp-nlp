# Pinocchio API Server
Backend server for Pinocchio Fake News Detection

## Dependencies
* Python 3.6.3
* Elassticsearch 6

## Configuration Options
|Name|Purpose|Default|
|----|-------|-------|
|ALLNLI_DEV_PATH|AllNLI Dev Dataset|AllNLI/dev.jsonl|
|ALLNLI_TRAIN_PATH|AllNLI Train Dataset|AllNLI/train.jsonl|
|CORS_ORIGIN|CORS allowed origins|localhost:3000|
|DATA_FOLDER|Base data folder|DATA_FOLDER|
|ENTAILMENT_MODEL|Entailment model weights|glove-full-glove-full-adam-checkpoint-weights.05-0.80.hdf5|
|ES_ENDPOINT|Elasticsearch Endpoint|localhost:9200|
|LOG_REG_MODEL|Logistic Regression model|logreg.model|
|RTE_TEST_PATH|RTE Test Dataset|rte_1.0/rte_test.json|
|RTE_TRAIN_PATH|RTE Train Dataset|rte_1.0/rte.json|
|SPACY_NLP_MODEL|Spacy model|en_core_web_md-2.2.5|
|WORD_EMBEDDINGS|GLoVE embeddings name|glove.840B.300d|

## Start Server
```
usage: server.py [-h] [--app-name APP_NAME]
                 [--env-name {LOCAL,DEV,TEST,STAGING,PROD}]
                 [--env-path ENV_PATH] [--bind BIND]
                 [--worker-class WORKER_CLASS] [--workers WORKERS]

Runs the App

optional arguments:
  -h, --help            show this help message and exit

General:
  --app-name APP_NAME   Application and process name (default app)
  --env-name {LOCAL,DEV,TEST,STAGING,PROD}
                        Environment to run as (default LOCAL)
  --env-path ENV_PATH   Environment to run as (default config/.env)

Waitress:
  --bind BIND           The socket to bind. (default 0.0.0.0:5000)
```