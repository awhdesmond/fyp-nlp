# Pinocchio Deployment Steps
```
git clone https://github.com/awhdesmond/fyp-newsbot
git clone https://github.com/awhdesmond/fyp-web
git clone https://github.com/awhdesmond/fyp-nlp
```

## Update `.env`
Inside `fyp-nlp`, there is a config folder with a sample environment file `.env.sample`. Copy this file and rename it to `.env`. Then update the neceesary configurations:
```
ALLNLI_DEV_PATH=AllNLI/dev.jsonl
ALLNLI_TRAIN_PATH=AllNLI/train.jsonl
CORS_ORIGIN=???
DATA_FOLDER=DATA_FOLDER
ENTAILMENT_MODEL=glove-full-glove-full-adam-checkpoint-weights.05-0.80.hdf5
ES_ENDPOINT=elasticsearch:9200
LOG_REG_MODEL=logreg.model
RTE_TEST_PATH=rte_1.0/rte_test.json
RTE_TRAIN_PATH=rte_1.0/rte.json
SPACY_NLP_MODEL=en_core_web_md-2.2.5
WORD_EMBEDDINGS=glove.840B.300d
```

> Note the `CORS_ORIGIN` has to be the hostname you are hosting the website

## Build Docker
```
cd fyp-web
docker build -t pinocchio-web .
cd fyp-nlp
docker build -t pinocchio-nlp .

docker network create pinocchio-network

docker run -d --name elasticsearch \
  -p 5601:5601 -p 9200:9200 \
  --network pinocchio-network \
  nshou/elasticsearch-kibana:kibana6

docker run -d --name pinocchio-web \
  -p 3000:3000 \
  --network pinocchio-network \
  pinocchio-web

docker run -d --name pinocchio-nlp \
  -p 5000:5000 \
  --network pinocchio-network \
  -v path/to/DATA_FOLDER:/pinocchio-nlp/DATA_FOLDER \
  -v path/to/.env:/pinocchio-nlp/config/.env \
  pinocchio-nlp
```

## Getting `DATA_FOLDER`
The `DATA_FOLDER` contains the models files and data files needed to build and run the NLP model.


## Setting up Elasticsearch
Once you create the elasticseaerch and kibana container, go to kibana on `hostname:5601` and find the console. Then, enter the following commands in the console to set up the elasticsearch indices.
```
PUT /articles

POST /articles/_close

PUT /articles/_settings
{
  "analysis": {
    "filter": {
      "english_stop" : {
        "type": "stop",
        "stopwords": "_english_"
      },
      "english_stemmer": {
        "type": "stemmer",
        "language": "english"
      },
      "english_possessive_stemmer": {
        "type": "stemmer",
        "language": "possessive_english"
      },
      "shingle_filter": {
        "type": "shingle",
        "min_shingle_size": 2,
        "max_shingle_size": 2,
        "output_unigrams": false
      }
    },
    "analyzer": {
      "rebuilt_english_shingle": {
        "tokenizer": "standard",
        "filter": [
          "english_possessive_stemmer",
          "lowercase",
          "english_stop",
          "english_stemmer",
          "shingle_filter"
        ]
      },
      "rebuilt_english": {
        "tokenizer": "standard",
        "filter": [
          "english_possessive_stemmer",
          "lowercase",
          "english_stop",
          "english_stemmer"
        ]
      }
    }
  }
}

PUT /articles/_mappings/_doc
{
  "properties": {
    "content": {
      "type": "text",
      "analyzer": "rebuilt_english",
      "fields": {
        "shingles": {
          "type": "text",
          "analyzer": "rebuilt_english_shingle"
        }
      }
    },
    "title": {
      "type": "text",
      "analyzer": "rebuilt_english",
      "fields": {
        "shingles": {
          "type": "text",
          "analyzer": "rebuilt_english_shingle"
        }
      }
    },
    "author": {
      "type": "text"
    },
    "source": {
      "type": "text",
      "index": false
    },
    "url": {
      "type": "text",
      "index": false
    },
    "imageurl": {
      "type": "text",
      "index": false
    },
    "publishedDate": {
      "type": "date"
    }
  }
}

POST /articles/_open

GET /articles/_doc/_search
{
  "query": {
    "match_all": {}
  }
}
```

