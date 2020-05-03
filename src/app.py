import pickle
from os import path

import spacy
import falcon
from falcon_cors import CORS

import conf
import entailment
import articles

import log
logger = log.init_stream_logger(__name__)


class HealthResource:

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.body = "OK"


def create_app(config: conf.Config):

    ############
    # Articles #
    ############
    article_repo = articles.repository.ArticleRepository(config.ES_ENDPOINT)
    article_svc = articles.service.ArticleService(article_repo)
    article_resource = articles.endpoint.ArticleResource(article_svc)

    ##############
    # Entailment #
    ##############
    with open(path.join(config.DATA_FOLDER, config.LOG_REG_MODEL), 'rb') as f:
        logreg = pickle.load(f)

    nlp = spacy.load(path.join(config.DATA_FOLDER, config.SPACY_NLP_MODEL))

    claim_analyser = entailment.nlp.claim.ClaimSimilarityAnalyser(nlp, logreg)
    claim_extractor = entailment.nlp.claim.ClaimExtractor(nlp)
    entailment_model = entailment.nlp.model.TextualEntailmentModel(
        config.DATA_FOLDER,
        [config.ALLNLI_TRAIN_PATH, config.ALLNLI_DEV_PATH],
        [config.RTE_TRAIN_PATH, config.RTE_TEST_PATH],
        config.ENTAILMENT_MODEL,
        config.WORD_EMBEDDINGS,
    )
    entailment_engine = entailment.nlp.engine.EntailmentEngine(entailment_model)
    entailment_svc = entailment.service.EntailmentService(
        claim_analyser,
        claim_extractor,
        entailment_engine,
        article_svc
    )
    entailment_resource = entailment.endpoint.EntailmentResource(entailment_svc)

    cors = CORS(
        allow_origins_list=[config.CORS_ORIGIN],
        allow_all_headers=True,
        allow_all_methods=True,
        allow_credentials_all_origins=True
    )
    app = falcon.API(middleware=[cors.middleware])
    app.add_route("/api/healthz", HealthResource())
    app.add_route("/api/articles/{id}", article_resource)
    app.add_route("/api/articles", article_resource, suffix="collection")
    app.add_route("/api/entailment", entailment_resource)

    return app
