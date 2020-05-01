import pickle
import falcon
import spacy

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
    with open(config.LOG_REG_MODEL, 'rb') as f:
        logreg = pickle.load(f)

    nlp = spacy.load(config.NLP_MODEL)

    claim_analyser = entailment.nlp.claim.ClaimSimilarityAnalyser(nlp, logreg)
    claim_extractor = entailment.nlp.claim.ClaimExtractor()
    model = entailment.nlp.model.TextualEntailmentModel()
    entailment_engine = entailment.nlp.engine.EntailmentEngine(model)
    entailment_svc = entailment.service.EntailmentService(
        claim_analyser,
        claim_extractor,
        entailment_engine,
        article_svc
    )
    entailment_resource = entailment.endpoint.EntailmentResource(entailment_svc)

    app = falcon.API()

    app.add_route("/api/healthz", HealthResource())
    app.add_route("/api/articles/{id}", article_resource)
    app.add_route("/api/articles", article_resource, suffix="collection")
    app.add_route("/api/entailment", entailment_resource)

    return app
