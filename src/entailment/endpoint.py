import falcon
from common import serializers
from entailment import exceptions


class EntailmentResource:

    TEXT_PARAM = "text"

    def __init__(self, entailment_svc):
        self.entailment_svc = entailment_svc

    def on_post_entailment(self, req, resp):
        """
        HTTP POST /api/entailment
        Body parameters:
        {
            text (str): user query text
        }
        """
        payload = req.media

        if EntailmentResource.TEXT_PARAM not in payload.keys():
            raise falcon.HTTPBadRequest("Missing body parameters")

        try:
            evidence_articles = self.entailment_svc.entailment_query(
                payload.get(EntailmentResource.TEXT_PARAM),
            )

            result = dict(evidence_articles=evidence_articles)
        except exceptions.NoClaimException as e:
            result = dict(error=str(e))

        resp.body = serializers.serialize_json(result)
        resp.content_type = falcon.MEDIA_JSON
