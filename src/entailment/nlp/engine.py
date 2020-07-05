from entailment.nlp import model

import log
logger = log.init_stream_logger(__name__)

class EntailmentEngine:

    ENTAILMENT = 0
    CONTRADICTION = 1
    NEUTRAL = 2

    THRESHOLD = 0.6

    def __init__(self, model: model.TextualEntailmentModel):
        self.model = model
        self.model.create_model()

    def predict(self, hypothesis: str, premise: str):
        """Predict the entailment between hypothesis and premise

        Returns (prediction, score/confidence) or (None, None) if score
        is not above THRESHOLD
        """

        logger.debug(f"""
            Predicting: {hypothesis}
            Against: {premise}
        """)
        prediction = self.model.predict(premise, hypothesis)
        logger.debug(f"Prediction: {prediction}\n")

        if prediction[EntailmentEngine.ENTAILMENT] > EntailmentEngine.THRESHOLD:
            return (
                EntailmentEngine.ENTAILMENT,
                prediction[EntailmentEngine.ENTAILMENT]
            )

        if prediction[EntailmentEngine.CONTRADICTION] > EntailmentEngine.THRESHOLD:
            return (
                EntailmentEngine.CONTRADICTION,
                prediction[EntailmentEngine.CONTRADICTION]
            )

        if prediction[EntailmentEngine.NEUTRAL] > EntailmentEngine.THRESHOLD:
            return (
                EntailmentEngine.NEUTRAL,
                prediction[EntailmentEngine.NEUTRAL]
            )

        return None, None
