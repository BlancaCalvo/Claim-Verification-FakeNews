
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")

# load data in gear format

# parse both the claim and the evidence with SRL
prediction = predictor.predict(
  sentence="Did Uriah honestly think he could beat the game in under three hours?"
)

# see output
print(prediction['verbs'][0])
