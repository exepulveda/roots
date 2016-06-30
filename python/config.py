
PROJECT_HOME = "."

def Configuration(object):
    pass

configuration = Configuration()

configuration.home = PROJECT_HOME
configuration.start_frame = 6
configuration.end_frame = 54

configuration.model = Configuration()
configuration.model.classifier = "models/binary.json"
configuration.model.classifier_weights = "models/binary.h5"
configuration.model.frames = "models/frames.json"
configuration.model.frames_weights = "models/frames.h5"

configuration.input = Configuration()
configuration.input.with = 384
configuration.input.height = 288

configuration.input = Configuration()
configuration.frame.with = 80
configuration.frame.height = 80

