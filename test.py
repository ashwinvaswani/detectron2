from main import Detector

detector_object = Detector()
detector_object.get_dataset()
# detector_object.visualise()
trainer = detector_object.train()