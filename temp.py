import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import cv2
import os
import json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


def get_my_dataset_dicts(img_dir,annotations_filename):
	json_file = os.path.join(img_dir, annotations_filename)
	with open(json_file) as f:
		imgs_anns = json.load(f)

	dataset_dicts = []
	for idx, v in enumerate(imgs_anns.values()):
		record = {}
		
		filename = os.path.join(img_dir, v["filename"])
		height, width = cv2.imread(filename).shape[:2]
		
		record["file_name"] = filename
		record["image_id"] = idx
		record["height"] = height
		record["width"] = width
	  
		annos = v["regions"]
		objs = []
		for _, anno in annos.items():
			assert not anno["region_attributes"]
			anno = anno["shape_attributes"]
			px = anno["all_points_x"]
			py = anno["all_points_y"]
			poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
			poly = [p for x in poly for p in x]

			obj = {
				"bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
				"bbox_mode": BoxMode.XYXY_ABS,
				"segmentation": [poly],
				"category_id": 0,
				"iscrowd": 0
			}
			objs.append(obj)
		record["annotations"] = objs
		dataset_dicts.append(record)
	return dataset_dicts

class Detector:

	def __init__(self):
		self.dataset_dicts = None
		self.my_dataset_metadata = None
		self.cfg = None
		self.trainer = None

	def get_dataset(self,
					coco=False,
					PATH_TO_TRAIN=None,
					NAME_TRAIN_JSON=None,
					PATH_TO_VAL=None,
					NAME_VAL_JSON=None,
					list_of_classes=None
					):

		if coco == True:
			register_coco_instances("my_dataset_train", {}, NAME_TRAIN_JSON, PATH_TO_TRAIN)
			if PATH_TO_VAL is not None:
				register_coco_instances("my_dataset_val", {}, NAME_VAL_JSON, PATH_TO_VAL)
			self.dataset_dicts = DatasetCatalog.get("my_dataset_train")
			self.my_dataset_metadata = MetadataCatalog.get("my_dataset_train")
		else:

			for d in ["train", "val"]:
				if d == "train":
					DatasetCatalog.register("my_dataset_" + d, lambda d=d: get_my_dataset_dicts(PATH_TO_TRAIN,NAME_TRAIN_JSON))
					MetadataCatalog.get("my_dataset_" + d).set(thing_classes=list_of_classes)
				else:
					DatasetCatalog.register("my_dataset_" + d, lambda d=d: get_my_dataset_dicts(PATH_TO_VAL,NAME_VAL_JSON))
					MetadataCatalog.get("my_dataset_" + d).set(thing_classes=list_of_classes)
			self.my_dataset_metadata = MetadataCatalog.get("my_dataset_train")
			self.dataset_dicts = get_my_dataset_dicts(PATH_TO_TRAIN,NAME_TRAIN_JSON)

	def visualise(self):
		for d in random.sample(self.dataset_dicts, 3):
			img = cv2.imread(d["file_name"])
			visualizer = Visualizer(img[:, :, ::-1], metadata=self.my_dataset_metadata, scale=0.5)
			vis = visualizer.draw_dataset_dict(d)
			cv2_imshow(vis.get_image()[:, :, ::-1])
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

	def set_parameters(self,
						NUM_CLASSES, 
						NUM_WORKERS = 2,
						WEIGHTS_PATH="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
						IMS_PER_BATCH = 2 ,
						BASE_LR = 0.00025,
						MAX_ITER = 1000,
						BATCH_SIZE_PER_IMAGE = 512
						 ):
		self.cfg = get_cfg()
		self.cfg.merge_from_file(model_zoo.get_config_file(WEIGHTS_PATH))
		self.cfg.DATASETS.TRAIN = ("my_dataset_train",)
		self.cfg.DATASETS.TEST = ()
		self.cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
		self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(WEIGHTS_PATH)  # Let training initialize from model zoo
		self.cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
		self.cfg.SOLVER.BASE_LR = BASE_LR
		self.cfg.SOLVER.MAX_ITER = MAX_ITER
		self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE_PER_IMAGE 
		self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES  

	def train(self):

		os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
		self.trainer = DefaultTrainer(self.cfg) 
		self.trainer.resume_or_load(resume=False)
		self.trainer.train()

	def infer(self,img_path):
		self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
		self.cfg.DATASETS.TEST = ("my_dataset_val", )
		predictor = DefaultPredictor(self.cfg)

		
		im = cv2.imread(img_path)
		outputs = predictor(im)
		v = Visualizer(im[:, :, ::-1],
						   metadata=self.my_dataset_metadata, 
						   scale=0.8, 
						   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
			)
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		cv2_imshow(v.get_image()[:, :, ::-1])

	def get_eval_score(self):
		evaluator = COCOEvaluator("my_dataset_val", self.cfg, False, output_dir="./output/")
		val_loader = build_detection_test_loader(self.cfg, "my_dataset_val")
		inference_on_dataset(self.trainer.model, val_loader, evaluator)