import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
# from google.colab.patches import cv2_imshow

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



def get_balloon_dicts(img_dir):
	json_file = os.path.join(img_dir, "via_region_data.json")
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

# for d in ["train", "val"]:
#     DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
#     MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
# balloon_metadata = MetadataCatalog.get("balloon_train")



# dataset_dicts = get_balloon_dicts("balloon/train")
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2_imshow(vis.get_image()[:, :, ::-1])

class Detector:

	def __init__(self):
		self.dataset_dicts = None
		self.balloon_metadata = None

	def get_dataset(self,coco=False):

		if coco == True:
			pass
		else:

			for d in ["train", "val"]:
				DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
				MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
			self.balloon_metadata = MetadataCatalog.get("balloon_train")
			self.dataset_dicts = get_balloon_dicts("balloon/train")

		register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "balloon/train")
		register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "balloon/val")


	def visualise(self):
		for d in random.sample(self.dataset_dicts, 3):
			img = cv2.imread(d["file_name"])
			visualizer = Visualizer(img[:, :, ::-1], metadata=self.balloon_metadata, scale=0.5)
			vis = visualizer.draw_dataset_dict(d)
			cv2.imshow("main",vis.get_image()[:, :, ::-1])
			cv2.waitKey(0)
			cv2.destroyAllWindows()


	def train(self):
		cfg = get_cfg()
		cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
		cfg.DATASETS.TRAIN = ("balloon_train",)
		cfg.DATASETS.TEST = ()
		cfg.DATALOADER.NUM_WORKERS = 2
		cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
		cfg.SOLVER.IMS_PER_BATCH = 2
		cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
		cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
		cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
		cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

		os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
		print("a#############################################################")
		trainer = DefaultTrainer(cfg) 
		print("b#############################################################")
		trainer.resume_or_load(resume=False)
		print("y#############################################################")
		trainer.train()


		return trainer