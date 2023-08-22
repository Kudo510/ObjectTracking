import collections

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import motmetrics as mm
from motmetrics import lap
lap.default_solver = 'lap'

import market.metrics as metrics
import copy

class Tracker:
	"""The main tracking file, here is where magic happens."""

	def __init__(self, obj_detect):
		self.obj_detect = obj_detect

		self.tracks = []
		self.track_num = 0
		self.im_index = 0
		self.results = {}

		self.mot_accum = None

	def reset(self, hard=True):
		self.tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def add(self, new_boxes, new_scores):
		"""Initializes new Track objects and saves them."""
		num_new = len(new_boxes)
		for i in range(num_new):
			self.tracks.append(Track(
				new_boxes[i],
				new_scores[i],
				self.track_num + i
			))
		self.track_num += num_new

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.tracks) == 1:
			box = self.tracks[0].box
		elif len(self.tracks) > 1:
			box = torch.stack([t.box for t in self.tracks], 0)
		else:
			box = torch.zeros(0).cuda()
		return box

	def data_association(self, boxes, scores):
		self.tracks = []
		self.add(boxes, scores)

	def step(self, frame):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		# object detection
		boxes, scores = self.obj_detect.detect(frame['img'])

		self.data_association(boxes, scores)

		# results
		for t in self.tracks:
			if t.id not in self.results.keys():
				self.results[t.id] = {}
			self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

		self.im_index += 1

	def get_results(self):
		return self.results


class ReIDTracker(Tracker):

	def add(self, new_boxes, new_scores, new_features):
		"""Initializes new Track objects and saves them."""
		num_new = len(new_boxes)
		for i in range(num_new):
			self.tracks.append(Track(
				new_boxes[i],
				new_scores[i],
				self.track_num + i,
				new_features[i]
			))
		self.track_num += num_new

	def reset(self, hard=True):
		self.tracks = []
		#self.inactive_tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0


	def data_association(self, boxes, scores, frame):
		self.tracks = []
		self.add(boxes, scores)

	def step(self, frame):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		# object detection
		boxes, scores = self.obj_detect.detect(frame['img'])

		self.data_association(boxes, scores, frame['img'])

		# results
		for t in self.tracks:
			if t.id not in self.results.keys():
				self.results[t.id] = {}
			self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

		self.im_index += 1

	def get_crop_from_boxes(self, boxes, frame, height=256, width=128):
		"""Crops all persons from a frame given the boxes.

		Args:
			boxes: The bounding boxes.
			frame: The current frame.
			height (int, optional): [description]. Defaults to 256.
			width (int, optional): [description]. Defaults to 128.
		"""
		person_crops = []
		norm_mean = [0.485, 0.456, 0.406] # imagenet mean
		norm_std = [0.229, 0.224, 0.225] # imagenet std
		for box in boxes:
			box = box.to(torch.int32)
			res = frame[:, :, box[1]:box[3], box[0]:box[2]]
			res = F.interpolate(res, (height, width), mode='bilinear')
			res = TF.normalize(res[0, ...], norm_mean, norm_std)
			person_crops.append(res.unsqueeze(0))

		return person_crops

	def compute_reid_features(self, model, crops):
		f_ = []
		model.eval()
		with torch.no_grad():
			for data in crops:
				img = data.cuda()
				features = model(img)
				features = features.cpu().clone()
				f_.append(features)
			f_ = torch.cat(f_, 0)
			return f_

	def ltrb_to_ltwh(self, ltrb_boxes):
		ltwh_boxes = copy.deepcopy(ltrb_boxes)
		ltwh_boxes[:, 2] = ltrb_boxes[:, 2] - ltrb_boxes[:, 0]
		ltwh_boxes[:, 3] = ltrb_boxes[:, 3] - ltrb_boxes[:, 1]

		return ltwh_boxes
	
	def compute_distance_matrix(self, track_features, pred_features, track_boxes, boxes, metric_fn, alpha=0.0):
		UNMATCHED_COST = 255.0
		# Build cost matrix.
		iou_track_boxes = self.ltrb_to_ltwh(track_boxes)
		iou_boxes = self.ltrb_to_ltwh(boxes)
		distance = mm.distances.iou_matrix(iou_track_boxes, iou_boxes.numpy(), max_iou=0.5)
		# distance = mm.distances.iou_matrix(track_boxes.numpy(), boxes.numpy(), max_iou=0.5)

		appearance_distance = metrics.compute_distance_matrix(track_features, pred_features, metric_fn=metric_fn)
		appearance_distance = appearance_distance.numpy() * 0.5
		# return appearance_distance

		assert np.alltrue(appearance_distance >= -0.1)
		assert np.alltrue(appearance_distance <= 1.1)

		combined_costs = alpha * distance + (1-alpha) * appearance_distance

		# Set all unmatched costs to _UNMATCHED_COST.
		distance = np.where(np.isnan(distance), UNMATCHED_COST, combined_costs)
		return distance


class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, box, score, track_id, feature=None, inactive=0):
		self.id = track_id
		self.box = box
		self.score = score
		self.feature = collections.deque([feature])
		self.inactive = inactive
		self.max_features_num = 10


	def add_feature(self, feature):
		"""Adds new appearance features to the object."""
		self.feature.append(feature)
		if len(self.feature) > self.max_features_num:
			self.feature.popleft()

	def get_feature(self):
		if len(self.feature) > 1:
			feature = torch.stack(list(self.feature), dim=0)
		else:
			feature = self.feature[0].unsqueeze(0)
		return feature.mean(0, keepdim=False)
