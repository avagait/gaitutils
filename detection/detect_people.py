##########################################
# TODO: AUTHORSHIP
# This code is part of GaitUtils
##########################################

import tensorflow as tf
import numpy as np
import cv2
import argparse
import glob
import os
import pickle

# Detection class.
class DetectionModel(object):
	def __init__(self, model_path):
		"""Creates and loads pretrained deeplab model."""
		self.graph = tf.Graph()
		with tf.gfile.GFile(model_path, 'rb') as fid:
			od_graph_def = tf.GraphDef.FromString(fid.read())

		with self.graph.as_default():
			tf.import_graph_def(od_graph_def, name='')

		self.sess = tf.Session(graph=self.graph)

	def run(self, image):
		"""Runs inference on a single image.

		Args:
			image: A PIL.Image object, raw input image.

		Returns:
			resized_image: RGB image resized from original input image.
			seg_map: Segmentation map of `resized_image`.
		"""
		# Get handles to input and output tensors
		ops = self.graph.get_operations()
		all_tensor_names = {output.name for op in ops for output in op.outputs}
		tensor_dict = {}
		for key in [
			'num_detections', 'detection_boxes', 'detection_scores',
			'detection_classes', 'detection_masks'
		]:
			tensor_name = key + ':0'
			if tensor_name in all_tensor_names:
				tensor_dict[key] = self.graph.get_tensor_by_name(
					tensor_name)

		image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

		# Run inference
		output_dict = self.sess.run(tensor_dict,
		                            feed_dict={image_tensor: np.expand_dims(image, 0)})

		# all outputs are float32 numpy arrays, so convert types as appropriate
		output_dict['num_detections'] = int(output_dict['num_detections'][0])
		output_dict['detection_classes'] = output_dict[
			'detection_classes'][0].astype(np.uint8)
		output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
		output_dict['detection_scores'] = output_dict['detection_scores'][0]

		return output_dict

# Prepare input
# Input arguments
parser = argparse.ArgumentParser(description='Build Optical Flow dataset. Note that no data augmentation is applied')

parser.add_argument('--videodir', type=str, required=False,
                    default='/home/GAIT_local/Datasets/TUM_GAID/video',
					help='Full path to original videos directory')

parser.add_argument('--outdir', type=str, required=False,
                    default='/home/GAIT_local/TUM_GAID_bb/',
                    help="Full path for output files.")

parser.add_argument('--fpatt', type=str, required=False,
                    default='*.avi',
                    help="Video file pattern.")

args = parser.parse_args()

videosdir = args.videodir
outdir = args.outdir
fpatt = args.fpatt

if not os.path.exists(outdir):
	os.makedirs(outdir)

videos = glob.glob(os.path.join(videosdir, fpatt))
model = DetectionModel('detection_model/frozen_inference_graph.pb')

print("* Found {} videos.".format(len(videos)))
for i in range(len(videos)):
	# VideoCapture to extract the frames.
	cap = cv2.VideoCapture(videos[i])

	videoname, ext = os.path.splitext(os.path.basename(videos[i]))

	boxes = []
	scores = []
	frames = []
	frame_ix = 0
	while (cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
			output_dict = model.run(frame)

			# Remove zero-score detections.
			boxes_ = []
			scores_ = []
			boxes_temp = output_dict['detection_boxes']
			classes_temp = output_dict['detection_classes']
			scores_temp = output_dict['detection_scores']
			for i in range(boxes_temp.shape[0]):
				if scores_temp[i] > 0 and classes_temp[i] == 1: # Only persons
					boxes_.append(boxes_temp[i])
					scores_.append(scores_temp[i])

			boxes.append(np.asarray(boxes_))
			scores.append(np.asarray(scores_))
			frames.append(frame_ix)
		else:
			break

		frame_ix = frame_ix + 1
	# Write output file
	outpath = os.path.join(outdir, videoname + '.pkl')
	with open(outpath, 'wb') as output:
		pickle.dump([boxes, scores, frames], output, pickle.HIGHEST_PROTOCOL)

print("Done!")
