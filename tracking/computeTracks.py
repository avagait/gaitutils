###################################################################################
# This file is part of GaitUtils.
# This code is offered without any warranty or support for only research purposes.
#
# If you either use this code or find useful this repository, please, cite any of the following related works:
# [A] Francisco M. Castro, Manuel J. Marín-Jiménez, Nicolás Guil, Santiago Lopez Tapia, Nicolas Pérez de la Blanca:
#     Evaluation of Cnn Architectures for Gait Recognition Based on Optical Flow Maps. BIOSIG 2017: 251-258
# [B] Rubén Delgado-Escaño, Francisco M. Castro, Julián Ramos Cózar, Manuel J. Marín-Jiménez, Nicolás Guil:
#     MuPeG - The Multiple Person Gait Framework. Sensors 20(5): 1358 (2020)
# [C] Francisco M. Castro, Manuel J. Marín-Jiménez, Nicolás Guil, Nicolás Pérez de la Blanca:
#     Multimodal feature fusion for CNN-based gait recognition: an empirical comparison. Neural Comput. Appl. 32(17): 14173-14193 (2020)
###################################################################################

import glob
import os
import pickle
import cv2
import numpy as np
import tensorflow as tf
import argparse


def computeIoU(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

# Prepare input
# Input arguments
parser = argparse.ArgumentParser(description='Build Optical Flow dataset. Note that no data augmentation is applied')

parser.add_argument('--videodir', type=str, required=False,
                    default='/home/GAIT_local/Datasets/TUM_GAID/video',
					help='Full path to original videos directory')

parser.add_argument('--bbdir', type=str, required=False,
                    default='/home/GAIT_local/TUM_GAID_bb/',
                    help='Full path to bb directory')

parser.add_argument('--outdir', type=str, required=False,
                    default='/home/GAIT_local/TUM_GAID_tr/',
                    help="Full path for output files.")

parser.add_argument('--threshold', type=float, required=False,
                    default=0.5,
                    help="Threshold for joining detections according to their euclidean distance.")

parser.add_argument('--fpatt', type=str, required=False,
                    default='*.avi',
                    help="Video file pattern.")

args = parser.parse_args()

videosdir = args.videodir
bbdir = args.bbdir
outdir = args.outdir
threshold = args.threshold
fpatt = args.fpatt

if not os.path.exists(outdir):
	os.makedirs(outdir)

videos = glob.glob(os.path.join(videosdir, fpatt))
model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', pooling='avg')

print("* Found {} videos.".format(len(videos)))
for i in range(len(videos)):
	# VideoCapture to extract the frames.
	cap = cv2.VideoCapture(videos[i])

	videoname, ext = os.path.splitext(os.path.basename(videos[i]))
	parts = videoname.split('-')

	# Load BBs.
	[boxes, scores_bb, frames_bb] = pickle.load(open(os.path.join(bbdir, videoname + '.pkl'), "rb"))

	# Read until video is completed
	frame_ix = 0
	scaled_bbs = []
	crops = []
	frames = []
	while (cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
			# Get size of the images.
			im_height = frame.shape[0]
			im_width = frame.shape[1]

			# Get bb for the current frame.
			bb = boxes[frames_bb.index(frame_ix)]
			score = scores_bb[frames_bb.index(frame_ix)]

			scaled_bbs_ = []
			crops_ = []
			if len(bb) > 0:  # Deal with empty detections
				poss = np.argmax(score)
				if score[poss] > 0.75:
					ymin1 = int(max(bb[poss][0], 0.0) * im_height)
					xmin1 = int(max(bb[poss][1], 0.0) * im_width)
					ymax1 = int(min(bb[poss][2], 1.0) * im_height)
					xmax1 = int(min(bb[poss][3], 1.0) * im_width)
					crops_.append(np.expand_dims(cv2.resize(frame[ymin1:ymax1, xmin1:xmax1, :], (224, 224)) - [123.68, 116.78, 103.94], 0))
					scaled_bbs_.append(np.array([ymin1, xmin1, ymax1, xmax1]))
					scaled_bbs.append(scaled_bbs_)
					crops.append(crops_)
				else:
					scaled_bbs.append([])
					crops.append([])
			else:
				scaled_bbs.append([])
				crops.append([])

			frames.append(frame_ix)
			frame_ix = frame_ix + 1
		else:
			break

	# Find tracks
	track_bbs = []
	track_feats = []
	frames_ = []
	empty = True
	for j in range(len(scaled_bbs)):
		if len(scaled_bbs[j]) > 0:
			# Get features for each crop
			feats = model.predict(np.vstack(crops[j]), batch_size=64)
			norms = np.linalg.norm(feats, axis=1)
			feats = feats / norms[:, None]

			if empty:
				# First frame so we create the first tracks.
				for k in range(len(scaled_bbs[j])):
					track_bbs.append([])
					track_bbs[k].append(scaled_bbs[j][k])
					track_feats.append([])
					track_feats[k].append(feats[k, :])
					frames_.append([])
					frames_[k].append(frames[j])
				empty = False
			else:
				# Compute distances between the new frame and the previous frame of each track.
				dists = np.zeros((len(scaled_bbs[j]), len(track_bbs)))

				for bb_ix in range(len(scaled_bbs[j])):
					for k in range(len(track_bbs)):
						dists[bb_ix, k] = np.linalg.norm(feats[bb_ix, :] - track_feats[k][-1])

				selected_bbs = np.zeros(dists.shape[0])
				selected_tracks = np.zeros(len(track_bbs))
				for bb_ix in range(dists.shape[0]):
					pos = np.argmin(dists[bb_ix, :])
					for dist_ix in range(len(dists[bb_ix, :])):
						if dist_ix == pos and dists[bb_ix, dist_ix] < threshold and selected_tracks[dist_ix] == 0:
							# Append frame to previous track.
							track_bbs[dist_ix].append(scaled_bbs[j][bb_ix])
							track_feats[dist_ix].append(feats[bb_ix, :])
							frames_[dist_ix].append(frames[j])
							selected_bbs[bb_ix] = 1
							selected_tracks[dist_ix] = 1

				for bb_ix in range(len(selected_bbs)):
					if selected_bbs[bb_ix] == 0:
						# New track
						track_bbs.append([])
						track_feats.append([])
						frames_.append([])
						track_bbs[-1].append(scaled_bbs[j][bb_ix])
						track_feats[-1].append(feats[bb_ix, :])
						frames_[-1].append(frames[j])

	# Clean short tracks
	final_tracks = []
	final_frames = []
	for j in range(len(track_bbs)):
		if len(track_bbs[j]) >= 25:
			final_tracks.append(track_bbs[j])
			final_frames.append(frames_[j])

	# Check lost frames.
	for j in range(len(final_frames)):
		all_frames = np.arange(start=final_frames[j][0], stop=final_frames[j][-1]+1)
		if len(final_frames[j]) != len(all_frames):
			poss = all_frames[np.logical_not(np.isin(all_frames, final_frames[j]))]
			lost_poss = np.where(poss > len(final_frames[j])*0.9)[0]
			if len(lost_poss) > 0:
				cut_point = np.where(final_frames[j] == poss[lost_poss[0]]-1)[0][0]
				final_frames[j] = final_frames[j][0:cut_point+1]
				final_tracks[j] = final_tracks[j][0:cut_point+1]
			else:
				print("WARN! Unconnected track: ", videoname)

	# Save tracks
	outpath = os.path.join(outdir, videoname + '.pkl')
	with open(outpath, 'wb') as output:
		pickle.dump([final_tracks, final_frames], output, pickle.HIGHEST_PROTOCOL)

print("Done!")
