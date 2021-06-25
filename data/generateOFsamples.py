##########################################
# TODO: AUTHORSHIP
# This code is part of GaitUtils
##########################################

import cv2
import numpy as np
import pickle
import argparse
import os
import os.path as osp
import deepdish as dd

# Prepare input
# Input arguments
parser = argparse.ArgumentParser(description='Build Optical Flow dataset. Note that no data augmentation is applied')

parser.add_argument('--ofdir', type=str, required=False,
                    default='/home/GAIT_local/TUM_GAID_of/',
					help='Full path to of directory')

parser.add_argument('--trackdir', type=str, required=False,
                    default='/home/GAIT_local/TUM_GAID_tr/',
                    help='Full path to tracks directory')

parser.add_argument('--outdir', type=str, required=False,
                    default='/tmp/ofsamples/',
                    help="Full path for output files. Note that one or more folders are created to store de files")

parser.add_argument('--dataset', type=str, required=False,
                    default='tum_gaid',
                    help="tum_gaid or casiab")

parser.add_argument('--mode', type=str, required=False,
                    default='train',
                    help="train: training subset; ft: fine-tuning subset; test: test subset")

parser.add_argument('--nframes', type=int, required=False,
                    default=25,
                    help="Number of frames to be stacked")

parser.add_argument('--step', type=int, required=False,
                    default=5,
                    help="Step size in number of frames")

parser.add_argument('--val_perc', type=float, required=False,
                    default=0.1,
                    help="Percentaje of validation samples")

parser.add_argument('--ids_file_path', type=str, required=False,
                    default='/home/GAIT_local/Datasets/TUM_GAID/labels',
                    help="Folder containing the id list files")

args = parser.parse_args()

ofdir = args.ofdir
trackdir = args.trackdir
outdir = args.outdir
dataset = args.dataset
mode = args.mode
n_frames = args.nframes
step = args.step
perc = args.val_perc
ids_file_path = args.ids_file_path

if not os.path.exists(outdir):
    os.makedirs(outdir)

# CHANGE ME, depending on your video resolution
im_width = 640
im_height = 480

# Initialize some parameters...
np.random.seed(0)
x_scale = 80 / im_width
y_scale = 60 / im_height
meanSample = 0

samplename = 'avamvg_m_tr01_cam03_clip'
id = 1  # Let's assume that this is the subject with ID 1
# of_file = os.path.join(ofdir, subject_pattern.format(id) + pattern + '.npz')
of_file = osp.join(ofdir, samplename+'.npz')
# track_file = os.path.join(trackdir, subject_pattern.format(id) + pattern + '.pkl')
track_file = osp.join(trackdir, samplename+'.pkl')

if os.path.exists(of_file) and os.path.exists(track_file):
    # Load files.
    of = np.load(of_file)['of']
    of = np.moveaxis(of, 1, -1)
    with open(track_file, 'rb') as f:
        [full_tracks, full_frames] = pickle.load(f)

    # Stack n_frames continuous frames
    if len(full_tracks) > 0:
        full_tracks = full_tracks[0]
        full_frames = full_frames[0]
        sample_id = 1
        for i in range(0, len(full_tracks), step):
            # Get data of the n_frames
            if (i + 1 + n_frames) < len(full_tracks):
                positions = full_frames[i:i + n_frames]
                ofs = of[positions, :, :, :]
                sub_position_list = full_tracks[i + 1:i + 1 + n_frames]  # We add 1 since the OF starts in 1

                # n_frames loop to compute the centroid of the  detection
                bbs = []
                centroids = []
                for j in range(len(sub_position_list)):
                    # Compute position of the BB and its centroid
                    x = int(np.round(sub_position_list[j][1] * x_scale))
                    y = int(np.round(sub_position_list[j][0] * y_scale))
                    xmax = int(np.round(sub_position_list[j][3] * x_scale))
                    ymax = int(np.round(sub_position_list[j][2] * y_scale))

                    bb_temp = [x, y, xmax, ymax]
                    centroids_temp = [(y + ymax) / 2, (x + xmax) / 2]
                    bbs.append(bb_temp)
                    centroids.append(centroids_temp)

                # Generate final of maps. Note that 30 is the central point of the frame in x-axis.
                dif_bb = 30 - centroids[round(n_frames / 2)][1]
                M = np.float32([[1, 0, dif_bb], [0, 1, 0]])
                resized_ofs = np.zeros([60, 60, 50], np.int16)
                for k in range(len(ofs)):
                    resized_of = cv2.resize(ofs[k, :, :, :], (80, 60))
                    resized_ofs[:, :, 2 * k:2 * k + 2] = cv2.warpAffine(resized_of, M, (60, 60))

                # Write output file.
                data = dict()
                data['data'] = np.int16(resized_ofs)
                data['label'] = np.uint16(id)
                # data['videoId'] = np.uint16(videoId)
                # data['gait'] = np.uint8(gaits[partition][pattern_ix])
                data['frames'] = np.uint16(positions)
                data['bbs'] = np.uint8(sub_position_list)
                data['compressFactor'] = np.uint8(100)
                meanSample = meanSample + np.uint8(resized_ofs)
                # if dataset == 'casiab':
                #     data['cam'] = int(pattern.split('-')[-1])
                # outpath = os.path.join(outdir, folders[partition],
                #                        subject_pattern.format(id) + pattern + '-{:02d}'.format(sample_id) + '.h5')
                outpath = os.path.join(outdir, samplename+'-{:02d}'.format(sample_id) + '.h5')
                dd.io.save(outpath, data)

                # Append data for the global file
                # labels_.append(id)
                # videoIds_.append(videoId)
                # gaits_.append(gaits[partition][pattern_ix])
                # bbs_.append(sub_position_list)
                # frames_.append(positions)
                # file_.append(subject_pattern.format(id) + pattern + '-{:02d}'.format(sample_id) + '.h5')
                # if dataset == 'casiab':
                #     cam_.append(int(pattern.split('-')[-1]))

                sample_id = sample_id + 1

print("Done!")