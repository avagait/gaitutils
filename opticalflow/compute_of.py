###################################################################################
# This file is part of GaitUtils.
# It has been adapted from the GitHub repository 'pytorch-spynet'.
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
import cv2
import torch
import math
import numpy
import os
import os.path as osp
import argparse

######### Parameters #########
MODEL_PATH = 'pytorch-spynet/network-' + 'sintel-final' + '.pytorch'
##############################

def Backward(tensorInput, tensorFlow):
	Backward_tensorGrid = {}
	if str(tensorFlow.size()) not in Backward_tensorGrid:
		tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
		tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

		Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
	# end

	tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		class Preprocess(torch.nn.Module):
			def __init__(self):
				super(Preprocess, self).__init__()
			# end

			def forward(self, tensorInput):
				tensorBlue = (tensorInput[:, 0:1, :, :] - 0.406) / 0.225
				tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
				tensorRed = (tensorInput[:, 2:3, :, :] - 0.485) / 0.229

				return torch.cat([ tensorRed, tensorGreen, tensorBlue ], 1)
			# end
		# end

		class Basic(torch.nn.Module):
			def __init__(self, intLevel):
				super(Basic, self).__init__()

				self.moduleBasic = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
				)
			# end

			def forward(self, tensorInput):
				return self.moduleBasic(tensorInput)
			# end
		# end

		self.modulePreprocess = Preprocess()

		self.moduleBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

		self.load_state_dict(torch.load('pytorch-spynet/network-' + 'sintel-final' + '.pytorch'))
	# end

	def forward(self, tensorFirst, tensorSecond):
		tensorFlow = []

		tensorFirst = [ self.modulePreprocess(tensorFirst) ]
		tensorSecond = [ self.modulePreprocess(tensorSecond) ]

		for intLevel in range(5):
			if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
				tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2))
				tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2))
			# end
		# end

		tensorFlow = tensorFirst[0].new_zeros([ tensorFirst[0].size(0), 2, int(math.floor(tensorFirst[0].size(2) / 2.0)), int(math.floor(tensorFirst[0].size(3) / 2.0)) ])

		for intLevel in range(len(tensorFirst)):
			tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

			if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
			if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

			tensorFlow = self.moduleBasic[intLevel](torch.cat([ tensorFirst[intLevel], Backward(tensorInput=tensorSecond[intLevel], tensorFlow=tensorUpsampled), tensorUpsampled ], 1)) + tensorUpsampled
		# end

		return tensorFlow
	# end
# end

##########################################################

def estimate(tensorFirst, tensorSecond):
	moduleNetwork = Network().cuda().eval()
	assert(tensorFirst.size(1) == tensorSecond.size(1))
	assert(tensorFirst.size(2) == tensorSecond.size(2))

	intWidth = tensorFirst.size(2)
	intHeight = tensorFirst.size(1)

	#assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	#assert(intHeight == 416) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
	tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

	tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tensorFlow = torch.nn.functional.interpolate(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tensorFlow[0, :, :, :].cpu()
# end

##########################################################

def compute_of(frame, previous_frame):
	torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance
	#torch.cuda.device(5)  # change this if you have a multiple graphics cards and you want to utilize them
	torch.cuda.set_device(0)
	torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

	tensorFirst = torch.FloatTensor(
		previous_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
					1.0 / 255.0))
	tensorSecond = torch.FloatTensor(
		frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
					1.0 / 255.0))

	optical_flow = estimate(tensorFirst, tensorSecond)
	optical_flow = optical_flow.numpy()

	return optical_flow

# Prepare input
# Input arguments
parser = argparse.ArgumentParser(description='Build Optical Flow dataset. Note that no data augmentation is applied')

parser.add_argument('--videodir', type=str, required=False,
                    default='/home/GAIT_local/Datasets/TUM_GAID/video',
					help='Full path to original videos directory')

parser.add_argument('--outdir', type=str, required=False,
                    default='/home/GAIT_local/TUM_GAID_of/',
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
print("* Found {} videos.".format(len(videos)))
for file in videos:
	previous_frame = None
	of_video = []
	cap = cv2.VideoCapture(file)
	while cap.isOpened():
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret:
			if previous_frame is not None:
				of = compute_of(frame, previous_frame)
				of = of*100
				of_video.append(of.astype(numpy.int16))

			previous_frame = frame.copy()
		# Break the loop
		else:
			break
	cap.release()
	outname = os.path.splitext(os.path.split(file)[1])[0]
	outfilename = osp.join(outdir,  outname + '.npz')
	numpy.savez_compressed(outfilename, of=of_video)
	# For reading.
	# of = numpy.load(OUTPUT_DIR + outname + '.npz')['of']
	# of = of / 100.0

print("Done!")
