# import upsample_function
#author: Xiao Zhang/Haoyuan Zhang
import numpy as np
import pdb
def upsample2d(input, output_size):
	h , w = output_size
	c = input.shape[1]
	n = input.shape[0]
	output = np.zeros((n,c,h,w)).astype(np.float32)
	input = input.copy(order='C').astype(np.float32)
	output = output.copy(order='C').astype(np.float32)
	output = upsample_function.bilinear_forward(input, output)
	return output

def get_gt_map(gt_label, h, w):
	batch_num = gt_label.shape[0]
	mesh_x, mesh_y = np.meshgrid(np.arange(w), np.arange(h))
	label = np.zeros((batch_num,5, h,w))
	for j in range(batch_num):
		for i in range(5):
			label[j, i,:,:] = np.exp(-1./50*((mesh_x - gt_label[j,i])**2 + (mesh_y - gt_label[j,i+5])**2))
	return label