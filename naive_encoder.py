import argparse, os

import numpy as np

from tqdm import tqdm

from audio_dataset import AudioMFCCDataset
from torch.utils.data import DataLoader


def main():
	parser=argparse.ArgumentParser()
	parser.add_argument("--embed_save_path", "-s", type = str, default = "./", help = "path to save the model")
	parser.add_argument("--data_path", "-d", type = str, default = "./", help = "path to computed mfcc numpy files")
	parser.add_argument("--mode", type=str, default="train", help="train, dev or test")
	args=parser.parse_args()

	curr_dir = os.getcwd()
	try:
		os.chdir(args.embed_save_path)
	except OSError:
		os.mkdir(args.embed_save_path)
	os.chdir(curr_dir)

	data_path = args.data_path
	
	data = np.load(os.path.join(data_path, args.mode+"_aud_mfcc_trimmed.npy") )
	mfcc_lengths = np.load(os.path.join(data_path, args.mode+"_sample_lengths_trimmed.npy"))

	print("loaded data")

	NUM_SUB_SAMPLES = 100.0
	embed_vecs=None
	for aud_sample_idx in tqdm(range(data.shape[1])):
		
		aud_encode = None
		
		min_timestep = 0
		step = int(mfcc_lengths[aud_sample_idx] / NUM_SUB_SAMPLES)
		#print("len", mfcc_lengths[aud_sample_idx])
		#print("step size", step)
		max_timestep = int(min(min_timestep + step, mfcc_lengths[aud_sample_idx]))
		#print("steps " + str(min_timestep) + "," + str(max_timestep))
		num_sub_samples=0
		if mfcc_lengths[aud_sample_idx] < NUM_SUB_SAMPLES:
			aud_encode = data[:int(NUM_SUB_SAMPLES), aud_sample_idx, :]
			#print("aud_encode shape(if)", aud_encode.shape)
		else:
			while num_sub_samples < NUM_SUB_SAMPLES:
				if aud_encode is None:
					aud_encode = np.mean(data[min_timestep:max_timestep, aud_sample_idx, :], axis=0)
				else:
					if num_sub_samples == NUM_SUB_SAMPLES - 1:
						aud_encode = np.vstack((aud_encode, np.mean(data[min_timestep:, aud_sample_idx, :], axis=0)))
						# aud_encode = np.concatenate((aud_encode, np.mean(data[min_timestep:, aud_sample_idx, :], axis=0)), axis=0)
					else:
						# print("vstack" + str(aud_encode.shape))
						# print("mean", np.mean(data[min_timestep:max_timestep, aud_sample_idx, :], axis=0).shape)
						# print(np.mean(data[min_timestep:max_timestep, aud_sample_idx, :]))
						aud_encode = np.vstack((aud_encode, np.mean(data[min_timestep:max_timestep, aud_sample_idx, :], axis=0)))
						# aud_encode = np.concatenate((aud_encode, np.mean(data[min_timestep:max_timestep, aud_sample_idx, :], axis=0)), axis=0)
				
				num_sub_samples += 1
				
				min_timestep += step
				max_timestep += step

				#print("aud_encode shape(loop)", aud_encode.shape)
		#aud_encode = aud_encode.reshape((1,-1))
		aud_encode = aud_encode.reshape(aud_encode.shape[0], 1, aud_encode.shape[1])
		#print("aud_encode shape", aud_encode.shape)
		if embed_vecs is None:
			embed_vecs = np.copy(aud_encode)
		else:
			#embed_vecs = np.concatenate((embed_vecs, aud_encode), axis=0)
			embed_vecs = np.concatenate((embed_vecs, aud_encode), axis=1)
			#print("embed_vecs shape(mid):", embed_vecs.shape)
	
	print("embed_vecs shape:", embed_vecs.shape)
	print("original data shape", data.shape)

	np.save(os.path.join(args.embed_save_path, args.mode+"_aud_mfcc_sub.npy"), embed_vecs)
	mfcc_lengths[mfcc_lengths>=NUM_SUB_SAMPLES] = NUM_SUB_SAMPLES
	np.save(os.path.join(args.embed_save_path, args.mode+"_sample_lengths_sub.npy"), mfcc_lengths)

if __name__ == "__main__":
	main()