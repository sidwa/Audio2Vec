import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class AudioMFCCDataset(Dataset):

	def __init__(self, 
			aud_mfcc_file="/shared/kgcoe-research/mil/multi_modal_instance/new_data/f8k_precomp/aud_mfcc.npy",
			aud_mfcc_lengths="/shared/kgcoe-research/mil/multi_modal_instance/new_data/f8k_precomp/sample_lengths.npy", 
			data_size=None):
		"""
			inits dataset object using numpy file to load mfcc features of audio files.

			:param aud_mfcc_file: <path/to/numpy/file.npy>  the numpy file containing mfcc features. 
															must have shape tuple of length 3!
															i.e. the numpy file should be a 3-D array with 
															shape (seq_len, batch, n_mfcc)

			:param aud_mfcc_lengths: <path/to/numpy/file.npy> the numpy file containing the time length of each 
															  audio file's mfcc features. Needed for packing unpacking
															  in RNNs
															  Must have shape (batch,)
		"""
		self.mfcc_features = np.load(aud_mfcc_file)
		self.lengths = np.load(aud_mfcc_lengths)

		if data_size is not None:
			self.mfcc_features = self.mfcc_features[:,:data_size, :]
			self.lengths = self.lengths[:data_size]

		# for data parallelism
		self.max_length = np.max(self.lengths)

	def __len__(self):
		return self.mfcc_features.shape[1]

	def num_features(self):
		return self.mfcc_features.shape[2]

	def __getitem__(self, idx):
		return self.mfcc_features[:, [idx], :], self.lengths[idx] 


	@staticmethod
	def pack_batch(sample_list):
		"""
			packs a batch in 1 tensor. instead of a list of tensors as returned by the dataloader by default
		"""
		#print("sample_list:", sample_list)
		np_samples = [sample[0] for sample in sample_list]
		len_tensor = [sample[1] for sample in sample_list]
		#print(np_samples)
		#print(np_samples[0].shape)
		numpy_packed_batch = np.concatenate(np_samples, axis=1)
		packed_batch = torch.tensor(numpy_packed_batch)

		return packed_batch, torch.tensor(len_tensor)

def main():

	dataset = AudioMFCCDataset(aud_mfcc_file="train_aud_mfcc_trimmed.npy", aud_mfcc_lengths="train_sample_lengths_trimmed.npy")

	print(dataset.mfcc_features)

	loader = DataLoader(dataset, batch_size=2, collate_fn=AudioMFCCDataset.pack_batch)

	for data in loader:

		print(data[1].shape)
		#print(AudioMFCCDataset.pack_batch(data))


if __name__ == "__main__":
	main()