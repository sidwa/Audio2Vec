"""
Preprocessing code, possible preprocessing that could be applied on data
"""
import numpy as np

def sub_sample(train_dat, test_dat, dev_dat):
	"""

	"""
	dat_mean = np.mean(train_dat, axis=(0,1))
	dat_std = np.std(train_dat, axis=(0,1))

	train_dat = ( train_dat - dat_mean ) / dat_std
	test_dat = ( test_dat - dat_mean ) / dat_std
	dev_dat = ( dev_dat - dat_mean ) / dat_std

	return train_dat, test_dat, dev_dat

def update_pad_val(dat, dat_length, pad_val=-6.0):
	"""
		Every sample is padded after the samples are over. This is to change
		the padded value from 0 to some other value which is not the average.
		Should be used after mean_normalize()

		:param train_dat: train data
		:param dat_len: list containing number of MFCC vectors extracted for each audio samples 
		:param pad_val: padding value to be used.

		:return:
		 returns the data with updated pad value.
	"""

	for sample_idx in range(dat.shape[1]):
		sample_len = dat_length[sample_idx]

		dat[sample_len:, sample_idx, :] = pad_val

	return dat

def mean_normalize(train_dat, test_dat, dev_dat):
	"""
		Mean subtraction and std normalization of all data based on mean and std of 
		train split of dataset.

		:param train_dat: train data
		:param test_dat: test data
		:param dev_dat: validation data

		:return:
		 returns the mean normalized data splits.
	"""

	dat_mean = np.mean(train_dat, axis=(0,1))
	dat_std = np.std(train_dat, axis=(0,1))

	train_dat = ( train_dat - dat_mean ) / dat_std
	test_dat = ( test_dat - dat_mean ) / dat_std
	dev_dat = ( dev_dat - dat_mean ) / dat_std

	print("training data mean", np.mean(train_dat, axis=(0,1)))
	print("training data std", np.std(train_dat, axis=(0,1)))

	return train_dat, test_dat, dev_dat


def load_data():
	train_dat = np.load("train_aud_mfcc_trimmed.npy")
	test_dat = np.load("test_aud_mfcc_trimmed.npy")
	dev_dat = np.load("dev_aud_mfcc_trimmed.npy")

	print("data loaded")

	train_len = np.load("train_sample_lengths_trimmed.npy")
	test_len = np.load("test_sample_lengths_trimmed.npy")
	dev_len = np.load("dev_sample_lengths_trimmed.npy")

	return train_dat, test_dat, dev_dat, train_len, test_len, dev_len

def save_data(train_dat, test_dat, dev_dat, train_len, test_len, dev_len):
	np.save("train_aud_mfcc_norm.npy", train_dat)
	np.save("test_aud_mfcc_norm.npy", test_dat)
	np.save("dev_aud_mfcc_norm.npy", dev_dat)

	np.save("train_sample_lengths_norm.npy", train_len)
	np.save("test_sample_lengths_norm.npy", test_len)
	np.save("dev_sample_lengths_norm.npy", dev_len)

def main():

	train_dat, test_dat, dev_dat, train_len, test_len, dev_len = load_data()
	print("lengths loaded")

	train_dat, test_dat, dev_dat = mean_normalize(train_dat, test_dat, dev_dat)

	# applying apt pad values
	train_dat = update_pad_val(train_dat, train_len, pad_val=-10.0)
	test_dat = update_pad_val(test_dat, test_len, pad_val=-10.0)
	dev_dat = update_pad_val(dev_dat, dev_len, pad_val=-10.0)

	save_data(train_dat, test_dat, dev_dat, train_len, test_len, dev_len)

if __name__ == "__main__":
	main()