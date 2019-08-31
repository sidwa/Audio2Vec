import pickle
import numpy as np

sample_list = pickle.load(open("sample_list.pkl", "rb"))
print("shape:", len(sample_list))
#final_batch = np.concatenate(sample_list, axis=1)
lengths = []
for sample in sample_list:
	lengths.append(sample.shape[0])

print("done")

#np.save("train_aud_mfcc.npy", final_batch)
np.save("train_sample_lengths.npy", lengths)