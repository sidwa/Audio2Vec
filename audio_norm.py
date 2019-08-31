import numpy as np
import pdb

file = np.load('/shared/kgcoe-research/mil/multi_modal_instance/new_data/f8k_precomp/train_aud_40_unnorm.npy')
min = np.min(file)
mean = np.mean(file)
# pdb.set_trace()
max = np.max(file)
# min = min.reshape((30000,1))
# max = max.reshape((30000,1))
numerator = np.subtract(file, min)
denominator = np.subtract(max,min)

final = np.divide(numerator,denominator)
np.save('/shared/kgcoe-research/mil/multi_modal_instance/new_data/f8k_precomp/train_aud.npy',final)


file = np.load('/shared/kgcoe-research/mil/multi_modal_instance/new_data/f8k_precomp/test_aud_40_unnorm.npy')
min = np.min(file)
max = np.max(file)
# min = min.reshape((5000,1))
# max = max.reshape((5000,1))
numerator = np.subtract(file, min)
denominator = np.subtract(max,min)

final = np.divide(numerator,denominator)
np.save('/shared/kgcoe-research/mil/multi_modal_instance/new_data/f8k_precomp/test_aud.npy',final)
# pdb.set_trace()