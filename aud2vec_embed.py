from aud2vec import Seq2SeqAutoencoder
from audio_dataset import AudioMFCCDataset

import torch
from torch.utils.data import DataLoader
import numpy as np

import argparse, os

def main():
	parser=argparse.ArgumentParser()
	parser.add_argument("--embed_save_path", type = str, default = "./precomp_data", help = "path to save the model")
	parser.add_argument("--model_checkpoint_path", type = str, default = "./saved_models", help = "path to save the model")
	parser.add_argument("--data_path", "-d", type = str, default = "./", help = "path to computed mfcc numpy files")
	parser.add_argument("--mode", type=str, default="train", help="train, test or dev mode?")
	parser.add_argument("--batch_size", type=int, default=5000, help="batch size")
	args=parser.parse_args()

	BATCH_SIZE = args.batch_size

	curr_dir = os.getcwd()
	try:
		os.chdir(args.embed_save_path)
	except OSError:
		os.mkdir(args.embed_save_path)
	os.chdir(curr_dir)
	
	data_path = args.data_path
	dataset = AudioMFCCDataset(aud_mfcc_file=os.path.join(data_path, args.mode+"_aud_mfcc_norm.npy"), 
								aud_mfcc_lengths=os.path.join(data_path, args.mode+"_sample_lengths_norm.npy"))
	print("loaded dataset")
	loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=AudioMFCCDataset.pack_batch)
	print("created iter on loaded data")
	
	seq2seq = Seq2SeqAutoencoder(dataset.num_features(), noise_prob=0.0)

	checkpoint = torch.load(args.model_checkpoint_path)
	seq2seq.load_state_dict(checkpoint["model_state_dict"])

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	gpu_seq2seq = seq2seq.to(device)
	gpu_seq2seq.encoder.eval()
	gpu_seq2seq.decoder.eval()
	print("model created")
	
	embed_vecs = None
	for input_dat, lengths in loader:
		input_dat = input_dat.to(device)
		lengths = lengths.to(device)
		
		if embed_vecs is None:
			embed_vecs = gpu_seq2seq.embed(input_dat, lengths)
			print(embed_vecs.shape)
		else:
			embed_vecs = np.concatenate( (embed_vecs, gpu_seq2seq.embed(input_dat, lengths)) )

	print(embed_vecs.shape)
	np.save(os.path.join(args.embed_save_path, args.mode+"_aud_emb.npy"), embed_vecs)

if __name__ == "__main__":
	main()
