from aud2vec import Seq2SeqAutoencoder
from audio_dataset import AudioMFCCDataset

import torch
from torch.utils.data import DataLoader

import argparse, os


DEVICE_ID = "cuda:0"

def main():
	parser=argparse.ArgumentParser()
	parser.add_argument("--model_save_path", type = str, default = "./saved_models", help = "path to save the model")
	parser.add_argument("--model_checkpoint_path", type = str, default = "./saved_models", help = "path to save the model")
	parser.add_argument("--data_path", "-d", type = str, default = "./", help = "path to computed mfcc numpy files")
	parser.add_argument("--batch_size", type=int, default=1000, help="batch size")
	parser.add_argument("--epochs", type=int, default=500, help="additional epochs to run")
	parser.add_argument("--epochs_per_save", type=int, default=100, help="number of epochs per save")
	parser.add_argument("--init_lr", type=float, default=0.1, help="initial learning rate")
	parser.add_argument("--noise", type=int, default=0.0, help="with what probability to add noise to augment training")
	args=parser.parse_args()

	BATCH_SIZE = args.batch_size
	epochs = args.epochs

	curr_dir = os.getcwd()
	try:
		os.chdir(args.model_save_path)
	except OSError:
		os.mkdir(args.model_save_path)
	os.chdir(curr_dir)
	
	data_path = args.data_path
	dataset = AudioMFCCDataset(aud_mfcc_file=os.path.join(data_path, "train_aud_mfcc_std_trimmed.npy"), 
								aud_mfcc_lengths=os.path.join(data_path,"train_sample_lengths_std_trimmed.npy"))
	print("loaded dataset")
	loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=AudioMFCCDataset.pack_batch)
	print("created iter on loaded data")
	
	dev_dataset = AudioMFCCDataset(aud_mfcc_file=os.path.join(data_path, "dev_aud_mfcc_trimmed.npy"), 
								aud_mfcc_lengths=os.path.join(data_path,"dev_sample_lengths_trimmed.npy"))
	print("loaded dev dataset")
	dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=AudioMFCCDataset.pack_batch)
	print("created iter on loaded dev data")

	seq2seq = Seq2SeqAutoencoder(dataset.num_features(), noise_prob=0.0)

	checkpoint = torch.load(args.model_checkpoint_path)
	seq2seq.load_state_dict(checkpoint["model_state_dict"])

	device = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
	gpu_seq2seq = seq2seq.to(device)

	print("model created")
	
	print("training resumes")
	losses = gpu_seq2seq.resume_train(checkpoint, loader, dev_loader, epochs, args.epochs_per_save, args.model_save_path, args.init_lr)

	torch.save(seq2seq.state_dict(), os.path.join(args.model_save_path, "aud2vec.pth"))

if __name__ == "__main__":
	main()
