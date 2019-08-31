
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
import math, random
import os, argparse
from audio_dataset import AudioMFCCDataset
from torch.utils.data import DataLoader

DEVICE_ID = "cuda:1"

class Encoder(nn.Module):
	"""
		takes in SEQ_LEN * batch * feature_len shaped tensor with main output being its hidden state
		being the audio2vec conversion. 
	"""
	def __init__(self, num_layers=1, num_units=100, num_features=150, bidirectional=False):
		super(Encoder, self).__init__()

		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.num_units = num_units
		self.bidirectional = bidirectional
		self.num_features = num_features
		self.lstm = nn.LSTM(input_size=self.num_features, hidden_size=self.num_units, 
							num_layers=self.num_layers, bidirectional=self.bidirectional)

	def forward(self, input_dat, input_lengths,hidden=None):
		input_dat = nn.utils.rnn.pack_padded_sequence(input_dat, input_lengths, enforce_sorted=False)
		output_dat, hidden_state = self.lstm(input_dat, hidden)

		return output_dat, hidden_state

class Decoder(nn.Module):
	"""
		Takes in as input, output of previous time step. Expected to match its output
		to the input MFCC features given to the encoder
	"""
	def __init__(self, num_features, output_dim, num_layers=1, num_units=100, bidirectional=False):
		"""
			num_features 100 since encoder will give a 100-dim audio2vec.
		"""
		super(Decoder, self).__init__()

		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.num_units = num_units
		self.bidirectional = bidirectional
		self.num_features = num_features
		self.lstm = nn.LSTM(input_size=self.num_features, hidden_size=self.num_units, 
							num_layers=self.num_layers, bidirectional=self.bidirectional)
		#self.first_linear = nn.Linear(output_dim, self.num_units)
		self.last_linear = nn.Linear(self.num_units, self.num_features)

	def forward(self, input_dat, hidden=None):
		"""
			decoder forward pass
			returns 3 things:
			1 output_dat: output of forward to be used as next input.
			3 hidden_state: hidden state
		"""
		output_dat, hidden_state = self.lstm(input_dat, hidden)
		#output_loss_dat, _ = nn.utils.rnn.pad_packed_sequence(output_dat)#, total_lengths)
		output_dat = self.last_linear(output_dat)
		#output_loss_dat = self.linear(output_loss_dat)
		return output_dat, hidden_state


class Seq2SeqAutoencoder(nn.Module):
	"""
		encapsulates encoder and decoder and has methods to train the total autoencoder.
	"""

	def __init__(self, num_features, noise_prob=0.3):
		"""
			:param num_features: number of mfccs used.
		"""
		super(Seq2SeqAutoencoder, self).__init__()
		self.noise_prob = noise_prob
		self.encoder = Encoder(num_features=num_features, num_units=100, num_layers=3)
		self.decoder = Decoder(num_features=num_features, num_units=100, num_layers=3, output_dim=num_features)
		self.loss = nn.MSELoss(reduction="none")
		self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=0.01)
		self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=0.01)

	def train_step(self, input_dat, lengths):
		
		self.encoder_optimizer.zero_grad()
		self.decoder_optimizer.zero_grad()

		encoder_hidden_state=None
		#for sample in input_dat:

		# noise_mask = torch.tensor(np.random.random(input_dat.shape))
		# noised_input = input_dat.clone().detach()
		# noised_input[noise_mask<self.noise_prob] = 0.0
		_, encoder_hidden_state = self.encoder.forward(input_dat.float(), lengths, hidden=encoder_hidden_state)
		
		#print("encoder_ hs: ", encoder_hidden_state[0].shape)
		#print("***")
		#print("encoder_ cs: ", encoder_hidden_state[1].shape)
		loss=0
		device = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
		start_mfcc = torch.ones( (1, input_dat.shape[1], input_dat.shape[2]) , device=device) * -900.
		for timestep, pred_target in enumerate(input_dat):
			if timestep == 0:
				# create a mfcc tensor filled with -1 to indicate start of mfcc samples. kind of like <SOS> tags in NLP.
				output, decoder_hidden_state = self.decoder.forward(start_mfcc, hidden=encoder_hidden_state)
			else:
				output, decoder_hidden_state = self.decoder.forward(output, hidden=decoder_hidden_state)

			#print("output shapes:", output.shape)
			#print("target shape:", pred_target.shape)
			pred_target = pred_target.reshape((1, pred_target.shape[0], pred_target.shape[1]))
			#print("target shape:", pred_target.shape)
			loss += self.loss(output.float(), pred_target.float())


		loss /= len(input_dat)
		# max_dim_loss = torch.max(loss, dim=2, keepdim=True)
		# print("max dim loss", max_dim_loss)
		# max_dim_loss = torch.min(loss, dim=2, keepdim=True)
		# print("min dim loss", max_dim_loss)
		loss = torch.mean(loss)
		print("consol loss", loss)
		loss.backward()

		self.decoder_optimizer.step()
		self.encoder_optimizer.step()

		return loss.detach().cpu().numpy()

	def train(self, dataloader, epochs, epochs_per_save, model_save_path):
		device = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
		
		losses = []
		eval_losses = []

		print("training")
		self.encoder.train()
		self.decoder.train()
		for epoch in range(epochs):
			print("epoch ", epoch)
			losses.append(0)
			num_iters = 0
			for input_dat, lengths in dataloader:
				# print(input_dat)
				#print("****\n", lengths)
				gpu_input_dat = input_dat.to(device)
				gpu_lengths = lengths.to(device)
				losses[epoch] += self.train_step(gpu_input_dat, gpu_lengths)
				num_iters += 1
			
			losses[epoch] /= num_iters
			
			print(losses[epoch])
			#eval_loss = self.evaluate(dev_dataloader)
			#eval_losses.append(eval_loss)
			#print("eval_loss:", eval_loss)
			if np.isnan(losses[len(losses)-1]):
				print("model diverging! stopping training..")
				break

			# reduce lr on plateau
			# if len(losses) > 2 and losses[len(losses)-2] - losses[len(losses)-1] < 0.1:
			# 	print("reducing lr!")
			# 	for grp in self.encoder_optimizer.param_groups:
			# 		grp["lr"] /= 3
			# 	for grp in self.decoder_optimizer.param_groups:
			# 		grp["lr"] /= 3
			# elif epoch % epochs_per_save == 0 or epoch == epochs-1:
			self.save_model(epoch, epochs, epochs_per_save, os.path.join(model_save_path, "aud2vec_epoch"+str(epoch)+".pth"))

		return losses

	def resume_train(self, checkpoint, dataloader, dev_dataloader, epochs, epochs_per_save, model_save_path, init_lr=None):
		device = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
		
		self.load_state_dict(checkpoint["model_state_dict"])
		#self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer_state_dict"])
		#self.decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer_state_dict"])
		curr_epoch = checkpoint["current_epoch"]
		for grp in self.encoder_optimizer.param_groups:
			grp["lr"] = init_lr
		for grp in self.decoder_optimizer.param_groups:
			grp["lr"] /= init_lr


		print("model trained for:" + str(curr_epoch) + "epochs")
		print("training for additional:" + str(epochs) + "epochs")

		losses = []
		print("training")
		self.encoder.train()
		self.decoder.train()
		for epoch in range(curr_epoch, curr_epoch+epochs):
			print("epoch ", epoch)
			for input_dat, lengths in dataloader:
				# print(input_dat)
				#print("****\n", lengths)
				gpu_input_dat = input_dat.to(device)
				gpu_lengths = lengths.to(device)
				losses.append(self.train_step(gpu_input_dat, gpu_lengths))
			
			eval_loss = self.evaluate(dev_dataloader)
			print("eval_loss:", eval_loss)
			# if epoch % epochs_per_save == 0 or epoch == curr_epoch+epochs-1:
			# 	self.save_model(epoch, epochs, epochs_per_save, os.path.join(model_save_path, "aud2vec_epoch"+str(epoch)+".pth"))

			# reduce lr on plateau
			if np.isnan(losses[len(losses)-1]):
				print("model diverging! stopping training..")
				break
			if losses[len(losses)-2] - losses[len(losses)-1] < 0.1:
				for grp in self.encoder_optimizer.param_groups:
					grp["lr"] /= 3
				for grp in self.decoder_optimizer.param_groups:
					grp["lr"] /= 3
			elif epoch % epochs_per_save == 0 or epoch == curr_epoch+epochs-1:
				self.save_model(epoch, epochs, epochs_per_save, os.path.join(model_save_path, "aud2vec_epoch"+str(epoch)+".pth"))
			

		return losses

	def eval_step(self, input_dat, lengths):
		
		encoder_hidden_state=None
		#for sample in input_dat:

		noise_mask = torch.tensor(np.random.random(input_dat.shape))
		noised_input = input_dat.clone().detach()
		noised_input[noise_mask<self.noise_prob] = 0.0
		_, encoder_hidden_state = self.encoder.forward(input_dat.float(), lengths, hidden=encoder_hidden_state)
		
		#print("encoder_ hs: ", encoder_hidden_state[0].shape)
		#print("***")
		#print("encoder_ cs: ", encoder_hidden_state[1].shape)
		loss=0
		device = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
		start_mfcc = torch.ones( (1, input_dat.shape[1], input_dat.shape[2]) , device=device) * -900.
		for timestep, pred_target in enumerate(input_dat):
			if timestep == 0:
				# create a mfcc tensor filled with -1 to indicate start of mfcc samples. kind of like <SOS> tags in NLP.
				output, decoder_hidden_state = self.decoder.forward(start_mfcc, hidden=encoder_hidden_state)
			else:
				output, decoder_hidden_state = self.decoder.forward(output, hidden=decoder_hidden_state)

			#print("output shapes:", output.shape)
			#print("target shape:", pred_target.shape)
			pred_target = pred_target.reshape((1, pred_target.shape[0], pred_target.shape[1]))
			#print("target shape:", pred_target.shape)
			loss += self.loss(output.float(), pred_target.float())

		loss /= len(input_dat-1)

		return loss.detach().cpu().numpy()


	def evaluate(self, dev_dataloader):
		device = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
		self.to(device)
		losses = 0
		print("eval")
		self.encoder.train()
		self.decoder.train()
		num_iters = 0
		for input_dat, lengths in dev_dataloader:
			# print(input_dat)
			#print("****\n", lengths)
			gpu_input_dat = input_dat.to(device)
			gpu_lengths = lengths.to(device)
			losses+=self.eval_step(gpu_input_dat, gpu_lengths)
			num_iters += 1
		
		# return loss avg over entire dev set
		return losses / num_iters

	def embed(self, input_dat, lengths):
		
		#self.encoder.forward(input_dat, lengths)
		hidden_state=None
		#for sample in input_dat:
		input_dat = input_dat.float()
		_, hidden_state = self.encoder.forward(input_dat, lengths, hidden=hidden_state)
		
		
		hidden = hidden_state[0].transpose(0,1).detach().cpu().numpy()
		hidden = hidden.reshape((hidden.shape[0], -1))
		cell = hidden_state[1].transpose(0,1).detach().cpu().numpy()
		cell = cell.reshape((cell.shape[0], -1))

		embed_vec = np.concatenate((hidden, cell), axis=1)
		print("*******")
		print(embed_vec.shape)
		return embed_vec
	
	def save_model(self, epoch, num_epochs, epochs_per_save, path):
		torch.save({"model_state_dict":self.state_dict(),
					"encoder_optimizer_state_dict":self.encoder_optimizer.state_dict(),
					"decoder_optimizer_state_dict":self.decoder_optimizer.state_dict(),
					"current_epoch":epoch,
					"num_epochs":num_epochs,
					"epochs_per_save":epochs_per_save}
					, path)

	@staticmethod
	def load2eval(num_features, path):
		seq2seq = Seq2SeqAutoencoder(num_features)
		seq2seq.load_state_dict(torch.load(path))
		seq2seq.eval()
		return seq2seq


	
def main():
	parser=argparse.ArgumentParser()
	parser.add_argument("--model_save_path", "-s", type = str, default = "./saved_models/digits", help = "path to save the model")
	parser.add_argument("--data_path", "-d", type = str, default = "./", help = "path to computed mfcc numpy files")
	parser.add_argument("--batch_size", type=int, default=1000, help="batch size")
	parser.add_argument("--epochs", type=int, default=500, help="batch size")
	parser.add_argument("--epochs_per_save", type=int, default=10, help="number of epochs after which to save model and training checkpoint")
	parser.add_argument("--noise", type=float, default=0.0, help="with what probability to add noise to augment training")
	args=parser.parse_args()

	BATCH_SIZE = args.batch_size
	epochs = args.epochs
	epochs_per_save = args.epochs_per_save

	curr_dir = os.getcwd()
	try:
		os.chdir(args.model_save_path)
	except OSError:
		os.mkdir(args.model_save_path)
	os.chdir(curr_dir)

	data_path = args.data_path
	dataset = AudioMFCCDataset(aud_mfcc_file=os.path.join(data_path, "digit_aud_mfcc_std_trimmed.npy"), 
								aud_mfcc_lengths=os.path.join(data_path, "digit_sample_lengths_std_trimmed.npy"))
	print("loaded dataset")
	loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=AudioMFCCDataset.pack_batch)
	print("created iter on loaded data")

	aud2vec = Seq2SeqAutoencoder(dataset.num_features(), noise_prob=0.0)
	device = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
	gpu_aud2vec = aud2vec.to(device)

	print("model created")
	
	print("training begins")
	losses = gpu_aud2vec.train(loader, epochs, epochs_per_save, args.model_save_path)

	torch.save(aud2vec.state_dict(), os.path.join(args.model_save_path, "aud2vec.pth"))

if __name__ == "__main__":
	main()