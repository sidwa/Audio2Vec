
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
import math, random
import argparse
import os

from audio_dataset import AudioMFCCDataset
from torch.utils.data import DataLoader

SEQ_LEN = 40

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
		self.first_linear = nn.Linear(output_dim, self.num_units)
		self.last_linear = nn.Linear(self.num_units, output_dim)

	def forward(self, input_dat, hidden=None):
		"""
			decoder forward pass
			returns 3 things:
			1 output_dat: output of forward to be used as next input. (packed)
			2 output_loss_dat: output of forward pass to calculate loss. (unpacked-padded) to match size of input.
			3 hidden_state: hidden state
		"""
		output_dat, hidden_state = self.lstm(input_dat, hidden)
		#output_loss_dat, _ = nn.utils.rnn.pad_packed_sequence(output_dat)#, total_lengths)
		output_loss_dat = self.last_linear(output_dat)
		#output_loss_dat = self.linear(output_loss_dat)
		return output_dat, output_loss_dat, hidden_state

class Seq2SeqAutoencoder(nn.Module):
	"""
		encapsulates encoder and decoder and has methods to train the total autoencoder.
	"""

	def __init__(self, num_features):
		"""
			:param num_features: number of mfccs used.
		"""
		super(Seq2SeqAutoencoder, self).__init__()
		self.encoder = Encoder(num_features=num_features)
		self.decoder = Decoder(num_features=self.encoder.num_units, output_dim=num_features)
		self.loss = nn.MSELoss()
		self.encoder_optimizer = optim.Adam(self.encoder.parameters())
		self.decoder_optimizer = optim.Adam(self.decoder.parameters())
	
	# def __init__(self, enc, dec):
	# 	super(Seq2SeqAutoencoder, self).__init__()
	# 	self.encoder = enc
	# 	self.decoder = dec
	# 	self.loss = nn.MSELoss()
	# 	self.encoder_optimizer = optim.Adam(self.encoder.parameters())
	# 	self.decoder_optimizer = optim.Adam(self.decoder.parameters())

	def train_step(self, input_dat, lengths):
		
		self.encoder_optimizer.zero_grad()
		self.decoder_optimizer.zero_grad()

		encoder_hidden_state=None
		#for sample in input_dat:
		_, encoder_hidden_state = self.encoder.forward(input_dat.float(), lengths, hidden=encoder_hidden_state)
		
		#print("encoder_ hs: ", encoder_hidden_state[0].shape)
		#print("***")
		#print("encoder_ cs: ", encoder_hidden_state[1].shape)
		loss=0
		decoder_hidden_state = None
		for timestep, pred_target in enumerate(input_dat):
			if timestep == 0:
				output, output_loss, decoder_hidden_state = self.decoder.forward(encoder_hidden_state[0], hidden=decoder_hidden_state)
			else:
				output, output_loss, decoder_hidden_state = self.decoder.forward(output, hidden=decoder_hidden_state)

			#print("output shapes:", output.shape)
			#print("target shape:", pred_target.shape)
			pred_target = pred_target.reshape((1, pred_target.shape[0], pred_target.shape[1]))
			#print("target shape:", pred_target.shape)
			loss += self.loss(output_loss.float(), pred_target.float())


		loss /= len(input_dat-1)
		print(loss)
		loss.backward()

		self.decoder_optimizer.step()
		self.encoder_optimizer.step()

		return loss.detach().cpu().numpy()

	def train(self, dataloader, epochs, epochs_per_save, model_save_path):
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.to(device)
		losses = []
		print("training")
	
		for epoch in range(epochs):
			self.encoder.train()
			self.decoder.train()
			print("epoch ", epoch)
			for lengths, input_dat in dataloader:
				# print(input_dat)
				#print("****\n", lengths)
				gpu_input_dat = input_dat.to(device)
				gpu_lengths = lengths.to(device)
				losses.append(self.train_step(gpu_input_dat, gpu_lengths))
			
			if epoch % epochs_per_save == 0:
				self.save_model(os.path.join(model_save_path, "aud2vec_epoch"+epoch+".pth"))

		return losses

	def embed(self, input_dat, lengths):

		#self.encoder.forward(input_dat, lengths)
		hidden_state=None
		#for sample in input_dat:
		_, hidden_state = self.encoder.forward(input_dat, lengths, hidden=hidden_state)

		return hidden_state[0]
	
	def save_model(self, path):
		torch.save(self.state_dict(), path)

	@staticmethod
	def load2eval(num_features, path):
		seq2seq = Seq2SeqAutoencoder(num_features)
		seq2seq.load_state_dict(path)
		seq2seq.eval()
		return seq2seq 
	
def main():
	parser=argparse.ArgumentParser()
	parser.add_argument("--model_save_path", "-s", type = str, default = "/shared/kgcoe-research/mil/multi_modal_instance/new_data/f8k_precomp/", help = "path to save the model")
	parser.add_argument("--data_path", "-d", type = str, default = "./", help = "path to computed mfcc numpy files")
	parser.add_argument("--mode", type=str, default='train', help='Feature extraction for which phase?')
	args=parser.parse_args()

	data_path = args.data_path
	dataset = AudioMFCCDataset(aud_mfcc_file=os.path.join(data_path, "train_aud_mfcc.npy"), 
								aud_mfcc_lengths=os.path.join(data_path,"train_sample_lengths.npy"))
	print("loaded dataset")
	loader = DataLoader(dataset, batch_size=200, collate_fn=AudioMFCCDataset.pack_batch)
	print("created iter on loaded data")
	
	aud2vec = Seq2SeqAutoencoder(dataset.num_features())
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	gpu_aud2vec = aud2vec.to(device)

	print("model created")
	
	print("training begins")
	losses = aud2vec.train(loader, 100, 10, args.model_save_path)

	torch.save(aud2vec.state_dict(), os.path.join(args.model_save_path, "aud2vec.pth"))
	
if __name__ == "__main__":
	main()