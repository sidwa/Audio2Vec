"""
	Main file Containing the code for all models. We use a seq2seq autoencoder.
	The autoencoder has it's own class with functions for the model forward prop and also to train it.
"""



from tensorboard_logger import Logger
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
import math, random
import os, argparse, datetime
from audio_dataset import AudioMFCCDataset
from torch.utils.data import DataLoader
import pdb

DEVICE_ID = "cuda:1"
TOTAL_OBS = None

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
		self.lstm1 = nn.LSTM(input_size=self.num_features, hidden_size=self.num_units, 
							num_layers=self.num_layers, bidirectional=self.bidirectional)
		# self.lstm2 = nn.LSTM(input_size=self.num_units, hidden_size=self.num_features,
		# 					num_layers=1, bidirectional=self.bidirectional)

	def forward(self, input_dat, input_lengths, hidden=None):
		input_dat = nn.utils.rnn.pack_padded_sequence(input_dat, input_lengths, enforce_sorted=False)
		output_dat, hidden_state = self.lstm1(input_dat, hidden)
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
		self.lstm1 = nn.LSTM(input_size=self.num_features, hidden_size=self.num_units, 
							num_layers=self.num_layers, bidirectional=self.bidirectional)
		self.linear = nn.Linear(self.num_units, self.num_features)
		# self.lstm2 = nn.LSTM(input_size=self.num_units, hidden_size=self.num_features, 
		# 					num_layers=1, bidirectional=self.bidirectional)

	def forward(self, input_dat, hidden=None):
		"""
			decoder forward pass
			returns 2 things:
			1 output_dat: output of forward to be used as next input.
			2 hidden state of first layers(lstm1): these layers should perform bulk of func approx
			XX---3 hidden_state of reduction layer(lstm2): reduction LSTM layer. Ensures output has same dims as input---XX
		"""
		output_dat, hidden_state = self.lstm1(input_dat, hidden)
		#output_loss_dat, _ = nn.utils.rnn.pad_packed_sequence(output_dat)#, total_lengths)
		output_dat = self.linear(output_dat)
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
		self.encoder = Encoder(num_features=num_features, num_units=100, num_layers=1)
		self.decoder = Decoder(num_features=num_features, num_units=100, num_layers=1, output_dim=num_features)
		self.loss = nn.MSELoss("""reduction="none" """)
		self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=1e-2)
		self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=1e-2)
		
		# <sos> tag for MFCC
		self.start_mfcc_val = 500.

		# <eos> tag for MFCC
		self.end_mfcc_val = 6.0

	def train_step(self, input_dat, lengths, teacher_forcing=0.6):
		"""
			Functions encapsulating the computations for exactly one training iteration.
			:param input_dat: input data as a numpy array of shape (SEQ_LEN, batch, feature_len)
			:param lengths: list containing the number of MFCC vectors extracted for each audio file.(needed for compute optim)
			:param teacher_forcing: probability of applying teacher forcing. The decoder would use the correct MFCC vector of prev
									timestep instead of the one produced by the decoder with probability of "teacher_forcing"
			:returns:
				loss: amount of MSE loss computed in current iteration(scalar)
		"""

		self.encoder_optimizer.zero_grad()
		self.decoder_optimizer.zero_grad()


		# code for denoising autoencoder..
		# noise_mask = torch.tensor(np.random.random(input_dat.shape))
		# noised_input = input_dat.clone().detach()
		# noised_input[noise_mask<self.noise_prob] = 0.0
		_, encoder_hidden_state = self.encoder.forward(input_dat.float(), lengths, hidden=None)
		
		loss=0
		device = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
		start_mfcc = torch.ones( (1, input_dat.shape[1], input_dat.shape[2]) , device=device) * self.start_mfcc_val
		prev_output= None

		for timestep, pred_target in enumerate(input_dat):
			if timestep == 0:
				# create a mfcc tensor filled with -1 to indicate start of mfcc samples. kind of like <SOS> tags in NLP.
				output, decoder_hidden_state = self.decoder.forward(start_mfcc, hidden=encoder_hidden_state)
				#print(output.shape)
			else:
				# if teacher forcing ignore previous output.
				if np.random.random() < teacher_forcing:
					correct_input = input_dat[timestep-1].view(1, input_dat.shape[1], input_dat.shape[2])

					output, decoder_hidden_state = \
													self.decoder.forward(correct_input.float(), \
																hidden=decoder_hidden_state)
				else:
					output, decoder_hidden_state = \
													self.decoder.forward(prev_output.float(), \
																hidden=decoder_hidden_state)
			
			prev_output = output.detach()


			pred_target = pred_target.reshape((1, pred_target.shape[0], pred_target.shape[1]))

			pred_target[:, lengths==timestep, :] = self.end_mfcc_val
			
			sl_output = output[:,lengths>=timestep,:]
			sl_pred_target = pred_target[:,lengths>=timestep,:]
			#print(sl_out.shape)
			#print(sl_pred.shape)
			loss += self.loss(sl_output.float(), sl_pred_target.float())

		# loss /= len(input_dat)
		# output_dim_0 = output[0,:,0]
		# pred_dim_0 = pred_target[0,:,0]
		# print("output dim 0 ", output_dim_0)
		# print("pred dim 0 ", pred_dim_0)
		# max_dim_loss = torch.max(loss, dim=2, keepdim=True)
		# print("max dim loss: ", max_dim_loss)
		# min_dim_loss = torch.min(loss, dim=2, keepdim=True)
		# print("min dim loss: ", min_dim_loss)
		# loss = torch.mean(loss)

		print("consol loss", loss)
		loss.backward()

		_ = nn.utils.clip_grad_norm_(self.encoder.parameters(), 50.0)
		_ = nn.utils.clip_grad_norm_(self.decoder.parameters(), 50.0)
	
		self.decoder_optimizer.step()
		self.encoder_optimizer.step()



		return loss.detach().cpu().numpy()

	def train(self, dataloader, dev_dataloader, epochs, epochs_per_save, model_save_path, logger):
		"""
			trains the model
			:param dataloader: pytorch dataloader for training data.
			:param dev_dataloader: pytorch dataloader for evaluation data
			:param epochs: number of epochs to train
			:param epochs_per_save: save every epochs_per_save epochs.
			:param model_save_path: model parameters save location
			:param logger: tensorboard logger object
		"""
		device = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
		const_lr =False
		losses = []
		eval_losses = []
		teacher_forcing = 0.0
		print("training")
		self.encoder.train()
		self.decoder.train()
		init_patience = 4
		patience = init_patience
		tot_iters = 0
		for epoch in range(epochs):
			print("epoch ", epoch)
			losses.append(0)
			num_iters = 0
			for input_dat, lengths in dataloader:
				# print(input_dat)
				#print("****\n", lengths)
				gpu_input_dat = input_dat.to(device)
				gpu_lengths = lengths.to(device)
				iter_loss = self.train_step(gpu_input_dat, gpu_lengths, teacher_forcing)
				losses[epoch] += iter_loss
				
				if tot_iters % 10 == 0:
					logger.log_value("loss", iter_loss, tot_iters)

					# Log values and gradients of the parameters (histogram summary)
					for tag, value in self.named_parameters():
						#print("tag:" + str(tag)  + "\nvalue: " + str(value.grad))
						tag = tag.replace('.', '/')
						logger.log_histogram(tag, value.data.cpu().numpy(), tot_iters)
						logger.log_histogram(tag+'/grad', value.grad.data.cpu().numpy(), tot_iters)
				
				num_iters += 1
				tot_iters += 1



			losses[epoch] /= num_iters
			
			eval_loss = self.evaluate(dev_dataloader)
			logger.log_value("eval loss", eval_loss, tot_iters)
			eval_losses.append(eval_loss)
			print("eval_loss:", eval_loss)

			# if np.isnan(losses[len(losses)-1]):
			# 	print("model diverging! stopping training..")
			# 	break

			# learning rate schedule
			if len(eval_losses) > 2 and eval_losses[len(eval_losses)-2] - eval_losses[len(eval_losses)-1] < 0.01:
				print("eval loss not reduced!")
				patience -= 1
				if patience == 0 and not const_lr:
					print("reducing lr!")
					for grp in self.encoder_optimizer.param_groups:
						grp["lr"] /= 3
					for grp in self.decoder_optimizer.param_groups:
						grp["lr"] /= 3
					for grp in self.decoder_optimizer.param_groups:
						if grp["lr"] <= 1e-8:
							print("lr super low! setting it constant")
							const_lr = True
					patience = init_patience
			if epoch % epochs_per_save == 0 or epoch == epochs-1:
				#teacher_forcing -= 0.1
				self.save_model(epoch, epochs, epochs_per_save, os.path.join(model_save_path, "aud2vec_epoch"+str(epoch)+".pth"))

		return losses

	def resume_train(self, checkpoint, dataloader, dev_dataloader, epochs, epochs_per_save, model_save_path, logger, init_lr=None):
		"""
			resumes training from a saved state
			:param checkpoin: model + optimizer save state
			:param dataloader: pytorch dataloader for training data.
			:param dev_dataloader: pytorch dataloader for evaluation data
			:param epochs: number of additional epochs to train for
			:param epochs_per_save: save every epochs_per_save epochs.
			:param model_save_path: model parameters save location
			:param logger: tensorboard logger object
			:param init_lr: initial learning rate to resume training 
		"""
		device = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
		
		self.load_state_dict(checkpoint["model_state_dict"])
		#self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer_state_dict"])
		#self.decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer_state_dict"])
		curr_epoch = checkpoint["current_epoch"]
		if init_lr is not None:
			for grp in self.encoder_optimizer.param_groups:
				grp["lr"] = init_lr
			for grp in self.decoder_optimizer.param_groups:
				grp["lr"] = init_lr


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
				losses.append(self.train_step(gpu_input_dat, gpu_lengths, teacher_forcing=0.0))
			
			eval_loss = self.evaluate(dev_dataloader)
			print("eval_loss:", eval_loss)
			# if epoch % epochs_per_save == 0 or epoch == curr_epoch+epochs-1:
			# 	self.save_model(epoch, epochs, epochs_per_save, os.path.join(model_save_path, "aud2vec_epoch"+str(epoch)+".pth"))

			# reduce lr on plateau
			# if np.isnan(losses[len(losses)-1]):
			# 	print("model diverging! stopping training..")
			# 	break
			if losses[len(losses)-2] - losses[len(losses)-1] < 0.1:
				print("reducing lr!")
				for grp in self.encoder_optimizer.param_groups:
					grp["lr"] /= 3
				for grp in self.decoder_optimizer.param_groups:
					grp["lr"] /= 3
			elif epoch % epochs_per_save == 0 or epoch == curr_epoch+epochs-1:
				self.save_model(epoch, epochs, epochs_per_save, os.path.join(model_save_path, "aud2vec_epoch"+str(epoch)+".pth"))
			

		return losses

	def eval_step(self, input_dat, lengths):
		"""
			evaluation step to compute evaluation loss on mini-batch
			:param input_dat: input data as a numpy array of shape (SEQ_LEN, batch, feature_len)
			:param lengths: list containing the number of MFCC vectors extracted for each audio file.(needed for compute optim)
		"""
		noise_mask = torch.tensor(np.random.random(input_dat.shape))
		noised_input = input_dat.clone().detach()
		noised_input[noise_mask<self.noise_prob] = 0.0
		_, encoder_hidden_state = self.encoder.forward(input_dat.float(), 
															lengths, hidden=None)
		
		#print("encoder_ hs: ", encoder_hidden_state[0].shape)
		#print("***")
		#print("encoder_ cs: ", encoder_hidden_state[1].shape)
		loss=0
		device = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
		start_mfcc = torch.ones( (1, input_dat.shape[1], input_dat.shape[2]) , device=device) * self.start_mfcc_val
		for timestep, pred_target in enumerate(input_dat):
			if timestep == 0:
				# create a mfcc tensor filled with -1 to indicate start of mfcc samples. kind of like <SOS> tags in NLP.
				output, decoder_hidden_state = self.decoder.forward(start_mfcc, \
																		hidden=encoder_hidden_state)
			else:
				output, decoder_hidden_state = self.decoder.forward(output, \
																		hidden=decoder_hidden_state)

			#print("output shapes:", output.shape)
			#print("target shape:", pred_target.shape)
			pred_target = pred_target.reshape((1, pred_target.shape[0], pred_target.shape[1]))
			#print("target shape:", pred_target.shape)


			pred_target[:, lengths==timestep, :] = self.end_mfcc_val
			
			sl_output = output[:,lengths>=timestep,:]
			sl_pred_target = pred_target[:,lengths>=timestep,:]

			loss += self.loss(sl_output.float(), sl_pred_target.float())

		#loss /= len(input_dat)
		loss = torch.mean(loss)

		return loss.detach().cpu().numpy()


	def evaluate(self, dev_dataloader):
		"""
			Called after each training iter to compute loss on evaluation dataset.
				:param dev_dataloader: pytorch dataloader for eval dataset

				:returns:
					loss: evaluation data set loss (scalar)
			
		"""
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
		"""
			Computes Audio2Vec if model already trained.
		"""
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
	parser.add_argument("--model_save_path", "-s", type = str, default = "./saved_models", help = "path to save the model")
	parser.add_argument("--data_path", "-d", type = str, default = "./", help = "path to computed mfcc numpy files")
	parser.add_argument("--log_path", "-l", type = str, default = "./summary/", help = "path to log dir")
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
	dataset = AudioMFCCDataset(aud_mfcc_file=os.path.join(data_path, "train_aud_mfcc_norm.npy"), 
								aud_mfcc_lengths=os.path.join(data_path,"train_sample_lengths_norm.npy"),
								data_size=None)
	print("loaded dataset")
	loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=AudioMFCCDataset.pack_batch)
	print("created iter on loaded data")
	
	TOTAL_OBS = dataset.__len__()

	# dev set not std trimmmed to check if longer audio is also getting properly encoded.
	dev_dataset = AudioMFCCDataset(aud_mfcc_file=os.path.join(data_path, "dev_aud_mfcc_norm.npy"), 
								aud_mfcc_lengths=os.path.join(data_path,"dev_sample_lengths_norm.npy"),
								data_size=None)
	print("loaded dev dataset")
	dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=AudioMFCCDataset.pack_batch)
	print("created iter on loaded dev data")

	# logging information
	now = datetime.datetime.now()
	logger = Logger(args.log_path + now.strftime("%Y-%m-%d_%H_%M"), flush_secs=5)

	aud2vec = Seq2SeqAutoencoder(dataset.num_features(), noise_prob=0.0)
	device = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
	gpu_aud2vec = aud2vec.to(device)

	print("model created")
	
	print("training begins")
	losses = gpu_aud2vec.train(loader, dev_loader, epochs, epochs_per_save, args.model_save_path, logger)

	torch.save(aud2vec.state_dict(), os.path.join(args.model_save_path, "aud2vec.pth"))


def test():
	dataset = AudioMFCCDataset(aud_mfcc_file="train_aud_mfcc_trimmed.npy", 
								aud_mfcc_lengths="train_sample_lengths_trimmed.npy")
	print("loaded dataset")
	loader = DataLoader(dataset, batch_size=200, collate_fn=AudioMFCCDataset.pack_batch)
	print("created iter on loaded data")
	
	aud2vec = Seq2SeqAutoencoder(dataset.num_features(), noise_prob=0.0)
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#gpu_aud2vec = aud2vec.to(device)
	idx = 0
	for input_dat, lengths in loader:
		print(aud2vec.embed(input_dat, lengths))
		break

if __name__ == "__main__":
	main()