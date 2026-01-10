import torch
import torch.nn as nn

# redefine model architecture
class CharLSTM(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim):
		super(CharLSTM, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(
			input_size=embedding_dim, 
			hidden_size=hidden_dim, 
			batch_first=True, 
			bidirectional=True
		)
		self.fc = nn.Linear(hidden_dim * 2, 1) 

	def forward(self, x):
		embedded = self.embedding(x)
		lstm_out, _ = self.lstm(embedded)
		logits = self.fc(lstm_out)
		return logits.squeeze(-1)

def initialize_model():
	# load the checkpoint file
	checkpoint = torch.load("segmentation_model_smaller.pth")

	# We need the exact same char-to-int mapping used during training
	config = checkpoint['config']
	char_to_ix = checkpoint['char_to_ix']
	#ix_to_char = checkpoint['ix_to_char'] # don't need for inference

	# instantiate config params and load weights
	model = CharLSTM(
		vocab_size=config['vocab_size'],
		embedding_dim=config['embedding_dim'],
		hidden_dim=config['hidden_dim']
	)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval() # Switch to evaluation mode (freezes layers)
	return model, char_to_ix

# segmentation function
def segment_text(text, model, char_to_ix, threshold=0.8):
	# Pre-processing: Text -> Indices
	idxs = [char_to_ix[ch] for ch in text.lower() if ch in char_to_ix]
	if not idxs: return ""
	inputs = torch.tensor(idxs, dtype=torch.long).unsqueeze(0)
	
	# Inference
	with torch.no_grad():
		logits = model(inputs)
		probs = torch.sigmoid(logits)[0]
		
	# Post-processing: Indices -> Text + Spaces
	output = ""
	valid_chars = [ch for ch in text if ch.lower() in char_to_ix]
	
	for i, ch in enumerate(valid_chars):
		output += ch
		# Check if the model predicts a space after this character
		if i < len(probs) and probs[i] > threshold:
			output += " "
			
	return output.strip()