# Add this class to your model.py (or a new file like embeddings.py)
import torch
import torch.nn as nn

class PretrainedEmbeddings(nn.Module):
    def __init__(self, filename, word_to_id, dim_embedding):
        super(PretrainedEmbeddings, self).__init__()

        # Load the vectors from the text file
        wordvectors = self.load_word_vectors(filename, word_to_id, dim_embedding)
        
        # Create the PyTorch embedding layer
        self.embed = nn.Embedding(len(word_to_id), dim_embedding)
        
        # Initialize the layer with the GloVe weights
        self.embed.weight = nn.Parameter(wordvectors)

    def forward(self, inputs):
        return self.embed(inputs)

    def load_word_vectors(self, filename, word_to_id, dim_embedding):
        # Initialize a tensor of zeros for the vocabulary
        wordvectors = torch.zeros(len(word_to_id), dim_embedding)
        
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                data = line.split(' ')
                word = data[0]
                vector = data[1:]
                
                # If the GloVe word is in our vocabulary, save its vector
                if word in word_to_id.keys():
                    wordvectors[word_to_id[word], :] = torch.Tensor([float(x) for x in vector])

        return wordvectors