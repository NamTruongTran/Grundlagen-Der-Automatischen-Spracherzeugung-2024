import torch

class Classification(torch.nn.Module):
    def __init__(self, idim=39, odim=1, hidden_dim=512):
        torch.nn.Module.__init__(self)
        # Define three fully connected layers followed by ReLU activation funtion
        # Define a fully connected output (classification) layer
        # Sigmoid function computes probability
        pass

    def forward(self, audio_feat):
        """
        Input: 
            audio_feat: <tensor.FloatTensor> the audio features in a tensor
        Return: 
            The predicted posterior probabilities
        """
        pass
                                                    # Features mapped by 3 fully connected layers
                                                    # Average the mapped representation over the sequence dimension
                                                    # Return the predicted probabilities by classification layer and Sigmoid function.

