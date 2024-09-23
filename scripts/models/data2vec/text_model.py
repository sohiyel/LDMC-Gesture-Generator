from transformers import Data2VecTextModel
import torch.nn as nn

class TextEncoder(nn.Module):
    """
    TextEncoder class that uses the pre-trained Data2VecText model to extract text embeddings. 
    The model is partially frozen, with only the last transformer block unfreezed for fine-tuning.
    
    Attributes:
        pretrained_model (Data2VecTextModel): Pre-trained Data2Vec model for text.
        pooling_layer (nn.AdaptiveAvgPool1d): Pooling layer to aggregate embeddings across the sequence dimension.
        fc (nn.Sequential): Fully connected layer to reduce the dimensionality of embeddings.
    """

    def __init__(self):
        """
        Initializes the TextEncoder by loading the pre-trained Data2VecText model and freezing most of its layers.
        """
        super(TextEncoder, self).__init__()
        
        # Load the pre-trained Data2Vec text model
        self.pretrained_model = Data2VecTextModel.from_pretrained("facebook/data2vec-text-base")

        # Pooling layer to aggregate embeddings across the sequence dimension
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)  # Output size of 1

        # Fully connected layer to reduce the dimensionality of embeddings to 512
        self.fc = nn.Sequential(nn.Linear(768, 512))

        # Freeze all model layers except the last transformer block
        for param in self.pretrained_model.parameters():
            param.requires_grad = False  # Freeze all parameters

        # Unfreeze the last transformer block for fine-tuning
        for param in self.pretrained_model.encoder.layer[-1].parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass to extract embeddings and apply a fully connected layer.

        Args:
            x (torch.Tensor): Input tokenized text tensor.

        Returns:
            torch.Tensor: Processed text embeddings.
        """
        embeddings = self.get_embeddings(x)  # Extract embeddings
        embeddings = self.fc(embeddings)  # Apply fully connected layer
        return embeddings
    
    def get_embeddings(self, x):
        """
        Extracts text embeddings from the Data2VecText model and applies pooling.

        Args:
            x (torch.Tensor): Input tokenized text tensor.

        Returns:
            torch.Tensor: Pooled text embeddings.
        """
        x = self.pretrained_model(x).last_hidden_state  # Get the last hidden state of the text model
        
        # Reshape for pooling across the sequence dimension
        x = x.permute(0, 2, 1)  # Reshape to [batch_size, features, sequence_length]
        x = self.pooling_layer(x)  # Apply pooling to aggregate embeddings
        x = x.squeeze(dim=-1)  # Remove the sequence dimension
        
        return x
