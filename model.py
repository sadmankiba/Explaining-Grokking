from einops import rearrange, repeat
import torch
from torch import nn, Tensor

from data import get_vocab_size

class DecoderBlock(torch.nn.Module):
  def __init__(self, dim_model: int, n_heads: int):
    super().__init__()

    # Create a multi-head attention layer with the specified input dimension and number of heads
    self.self_attn = nn.MultiheadAttention(dim_model, n_heads)
    
    # Apply layer normalization to the output of the self-attention layer
    self.self_attn_norm = nn.LayerNorm(dim_model)
    
    # Create a feed-forward neural network with two linear layers and a GELU activation function
    self.ffn = nn.Sequential(
        nn.Linear(dim_model, dim_model * 4),
        nn.GELU(),
        nn.Linear(dim_model * 4, dim_model)
    )
    # Apply layer normalization to the output of the feed-forward network
    self.ffn_norm = nn.LayerNorm(dim_model)

    self.register_buffer('dim_model', torch.zeros(1))
    self.dim_model.fill_(dim_model)

  def forward(self, x: Tensor):
    context_size = x.size()[0]
    batch_size = x.size()[1]
    assert x.size()[2] == self.dim_model

    # Create an attention mask with dimensions (len(x), len(x)) filled with -inf
    attn_mask = torch.full(
        (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
    )
    # Set the upper triangular part of the attention mask to 0
    attn_mask = torch.triu(attn_mask, diagonal=1)
    
    # Apply self-attention to the input tensor x using the attention mask
    a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
    
    # Use residual connection and apply layer normalization
    a1 = self.self_attn_norm(x + a1)
    
    # Pass the output of the self-attention layer through the feed-forward network
    a2 = self.ffn(a1)
    
    # Use residual connection and apply layer normalization
    a2 = self.ffn_norm(a1 + a2)
    assert list(a2.size()) == [context_size, batch_size, self.dim_model.item()]

    # Return the final output
    return a2
  

class Transformer(torch.nn.Module):
  """
  Transformer model for sequence-to-sequence tasks.

  Args:
    num_layers (int): Number of decoder layers.
    dim_model (int): Dimensionality of the model.
    num_heads (int): Number of attention heads.
    vocab_size (int): Number of tokens in the vocabulary.
    context_size (int): Length of the input sequence.

  Attributes:
    token_embeddings (nn.Embedding): Embedding layer for token inputs.
    position_embeddings (nn.Embedding): Embedding layer for positional encodings.
    model (nn.Sequential): Sequential model consisting of decoder blocks.

  """

  def __init__(self, num_layers: int, dim_model: int, num_heads: int, vocab_size: int, context_size: int):
    super().__init__()

    # Embedding layers
    self.token_embeddings = nn.Embedding(vocab_size, dim_model)
    self.position_embeddings = nn.Embedding(context_size, dim_model)
    self.dim_model = dim_model
    self.vocab_size = vocab_size
    self.context_size = context_size

    # Decoder blocks
    self.model = nn.Sequential(
      *[DecoderBlock(dim_model, num_heads) for _ in range(num_layers)],
      nn.LayerNorm(dim_model),
      nn.Linear(dim_model, vocab_size)
    )

  def forward(self, inputs: Tensor) -> Tensor:
    """
    Forward pass of the Transformer model.

    Args:
      inputs (Tensor): Input tensor of shape (batch_size, context_size).

    Returns:
      Tensor: Output tensor of shape (context_size, batch_size, vocab_size).

    """
    batch_size, context_size = inputs.shape
    assert context_size == self.context_size
    assert context_size == 4

    # Token embeddings
    token_embedding = self.token_embeddings(inputs)
    assert list(token_embedding.size()) == [batch_size, context_size, self.dim_model]

    # Positional embeddings
    positions = repeat(torch.arange(context_size, device=inputs.device), "p -> b p", b=batch_size)
    assert list(positions.size()) == [batch_size, context_size]
    position_embedding = self.position_embeddings(positions)
    assert list(position_embedding.size()) == [batch_size, context_size, self.dim_model]

    # Combine token and positional embeddings
    embedding = token_embedding + position_embedding

    # Rearrange dimensions for multi-head attention
    embedding = rearrange(embedding, 'b s d -> s b d')
    assert list(embedding.size()) == [context_size, batch_size, self.dim_model]

    # Pass through the decoder blocks
    output = self.model(embedding)
    assert list(output.size()) == [context_size, batch_size, self.vocab_size]

    return output

def create_model(config):
    if config['init_from'] == "scratch":
        print("Training model from scratch")
        # Create model
        model = Transformer(num_layers=config["num_layers"], dim_model=config["dim_model"],
            num_heads=config["num_heads"], vocab_size=get_vocab_size(config["prime"]), context_size=config['context_size']
            ).to(config['device'])
        print(model)

    elif config['init_from'] == "load":
        # Load model
        print("Loading model from checkpoint")
        checkpoint = torch.load(f"checkpoints/{config['ckpt_name']}.pt", map_location=config['device'])
        model_args = {}
        for k in ['num_layers', 'dim_model', 'num_heads']:
            model_args[k] = checkpoint['config'][k]
            config[k] = checkpoint['config'][k]

        config['prime'] = checkpoint['config']['prime']
        config['seed'] = checkpoint['config']['seed']
        config['train_frac'] = checkpoint['config']['train_frac']
        model = Transformer(**model_args, 
                            vocab_size=get_vocab_size(checkpoint['config']['prime']),
                            context_size=config['context_size']).to(config['device'])

        state_dict = checkpoint['model']
        # Remove unwanted prefix
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        config['steps'] = checkpoint['steps']

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98),
        weight_decay=config["weight_decay"]
    )
    
    if config['init_from'] == 'load':
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    checkpoint = None # free up memory

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=9
    )

    if config['compile']:
        print("Compiling the model")
        model = torch.compile(model)
    
    return model, optimizer, scheduler, config