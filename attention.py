import torch 
import torch.nn as nn 
import torch.nn.functional as F 


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Attention(nn.Module):
    def __init__(self,dim_in:int,dim_out:int):
        super().__init__()
        # initializing the dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out

        # defining the linear layers for query, key, and value
        self.Q = nn.Linear(dim_in, dim_out)
        self.K = nn.Linear(dim_in, dim_out)
        self.V = nn.Linear(dim_in, dim_out)

        # scaling factor for the dot product attention
        self.scale = dim_out ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.queries = self.Q(x)
        self.keys = self.K(x)
        self.values = self.V(x)

        # calculating the dot product attention scores
        scores = torch.matmul(self.queries, self.keys.transpose(-2, -1)) / self.scale 
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, self.values)
        return scores,output


class MultiheadAttention(nn.Module):
    def __init__(self,dim_in:int,dim_out:int,num_heads:int):
        super().__init__()
        # initializing the dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        # defining the linear layers for query, key, and value
        self.Q = nn.Linear(dim_in, dim_out)
        self.K = nn.Linear(dim_in, dim_out)
        self.V = nn.Linear(dim_in, dim_out)

        # scaling factor for the dot product attention
        self.scale = self.head_dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        # linear projections
        self.queries = self.Q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        self.keys = self.K(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        self.values = self.V(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # calculating the dot product attention scores
        scores = torch.matmul(self.queries, self.keys.transpose(-2, -1)) / self.scale 
        scores = F.softmax(scores, dim=-1)

        output = torch.matmul(scores, self.values)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim_out)
        return output


class SampleTokenizer():
    def build_vocab(self) -> dict:
        self.vocab = {word: idx for idx, word in enumerate(set(self.sentence.split(' ')))}
        # adding the EOS and SOS in the vocab 
        self.vocab["<EOS>"] = len(self.vocab)
        self.vocab["<SOS>"] = len(self.vocab)
        return self.vocab
    def __init__(self,sentence:str, hidden_dim:int):
        self.sentence = sentence
        self.vocab = self.build_vocab()
        self.vocab_size = len(self.vocab)
        self.encoded = [self.vocab[word] for word in self.sentence.split(' ')]
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=hidden_dim)
        self.embedded = self.embedding(torch.tensor(self.encoded))


if __name__ == "__main__":
    sentence = "this is a sample sentence for attention mechanism"
    hidden_dim = 10
    tokenizer = SampleTokenizer(sentence, hidden_dim)
    attention_layer = Attention(dim_in=hidden_dim, dim_out=hidden_dim)
    output = attention_layer(tokenizer.embedded.unsqueeze(0))  
    print("Input shape:", tokenizer.embedded.unsqueeze(0).shape)
    print("Output shape:", output.shape)