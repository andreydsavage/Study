import torch.nn as nn
import torch



class GRUnet(nn.Module):
    def __init__(self, rnn_conf) -> None:
        super().__init__()
        self.rnn_conf = rnn_conf
        self.seq_len    = rnn_conf.seq_len 
        self.emb_size   = rnn_conf.embedding_dim 
        self.hidden_dim = rnn_conf.hidden_size
        self.n_layers   = rnn_conf.n_layers
        self.vocab_size = rnn_conf.vocab_size
        self.bidirectional = bool(rnn_conf.bidirectional)

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_size)

        self.bidirect_factor = 2 if self.bidirectional == 1 else 1
        
        self.gru = nn.GRU(
            input_size=self.emb_size,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            # device=rnn_conf.device
            )
        
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_dim*self.seq_len*self.bidirect_factor, 128),
            nn.Tanh(),
            nn.Linear(128,32),
            nn.Tanh(),
            nn.Linear(32,3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x.to(self.rnn_conf.device)) # Пернесим данные на девайс и создаем ембеддинги
        output, _ = self.gru(x) # Забираем hidden states со всех промежуточных состояний, второй выход финального слоя отправляем в _
        output = output.reshape(output.shape[0], -1)
        out = self.linear(output.squeeze())
        return out