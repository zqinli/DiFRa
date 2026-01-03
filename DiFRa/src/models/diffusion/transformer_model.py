import torch
import torch.nn as nn
from transformers import AutoConfig, BertModel
from transformers.models.bert.modeling_bert import BertEncoder
import numpy as np

def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0, device=device)) / max(half - 1, 1))
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

class TransformerNetModel(nn.Module):
    def __init__(self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
        learned_mean_embed=False,
        _attn_implementation="eager"
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout_prob = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        time_embed_dim = hidden_t_dim * 4
        
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_t_dim, time_embed_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(time_embed_dim, config.hidden_size),
        )

        if init_pretrained == 'bert':
            temp_bert = BertModel.from_pretrained(config_name, config=config, attn_implementation=_attn_implementation)

            self.word_embedding = temp_bert.embeddings.word_embeddings
            with torch.no_grad():
                self.lm_head.weight = self.word_embedding.weight
            
            self.input_transformers = temp_bert.encoder
            
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm

            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'no':
            config._attn_implementation = _attn_implementation
            self.input_transformers = BertEncoder(config)

            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        else:
            assert False, "invalid type of init_pretrained"
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if learned_mean_embed:
            self.mean_embed = nn.Parameter(torch.randn(input_dims))
            nn.init.normal_(self.mean_embed, mean=0, std=input_dims ** -0.5)
        else:
            self.mean_embed = None

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1) 
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1) 
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(self.lm_head.weight, text_emb_t) 
            scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0), hidden_repr.size(1))
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError

    def forward(self, x, timesteps, **kwargs):
        try:
            weight_dtype = next(self.time_embed.parameters()).dtype
        except StopIteration:
            weight_dtype = torch.float32

        t_emb_input = timestep_embedding(timesteps, self.hidden_t_dim).to(weight_dtype).to(x.device)
        emb_t = self.time_embed(t_emb_input) 
        time_token = emb_t.unsqueeze(1)

        emb_x = x.to(weight_dtype) 
        
        seq_length = emb_x.size(1)
        
        if self.position_ids.device != x.device:
            self.position_ids = self.position_ids.to(x.device)
            
        data_position_ids = self.position_ids[:, :seq_length] 
        data_pos_emb = self.position_embeddings(data_position_ids)

        emb_x_with_pos = (emb_x.float() + data_pos_emb.float()).to(weight_dtype)

        combined_with_pos = torch.cat([time_token, emb_x_with_pos], dim=1)

        h_normalized = self.LayerNorm(combined_with_pos)
        emb_inputs = self.dropout(h_normalized)

        original_mask = kwargs.get('input_attention_mask') 
        if original_mask is None:
            original_mask = torch.ones(emb_x.shape[0], emb_x.shape[1], device=emb_x.device, dtype=torch.long)
            
        time_token_mask = torch.ones(emb_x.shape[0], 1, device=emb_x.device, dtype=torch.long)
        extended_attention_mask = torch.cat([time_token_mask, original_mask], dim=1)

        encoder_outputs = self.input_transformers(
            hidden_states=emb_inputs,
            attention_mask=extended_attention_mask
        )
        
        input_trans_hidden_states = encoder_outputs.last_hidden_state

        h = input_trans_hidden_states[:, 1:, :]
        h = h.type(x.dtype)
        
        return h