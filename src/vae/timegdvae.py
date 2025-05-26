import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

from vae.vae_base import VAE_Base, Sampling
from torch.nn import TransformerDecoder, TransformerDecoderLayer



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.4):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on position and i
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Tokenized_Sampling(nn.Module):
    def forward(self, inputs):
        z_mean, z_log_var = inputs
        batch, seq_len, dim = z_mean.size()
        epsilon = torch.randn(batch, seq_len, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class TimeGDVAEEncoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim, latent_token_dim):
        super(TimeGDVAEEncoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_dim = latent_dim
        self.latent_token_dim = latent_token_dim
        self.num_tokens = (seq_len + 1) // 2

        token_layers = [
            nn.Conv1d(feat_dim, hidden_layer_sizes[0], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_layer_sizes[0], hidden_layer_sizes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_layer_sizes[1], hidden_layer_sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ]
        self.tokenizing_layers = nn.Sequential(*token_layers)

        conv_layers = [
            nn.Conv1d(feat_dim, hidden_layer_sizes[0], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_layer_sizes[0], hidden_layer_sizes[1], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_layer_sizes[1], hidden_layer_sizes[2], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        ]
        self.layers = nn.Sequential(*conv_layers)

        self.encoder_last_dense_dim = self._get_last_dense_dim(seq_len, feat_dim, hidden_layer_sizes)
        self.z_mean = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.z_log_var = nn.Linear(self.encoder_last_dense_dim, latent_dim)

        self.token_mean = nn.Linear(hidden_layer_sizes[2], latent_token_dim)
        self.token_log_var = nn.Linear(hidden_layer_sizes[2], latent_token_dim)
        self.sampling = Sampling()

    def _get_last_dense_dim(self, seq_len, feat_dim, hidden_layer_sizes):
        with torch.no_grad():
            x = torch.randn(1, feat_dim, seq_len)
            for conv in self.layers:
                x = conv(x)
            return x.numel()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.layers(x)
        x2 = self.tokenizing_layers(x)

        z_mean = self.z_mean(x1)
        z_log_var = self.z_log_var(x1)
        z = Sampling()([z_mean, z_log_var])

        token_means = []
        token_log_vars = []
        tokens = []
        for i in range(self.num_tokens):
            t = x2[:, :, i]
            t_mean = self.token_mean(t)
            t_log_var = self.token_log_var(t)
            t = self.sampling((t_mean, t_log_var))
            token_means.append(t_mean)
            token_log_vars.append(t_log_var)
            tokens.append(t)

        token_mean = torch.stack(token_means, dim=1)
        token_log_var = torch.stack(token_log_vars, dim=1)
        token = torch.stack(tokens, dim=1)
        #print(token.shape)
        return z_mean, z_log_var, z, token_mean, token_log_var, token


class TrendLayer(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, trend_poly):
        super(TrendLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.trend_dense1 = nn.Linear(self.latent_dim, self.feat_dim * self.trend_poly)
        self.trend_dense2 = nn.Linear(self.feat_dim * self.trend_poly, self.feat_dim * self.trend_poly)

    def forward(self, z):
        trend_params = F.relu(self.trend_dense1(z))
        trend_params = self.trend_dense2(trend_params)
        trend_params = trend_params.view(-1, self.feat_dim, self.trend_poly)

        lin_space = torch.arange(0, float(self.seq_len), 1, device=z.device) / self.seq_len 
        poly_space = torch.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], dim=0) 

        trend_vals = torch.matmul(trend_params, poly_space) 
        trend_vals = trend_vals.permute(0, 2, 1) 
        return trend_vals




class GPTTrendLayer(nn.Module):
    def __init__(self, z_dim, seq_len, d_model, nhead, num_layers, dropout, output_dim=1):
        super(GPTTrendLayer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.output_dim = output_dim

        # Project token (latent_dim) to model dim
        self.memory_proj = nn.Linear(z_dim, d_model)
        self.token_proj = nn.Linear(z_dim, d_model)
        

        # Learnable decoder query input
        self.decoder_input = nn.Parameter(torch.randn(1, self.seq_len, d_model))
        nn.init.normal_(self.decoder_input, mean=0.0, std=0.02)

        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        decoder_layer = TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
            activation="relu",
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, output_dim)
    
    def _generate_causal_mask(self, size, device):
        # Prevent position i from attending to positions > i (upper triangular)
        return torch.triu(torch.ones(size, size, device=device) * float("-inf"), diagonal=1)

    def forward(self, tokens):
        # tokens: (B, num_tokens, z_dim)
        # print(tokens.shape)
        memory = self.memory_proj(tokens)  # (B, num_tokens, d_model)

        batch_size = memory.size(0)
        #batch_size, token_seq_len, _ = x.size()
        #causal_mask = self._generate_causal_mask(seq_len, tokens.device)


        # tokens: (B, seq_len, z_dim) - GPT
        #x = self.token_proj(tokens)  # (B, seq_len, d_model)
        #x = self.pos_encoder(x)
        # Use causal mask
        #causal_mask = self._generate_causal_mask(self.seq_len, tokens.device)


        # Learnable query expanded to batch
        tgt = self.decoder_input.expand(batch_size, -1, -1)  # (B, seq_len, d_model)
        tgt = self.pos_encoder(tgt)

        out = self.decoder(tgt=tgt, memory=memory)
        #out = self.decoder(tgt=x, memory=None, tgt_mask=causal_mask)
        out = self.norm(out)
        return self.output_proj(out)  # (B, seq_len, output_dim)



class SeasonalLayer(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, custom_seas):
        super(SeasonalLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.custom_seas = custom_seas

        self.dense_layers = nn.ModuleList([
            nn.Linear(latent_dim, feat_dim * num_seasons)
            for num_seasons, len_per_season in custom_seas
        ])
        

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_indexes = torch.arange(num_seasons).unsqueeze(1) + torch.zeros(
            (num_seasons, len_per_season), dtype=torch.int32
        )
        season_indexes = season_indexes.view(-1)
        season_indexes = season_indexes.repeat(self.seq_len // len_per_season + 1)[: self.seq_len]
        return season_indexes

    def forward(self, z):
        N = z.shape[0]
        ones_tensor = torch.ones((N, self.feat_dim, self.seq_len), dtype=torch.int32, device=z.device)

        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)
            season_params = season_params.view(-1, self.feat_dim, num_seasons)

            season_indexes_over_time = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            ).to(z.device, dtype=torch.long)

            dim2_idxes = ones_tensor * season_indexes_over_time.view(1, 1, -1)
            season_vals = torch.gather(season_params, 2, dim2_idxes)

            all_seas_vals.append(season_vals)

        all_seas_vals = torch.stack(all_seas_vals, dim=-1) 
        all_seas_vals = torch.sum(all_seas_vals, dim=-1)  
        all_seas_vals = all_seas_vals.permute(0, 2, 1)  

        return all_seas_vals

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.seq_len, self.feat_dim)
    

class LevelModel(nn.Module):
    def __init__(self, latent_dim, feat_dim, seq_len):
        super(LevelModel, self).__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.level_dense1 = nn.Linear(self.latent_dim, self.feat_dim)
        self.level_dense2 = nn.Linear(self.feat_dim, self.feat_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        level_params = self.relu(self.level_dense1(z))
        level_params = self.level_dense2(level_params)
        level_params = level_params.view(-1, 1, self.feat_dim)

        ones_tensor = torch.ones((1, self.seq_len, 1), dtype=torch.float32, device=z.device)
        level_vals = level_params * ones_tensor
        return level_vals


class ResidualConnection(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim, encoder_last_dense_dim):
        super(ResidualConnection, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        
        self.dense = nn.Linear(latent_dim, encoder_last_dense_dim)
        self.deconv_layers = nn.ModuleList()
        in_channels = hidden_layer_sizes[-1]
        
        for i, num_filters in enumerate(reversed(hidden_layer_sizes[:-1])):
            self.deconv_layers.append(
                nn.ConvTranspose1d(in_channels, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            in_channels = num_filters
            
        self.deconv_layers.append(
            nn.ConvTranspose1d(in_channels, feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        L_in = encoder_last_dense_dim // hidden_layer_sizes[-1] 
        for i in range(len(hidden_layer_sizes)):
            L_in = (L_in - 1) * 2 - 2 * 1 + 3 + 1 
        L_final = L_in 

        self.final_dense = nn.Linear(feat_dim * L_final, seq_len * feat_dim)

    def forward(self, z):
        batch_size = z.size(0)
        x = F.relu(self.dense(z))
        x = x.view(batch_size, -1, self.hidden_layer_sizes[-1])
        x = x.transpose(1, 2)
        
        for deconv in self.deconv_layers[:-1]:
            x = F.relu(deconv(x))
        x = F.relu(self.deconv_layers[-1](x))
        
        x = x.flatten(1)
        x = self.final_dense(x)
        residuals = x.view(-1, self.seq_len, self.feat_dim)
        return residuals



class TimeGDVAEDecoder(nn.Module):
    def __init__(
        self,
        seq_len,
        feat_dim,
        hidden_layer_sizes,
        latent_dim,
        latent_token_dim,
        trend_poly,
        use_transformer=True,
        custom_seas=None,
        use_residual_conn=True,
        encoder_last_dense_dim=None,
    ):
        super(TimeGDVAEDecoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_dim = latent_dim
        self.latent_token_dim = latent_token_dim
        self.num_tokens = (seq_len + 1) // 2
        self.use_transformer = use_transformer
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        self.trend_poly = trend_poly
        self.encoder_last_dense_dim = encoder_last_dense_dim

        #self.w_lev = nn.Parameter(torch.tensor(1.0))
        self.w_trans = nn.Parameter(torch.tensor(1.0))
        self.w_poly = nn.Parameter(torch.tensor(1.0))
        #self.w_res = nn.Parameter(torch.tensor(1.0))

        self.level_model = LevelModel(self.latent_dim, self.feat_dim, self.seq_len)

        if self.use_transformer:
            self.trend_model = GPTTrendLayer(
                z_dim=self.latent_token_dim,
                seq_len=self.seq_len,
                d_model=12,
                nhead=2,
                num_layers=2,
                dropout=0.4,
                output_dim=self.feat_dim,
            )

        if self.trend_poly is not None and self.trend_poly > 0:
            #print(self.trend_poly)
            self.trend_layer = TrendLayer(self.seq_len, self.feat_dim, self.latent_dim, self.trend_poly)

        if self.custom_seas is not None and len(self.custom_seas) > 0:
            self.seasonal_layer = SeasonalLayer(self.seq_len, self.feat_dim, self.latent_dim, self.custom_seas)
        else:
            self.seasonal_layer = None

        if use_residual_conn:
            self.residual_conn = ResidualConnection(seq_len, feat_dim, hidden_layer_sizes, latent_dim, encoder_last_dense_dim)

    def forward(self, z, token=None):
        z = z.to(next(self.parameters()).device)
        weights = torch.softmax(torch.stack([
            self.w_trans,
            self.w_poly,
        ]), dim=0)
        outputs = self.level_model(z)

        if self.use_transformer and token is not None:
            trend_vals = self.trend_model(token)
            #print(trend_vals)
            outputs += weights[0] * trend_vals
        
        if self.trend_poly is not None and self.trend_poly > 0:
            trend_vals_2 = self.trend_layer(z)
            outputs += weights[1] * trend_vals_2

        # custom seasons
        if self.custom_seas is not None and len(self.custom_seas) > 0:
            cust_seas_vals = self.seasonal_layer(z)
            outputs += cust_seas_vals

        if self.use_residual_conn:
            residuals = self.residual_conn(z)
            outputs += residuals

        #print(f"output shape: {outputs.shape}")
        return outputs


class TimeGDVAE(VAE_Base):
    model_name = "TimeGDVAE"

    def __init__(
        self,
        latent_token_dim,
        trend_poly,
        hidden_layer_sizes=None,
        custom_seas=None,
        use_residual_conn=True,
        **kwargs,
    ):
        super(TimeGDVAE, self).__init__(**kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.latent_token_dim = latent_token_dim

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]

        self.hidden_layer_sizes = hidden_layer_sizes
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        self.trend_poly = trend_poly

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _get_encoder(self):
        return TimeGDVAEEncoder(self.seq_len, self.feat_dim, self.hidden_layer_sizes, self.latent_dim, self.latent_token_dim)

    def _get_decoder(self):
        return TimeGDVAEDecoder(
          self.seq_len,
          self.feat_dim,
          self.hidden_layer_sizes,
          self.latent_dim,
          self.latent_token_dim,
          self.trend_poly,
          use_transformer=self.use_transformer,
          custom_seas=self.custom_seas,
          use_residual_conn=self.use_residual_conn,
          encoder_last_dense_dim=self.encoder.encoder_last_dense_dim 
      )

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, f"{self.model_name}_weights.pth"))

        if self.custom_seas is not None:
            self.custom_seas = [(int(num_seasons), int(len_per_season)) for num_seasons, len_per_season in self.custom_seas]

        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "latent_token_dim": self.latent_token_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "custom_seas": self.custom_seas,
            "use_residual_conn": self.use_residual_conn,
            "trend_poly": self.trend_poly,
            "use_transformer": bool(self.use_transformer)
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)

    @classmethod
    def load(cls, model_dir: str) -> "TimeGDVAE":
        params_file = os.path.join(model_dir, f"{cls.model_name}_parameters.pkl")
        dict_params = joblib.load(params_file)
        vae_model = TimeGDVAE(**dict_params)
        vae_model.load_state_dict(torch.load(os.path.join(model_dir, f"{cls.model_name}_weights.pth")), strict=False)
        return vae_model