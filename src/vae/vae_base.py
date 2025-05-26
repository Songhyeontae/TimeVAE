import os
from abc import ABC, abstractmethod
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from .metrics import VAEMetricsTracker


class Sampling(nn.Module):
    def forward(self, inputs):
        z_mean, z_log_var = inputs
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class VAE_Base(nn.Module, ABC):
    model_name = None

    def __init__(
        self,
        seq_len,
        feat_dim,
        latent_dim,
        reconstruction_wt=3.0,
        batch_size=16,
        **kwargs
    ):
        super(VAE_Base, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.batch_size = batch_size
        self.use_transformer = kwargs.get("use_transformer", False)
        self.encoder = None
        self.decoder = None
        self.sampling = Sampling()
        self.z_kl_wt = 5

    def fit_on_data(self, train_data, max_epochs=1000, verbose=0, dataset_name="unknown"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # Create log directory with timestamp and dataset name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f'runs/{dataset_name}/{timestamp}_{self.model_name}_seq{self.seq_len}_lat{self.latent_dim}_trans{self.use_transformer}'
        writer = SummaryWriter(log_dir)
        
        # Save experiment configuration as text
        config_text = f"""
        Model Configuration:
        -------------------
        Model Name: {self.model_name}
        Sequence Length: {self.seq_len}
        Feature Dimension: {self.feat_dim}
        
        Architecture:
        ------------
        Latent Dimension: {self.latent_dim}
        Hidden Layer Sizes: {getattr(self, 'hidden_layer_sizes', 'Not specified')}
        Use Transformer: {self.use_transformer}
        Use Residual Connection: {getattr(self, 'use_residual_conn', False)}
        Latent Token Dimension: {getattr(self, 'latent_token_dim', 'N/A')}
        Trend Polynomial Degree: {getattr(self, 'trend_poly', 'N/A')}
        Custom Seasonality: {getattr(self, 'custom_seas', None)}
        
        Training Parameters:
        ------------------
        Batch Size: {self.batch_size}
        Max Epochs: {max_epochs}
        Learning Rate: {0.002}  # Adam optimizer default
        Device: {device}
        
        Loss Weights:
        ------------
        Reconstruction Weight: {self.reconstruction_wt}
        Z KL Weight: {self.z_kl_wt}
        
        Runtime Info:
        ------------
        Timestamp: {timestamp}
        Total Parameters: {self.get_num_trainable_variables():,}
        """
        writer.add_text('Experiment Config', config_text)
        
        train_tensor = torch.FloatTensor(train_data).to(device)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.parameters(), lr=0.002)
        metrics_tracker = VAEMetricsTracker()
        
        global_step = 0
        
        for epoch in range(max_epochs):
            self.train()
            metrics_tracker.reset()
            
            for batch_idx, batch in enumerate(train_loader):
                X = batch[0]
                optimizer.zero_grad()
                
                if self.use_transformer:
                    z_mean, z_log_var, z, token_mean, token_log_var, tokens = self.encoder(X.to(device))
                    reconstruction = self.decoder(z, tokens)
                    loss, recon_loss, kl = self.loss_function(X, reconstruction, z_mean, z_log_var, token_mean, token_log_var)
                else:
                    z_mean, z_log_var, z = self.encoder(X.to(device))
                    reconstruction = self.decoder(z)
                    loss, recon_loss, kl = self.loss_function(X, reconstruction, z_mean, z_log_var)
                
                # Normalize the loss by the batch size
                loss = loss / X.size(0)
                recon_loss = recon_loss / X.size(0)
                kl = kl / X.size(0)
                
                # Update metrics
                batch_metrics = metrics_tracker.compute_batch_metrics(
                    z_mean, z_log_var, loss, recon_loss, kl, X.size(0)
                )
                metrics_tracker.update(batch_metrics, X=X, X_recons=reconstruction)
                
                loss.backward()
                optimizer.step()
                
                # Log batch-level metrics
                writer.add_scalar('Loss/batch/total', loss.item(), global_step)
                writer.add_scalar('Loss/batch/reconstruction', recon_loss.item(), global_step)
                writer.add_scalar('Loss/batch/kl', kl.item(), global_step)
                
                global_step += 1
            
            # Get averaged metrics for the epoch
            epoch_metrics = metrics_tracker.get_average_metrics()
            
            # Log epoch-level metrics
            writer.add_scalar('Loss/epoch/total', epoch_metrics['total_loss'], epoch)
            writer.add_scalar('Loss/epoch/reconstruction', epoch_metrics['reconstruction_loss'], epoch)
            writer.add_scalar('Loss/epoch/kl', epoch_metrics['kl_loss'], epoch)
            
            # Log discriminative and predictive scores if available
            if 'discriminative_score' in epoch_metrics:
                writer.add_scalar('Scores/discriminative', epoch_metrics['discriminative_score'], epoch)
            if 'predictive_score' in epoch_metrics:
                writer.add_scalar('Scores/predictive', epoch_metrics['predictive_score'], epoch)
            
            # Log learning rate
            writer.add_scalar('Learning/lr', optimizer.param_groups[0]['lr'], epoch)
            
            # Log latent space distributions
            writer.add_histogram('Latent/z_mean', z_mean, epoch)
            writer.add_histogram('Latent/z_log_var', z_log_var, epoch)
            
            # Plot reconstruction comparison at epoch level
            fig_recon = self._plot_reconstruction(X[0], reconstruction[0])
            writer.add_figure('Reconstruction', fig_recon, epoch)
            plt.close(fig_recon)
            
            if verbose:
                base_msg = (f"Epoch {epoch + 1}/{max_epochs} | Total loss: {epoch_metrics['total_loss']:.4f} | "
                    f"Recon loss: {epoch_metrics['reconstruction_loss']:.4f} | "
                    f"KL loss: {epoch_metrics['kl_loss']:.4f} | "
                    f"I/O shape: {X.shape} | "
                    f"usesTransformer: {self.use_transformer}")
                
                disc_score = epoch_metrics.get('discriminative_score')
                pred_score = epoch_metrics.get('predictive_score')
                
                if disc_score is not None:
                    base_msg += f" | Disc score: {disc_score:.4f}"
                if pred_score is not None:
                    base_msg += f" | Pred score: {pred_score:.4f}"
                
                print(base_msg)
                if self.use_transformer:
                    print(f"Transformer weights: {[round(self.decoder.w_trans.item(), 3), round(self.decoder.w_poly.item(), 3)]}")
        
        writer.close()

    def _plot_reconstruction(self, original, reconstruction):
        """Plot comparison between original and reconstructed time series"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # Ensure we're plotting a single time series
        if len(original.shape) > 1:  # If we have a batch or multiple features
            original = original[0] if original.shape[0] > 1 else original.squeeze()
            reconstruction = reconstruction[0] if reconstruction.shape[0] > 1 else reconstruction.squeeze()
            
        original = original.cpu().detach().numpy()
        reconstruction = reconstruction.cpu().detach().numpy()
        
        time = np.arange(len(original))
        ax1.plot(time, original, label='Original')
        ax1.set_title('Original Time Series')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(time, reconstruction, label='Reconstruction', color='orange')
        ax2.set_title('Reconstructed Time Series')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig

    def forward(self, X):
        #print(f"input_shape: {X.shape}")
        device = next(self.parameters()).device
        if self.use_transformer:
            z_mean, z_log_var, z, token_mean, token_log_var, token = self.encoder(X.to(device))
            x_decoded = self.decoder(z, token)
        else:
            z_mean, z_log_var, z = self.encoder(X.to(device))
            x_decoded = self.decoder(z)
        return x_decoded
    
    def predict(self, X):
        self.eval()
        device = next(self.parameters()).device
        num_samples = len(X)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        decoded_list = []

        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_samples)
                batch_X = torch.FloatTensor(X[start_idx:end_idx]).to(device)

                if self.use_transformer:
                    z_mean, z_log_var, z, token_mean, token_log_var, token = self.encoder(batch_X)
                    batch_decoded = self.decoder(z_mean, token_mean)
                else:
                    z_mean, z_log_var, z = self.encoder(batch_X)
                    batch_decoded = self.decoder(z_mean)
                
                decoded_list.append(batch_decoded.cpu().numpy())
        
        return np.concatenate(decoded_list, axis=0)

    def get_num_trainable_variables(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_prior_samples(self, num_samples):
        device = next(self.parameters()).device
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        samples_list = []

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, num_samples)
            current_batch_size = end_idx - start_idx

            if self.use_transformer:
                z = torch.randn(current_batch_size, self.latent_dim).to(device)
                tokens = torch.randn(current_batch_size, 12, self.latent_token_dim).to(device)
                batch_samples = self.decoder(z, tokens)
            else:
                z = torch.randn(current_batch_size, self.latent_dim).to(device)
                batch_samples = self.decoder(z)
            
            samples_list.append(batch_samples.cpu().detach().numpy())
        
        return np.concatenate(samples_list, axis=0)

    def get_prior_samples_given_Z(self, Z):
        Z = torch.FloatTensor(Z).to(next(self.parameters()).device)
        samples = self.decoder(Z)
        return samples.cpu().detach().numpy()

    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError

    def _get_reconstruction_loss(self, X, X_recons):
        def get_reconst_loss_by_axis(X, X_recons, dim):
            x_r = torch.mean(X, dim=dim)
            x_c_r = torch.mean(X_recons, dim=dim)
            err = torch.pow(x_r - x_c_r, 2)
            loss = torch.sum(err)
            return loss

        err = torch.pow(X - X_recons, 2)
        reconst_loss = torch.sum(err)
        
        reconst_loss += get_reconst_loss_by_axis(X, X_recons, dim=2)  # by time axis
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, dim=1)  # by feature axis 

        return reconst_loss

    """
    def loss_function(self, X, X_recons, z_mean, z_log_var):
        reconstruction_loss = self._get_reconstruction_loss(X, X_recons)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss
    """
    
    def loss_function(self, X, X_recons, z_mean, z_log_var, token_mean=None, token_log_var=None):
        reconstruction_loss = self._get_reconstruction_loss(X, X_recons)

        # KL divergence for z (standard VAE latent vector)
        kl_z = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        # KL divergence for tokens (optional, only if use_transformer=True)
        if self.use_transformer and token_mean is not None and token_log_var is not None:
            # shape: [B, num_tokens, latent_token_dim]
            kl_token = -0.5 * torch.sum(
                1 + token_log_var - token_mean.pow(2) - token_log_var.exp()
            )
        else:
            kl_token = 0.0
        #print(kl_token)
        num_tokens = (self.seq_len + 1) // 2

        kl_total = kl_z + (kl_token / num_tokens)
        total_loss = self.reconstruction_wt * reconstruction_loss + kl_total
        return total_loss, reconstruction_loss, kl_total

    def save_weights(self, model_dir):
        if self.model_name is None:
            raise ValueError("Model name not set.")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(model_dir, f"{self.model_name}_encoder_wts.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(model_dir, f"{self.model_name}_decoder_wts.pth"))

    def load_weights(self, model_dir):
        self.encoder.load_state_dict(torch.load(os.path.join(model_dir, f"{self.model_name}_encoder_wts.pth")))
        self.decoder.load_state_dict(torch.load(os.path.join(model_dir, f"{self.model_name}_decoder_wts.pth")))

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.save_weights(model_dir)
        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes) if hasattr(self, 'hidden_layer_sizes') else None,
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)

if __name__ == "__main__":
    pass