import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def pad_sequences(sequences, max_len=None, padding_value=0.0):
    """Pad sequences to the same length.
    
    Args:
        sequences: list of variable length sequences
        max_len: maximum length to pad to (if None, use longest sequence)
        padding_value: value to use for padding
        
    Returns:
        padded_sequences: tensor of padded sequences
        sequence_lengths: original lengths of sequences
    """
    if max_len is None:
        max_len = max(seq.shape[0] for seq in sequences)
        
    batch_size = len(sequences)
    feat_dim = sequences[0].shape[-1]
    
    padded_sequences = torch.full((batch_size, max_len, feat_dim), padding_value)
    sequence_lengths = []
    
    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        sequence_lengths.append(seq_len)
        padded_sequences[i, :seq_len] = torch.FloatTensor(seq)
        
    return padded_sequences, torch.tensor(sequence_lengths)

class RNNDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(RNNDiscriminator, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: Padded sequence tensor (batch_size, max_seq_len, input_dim)
            lengths: Original sequence lengths
        """
        if lengths is not None:
            # Pack the padded sequences
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # Process through RNN
        out, _ = self.rnn(x)
        
        if lengths is not None:
            # Unpack the sequences
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            
            # Get the last hidden state for each sequence using the lengths
            batch_size = out.size(0)
            idx = (lengths - 1).view(-1, 1).expand(batch_size, out.size(2))
            idx = idx.unsqueeze(1).to(out.device)
            out = out.gather(1, idx).squeeze(1)
        else:
            # If no lengths provided, use the last state of the sequence
            out = out[:, -1]
            
        return self.fc(out)

def discriminative_score(feat_real, feat_fake, epochs=10):
    """Discriminative score for time-series data using RNN.
    
    Args:
        - feat_real: real data features
        - feat_fake: fake data features
        - epochs: number of training epochs
        
    Returns:
        - discriminative_score: np.float
    """
    # Convert to torch tensors and handle padding
    feat_real = [torch.FloatTensor(x) for x in feat_real]
    feat_fake = [torch.FloatTensor(x) for x in feat_fake]
    
    # Pad sequences
    padded_real, real_lengths = pad_sequences(feat_real)
    padded_fake, fake_lengths = pad_sequences(feat_fake)
    
    # Train test split
    idx = torch.randperm(len(real_lengths))
    train_idx = idx[:int(len(idx)*0.8)]
    test_idx = idx[int(len(idx)*0.8):]
    
    # Prepare training data
    train_real = padded_real[train_idx]
    train_real_lengths = real_lengths[train_idx]
    test_real = padded_real[test_idx]
    test_real_lengths = real_lengths[test_idx]
    
    train_fake = padded_fake[train_idx]
    train_fake_lengths = fake_lengths[train_idx]
    test_fake = padded_fake[test_idx]
    test_fake_lengths = fake_lengths[test_idx]
    
    train_data = torch.cat([train_real, train_fake])
    train_lengths = torch.cat([train_real_lengths, train_fake_lengths])
    train_labels = torch.cat([
        torch.ones(len(train_real)),
        torch.zeros(len(train_fake))
    ]).unsqueeze(-1)
    
    test_data = torch.cat([test_real, test_fake])
    test_lengths = torch.cat([test_real_lengths, test_fake_lengths])
    test_labels = torch.cat([
        torch.ones(len(test_real)),
        torch.zeros(len(test_fake))
    ]).unsqueeze(-1)
    
    # Initialize model
    input_dim = feat_real[0].shape[-1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    discriminator = RNNDiscriminator(input_dim).to(device)
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.005)
    batch_size = 32
    
    # Create data loaders with lengths
    train_dataset = torch.utils.data.TensorDataset(train_data, train_lengths, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # Training loop
    discriminator.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_data, batch_lengths, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_lengths = batch_lengths.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = discriminator(batch_data, batch_lengths)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = (outputs > 0.5).float()
            correct += (pred == batch_labels).sum().item()
            total += batch_labels.size(0)
    
    # Evaluation
    discriminator.eval()
    with torch.no_grad():
        test_data = test_data.to(device)
        test_lengths = test_lengths.to(device)
        test_labels = test_labels.to(device)
        pred = discriminator(test_data, test_lengths)
        pred_labels = (pred > 0.5).float()
        accuracy = (pred_labels == test_labels).float().mean().item()
    
    return abs(accuracy - 0.5)

class RNNPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(RNNPredictor, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x, lengths=None):
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        out, _ = self.rnn(x)
        
        if lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        return self.fc(out)

def predictive_score(feat_real, feat_fake, epochs=10):
    """Predictive score for time-series data using RNN.
    
    Args:
        - feat_real: real data features
        - feat_fake: fake data features
        - epochs: number of training epochs
        
    Returns:
        - predictive_score: np.float
    """
    # Convert to torch tensors and handle padding
    feat_real = [torch.FloatTensor(x) for x in feat_real]
    feat_fake = [torch.FloatTensor(x) for x in feat_fake]
    
    # Pad sequences
    padded_real, real_lengths = pad_sequences(feat_real)
    padded_fake, fake_lengths = pad_sequences(feat_fake)
    
    # Train test split for real data
    idx = torch.randperm(len(real_lengths))
    train_idx = idx[:int(len(idx)*0.8)]
    test_idx = idx[int(len(idx)*0.8):]
    
    train_real = padded_real[train_idx]
    train_real_lengths = real_lengths[train_idx]
    test_real = padded_real[test_idx]
    test_real_lengths = real_lengths[test_idx]
    
    # Train test split for fake data
    idx = torch.randperm(len(fake_lengths))
    train_idx = idx[:int(len(idx)*0.8)]
    test_idx = idx[int(len(idx)*0.8):]
    
    train_fake = padded_fake[train_idx]
    train_fake_lengths = fake_lengths[train_idx]
    test_fake = padded_fake[test_idx]
    test_fake_lengths = fake_lengths[test_idx]
    
    # Initialize models and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = feat_real[0].shape[-1]
    predictor_real = RNNPredictor(input_dim).to(device)
    predictor_fake = RNNPredictor(input_dim).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer_real = torch.optim.Adam(predictor_real.parameters(), lr=0.005)
    optimizer_fake = torch.optim.Adam(predictor_fake.parameters(), lr=0.005)
    batch_size = 32
    
    # Create data loaders
    train_real_dataset = torch.utils.data.TensorDataset(train_real, train_real_lengths)
    train_fake_dataset = torch.utils.data.TensorDataset(train_fake, train_fake_lengths)
    
    train_real_loader = torch.utils.data.DataLoader(
        train_real_dataset, batch_size=batch_size, shuffle=True
    )
    train_fake_loader = torch.utils.data.DataLoader(
        train_fake_dataset, batch_size=batch_size, shuffle=True
    )
    
    # Train predictor on real data
    predictor_real.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_data, batch_lengths in train_real_loader:
            batch_data = batch_data.to(device)
            batch_lengths = batch_lengths.to(device)
            
            optimizer_real.zero_grad()
            # Predict next timestep
            outputs = predictor_real(batch_data[:, :-1], batch_lengths-1)
            loss = criterion(outputs, batch_data[:, 1:])
            loss.backward()
            optimizer_real.step()
            
            total_loss += loss.item()
    
    # Train predictor on fake data
    predictor_fake.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_data, batch_lengths in train_fake_loader:
            batch_data = batch_data.to(device)
            batch_lengths = batch_lengths.to(device)
            
            optimizer_fake.zero_grad()
            # Predict next timestep
            outputs = predictor_fake(batch_data[:, :-1], batch_lengths-1)
            loss = criterion(outputs, batch_data[:, 1:])
            loss.backward()
            optimizer_fake.step()
            
            total_loss += loss.item()
    
    # Evaluation
    predictor_real.eval()
    predictor_fake.eval()
    with torch.no_grad():
        # Move test data to device
        test_real = test_real.to(device)
        test_real_lengths = test_real_lengths.to(device)
        test_fake = test_fake.to(device)
        test_fake_lengths = test_fake_lengths.to(device)
        
        # Evaluate real predictor
        real_outputs = predictor_real(test_real[:, :-1], test_real_lengths-1)
        real_mse = criterion(real_outputs, test_real[:, 1:]).item()
        
        # Evaluate fake predictor
        fake_outputs = predictor_fake(test_fake[:, :-1], test_fake_lengths-1)
        fake_mse = criterion(fake_outputs, test_fake[:, 1:]).item()
    
    return abs(real_mse - fake_mse)

class VAEMetricsTracker:
    def __init__(self):
        # Batch-level metrics that need to be averaged
        self.batch_metrics = {
            'total_loss': 0,
            'reconstruction_loss': 0,
            'kl_loss': 0,
            'z_mean': 0,
            'z_logvar': 0,
            'z_mean_std': 0,
            'z_logvar_std': 0,
        }
        self.num_batches = 0
        self.real_features = []
        self.fake_features = []
    
    def update(self, batch_metrics, X=None, X_recons=None):
        """
        Update batch-level metrics and store features
        Args:
            batch_metrics: Dictionary of batch-level metrics
            X: Original input data (optional)
            X_recons: Reconstructed data (optional)
        """
        for key, value in batch_metrics.items():
            if key in self.batch_metrics:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.batch_metrics[key] += value
        
        self.num_batches += 1
        
        # Store features for global metric computation
        if X is not None and X_recons is not None:
            self.real_features.append(X.detach().cpu().numpy())
            self.fake_features.append(X_recons.detach().cpu().numpy())
    
    def get_average_metrics(self):
        """
        Get averaged batch metrics and compute global metrics
        Returns:
            dict: Dictionary containing both averaged batch metrics and computed global metrics
        """
        # Average batch-level metrics
        metrics = {
            key: value / self.num_batches 
            for key, value in self.batch_metrics.items()
        }
        
        # Compute global metrics if features are available
        if len(self.real_features) > 0 and len(self.fake_features) > 0:
            real_features = np.concatenate(self.real_features)
            fake_features = np.concatenate(self.fake_features)
            
            # Add global metrics
            metrics['discriminative_score'] = discriminative_score(real_features, fake_features)
            metrics['predictive_score'] = predictive_score(real_features, fake_features)
        
        return metrics
    
    def reset(self):
        """
        Reset all metrics and stored features
        """
        # Reset batch metrics
        for key in self.batch_metrics:
            self.batch_metrics[key] = 0
            
        self.num_batches = 0
        self.real_features = []
        self.fake_features = []
    
    def compute_batch_metrics(self, z_mean, z_log_var, loss, recon_loss, kl_loss, batch_size):
        """
        Compute batch-level metrics
        Note: loss values are already normalized by batch_size in VAE
        """
        return {
            'total_loss': loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'z_mean': torch.mean(z_mean),
            'z_logvar': torch.mean(z_log_var),
            'z_mean_std': torch.std(z_mean),
            'z_logvar_std': torch.std(z_log_var)
        }
    
    def format_metrics(self, metrics):
        """
        Format metrics for printing
        """
        return " | ".join([f"{key}: {value:.4f}" for key, value in metrics.items()]) 