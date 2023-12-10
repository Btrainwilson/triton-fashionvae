from . import dataset
from . import model as fashion_models
from torch.utils.data import DataLoader
import torch
import time

def train_loop(batch_size,
               lr,
               device,
               epochs=10,
               num_workers=0,
               pin_memory=False,
               benchmark = False,
               latent_dim = 2,
               ):
  torch.backends.cudnn.benchmark = benchmark
  train_data, test_data = dataset.download_fashion_mnist()
  train_loader = DataLoader(dataset=train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
  start_time = time.time()

  model = fashion_models.VAE(latent_dim=latent_dim).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)


  # Training
  for epoch in range(epochs):
      for i, (x, _) in enumerate(train_loader):
          x = x.to(device)
          x_reconst, mu, log_var = model(x)

          recon_loss, kl_div = fashion_models.vae_loss(x_reconst.flatten(start_dim=1), x, mu, log_var)

          optimizer.zero_grad()
          (recon_loss + kl_div).backward()
          optimizer.step()

          if (i+1) % 100 == 0:
              print(f"Epoch[{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], KL Divergence: {kl_div.item():.4f}, Recon Loss: {recon_loss.item():.4f}")

  end_time = time.time()
  return model, end_time - start_time
