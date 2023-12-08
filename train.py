import triton_fvae


if __name__ == "__main__":
    model = triton_fvae.train(latent_dim=50)
