# backend/config.py

models = {
    "MSE": "mse_model",
    "SC": "style_model",
    "MSE_SC": "mse_and_style",
    "cGAN_MAE": "gan_generator",
}

synthetics = {
    "GAN_MAE": "gan_mae_weights",
    "MSE_VGG": "mse_vgg_weights",
    "MSE": "mse_weights"
}