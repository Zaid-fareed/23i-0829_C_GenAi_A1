import torch, mlflow, optuna, onnxruntime, pytorch_msssim
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("optuna", optuna.__version__, "| onnxruntime", onnxruntime.__version__)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("smoke_test")
with mlflow.start_run():
    mlflow.log_param("ok", True)
    mlflow.log_metric("loss", 0.123, step=0)
print("MLflow logging works. View with: mlflow ui --port 5000")
