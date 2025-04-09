import torch


class Config:
    def __init__(self):
        # Data
        self.data_path = "data.csv"
        self.model_save_path = "TextCNN_best.pth"
        self.log_path = "logs"
        self.train_data_rate = 0.8
        self.batch_size = 64

        # Tokenizer
        self.min_freq = 2
        self.text_size = 32

        # Model
        self.is_Trained = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 512
        self.num_filter = 128
        self.kernel_size = [3, 4, 5]
        self.num_classes = 3
        self.epoch_num = 25
        self.LR = 1e-5
