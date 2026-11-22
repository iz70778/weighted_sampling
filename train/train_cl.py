from data.data_loder import get_cl_train_dataloader
from train.train_utils import train_diff_model

if __name__ == '__main__':
    train_dataloader = get_cl_train_dataloader()

    model_name = "cl"

    train_diff_model(train_dataloader, model_name)

