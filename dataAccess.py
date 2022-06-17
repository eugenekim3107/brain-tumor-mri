from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

ROOT_PATH = "./"

cities = ["austin", "miami", "pittsburgh", "dearborn", "washington-dc",
          "palo-alto"]

splits = ["train", "val", "test"]

def get_city_trajectories(city="palo-alto", split="train", normalized=False):
    outputs = None

    if split == "train":
        f_in = ROOT_PATH + split + "/" + city + "_inputs"
        inputs = pickle.load(open(f_in, "rb"))
        n = len(inputs)
        inputs = np.asarray(inputs)[:int(n * 0.8)]

        f_out = ROOT_PATH + split + "/" + city + "_outputs"
        outputs = pickle.load(open(f_out, "rb"))
        outputs = np.asarray(outputs)[:int(n * 0.8)]

    elif split == "val":
        f_in = ROOT_PATH + 'train' + "/" + city + "_inputs"
        inputs = pickle.load(open(f_in, "rb"))
        n = len(inputs)
        inputs = np.asarray(inputs)[int(n * 0.8):]

        f_out = ROOT_PATH + 'train' + "/" + city + "_outputs"
        outputs = pickle.load(open(f_out, "rb"))
        outputs = np.asarray(outputs)[int(n * 0.8):]

    else:
        f_in = ROOT_PATH + split + "/" + city + "_inputs"
        inputs = pickle.load(open(f_in, "rb"))
        n = len(inputs)
        inputs = np.asarray(inputs)

    return inputs, outputs


class ArgoverseDataset(Dataset):
    """Dataset class for Argoverse"""

    def __init__(self, city: str, split: str, transform=None):
        super(ArgoverseDataset, self).__init__()
        self.transform = transform

        self.inputs, self.outputs = get_city_trajectories(city=city,
                                                          split=split,
                                                          normalized=False)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        data = (self.inputs[idx], self.outputs[idx])

        if self.transform:
            data = self.transform(data)

        return data

# intialize a dataset
city = 'palo-alto' # choose desired city
split = 'train' # choose either train, val, test
train_dataset = ArgoverseDataset(city=city, split=split)
val_dataset = ArgoverseDataset(city)