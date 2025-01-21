
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO

class MedMNISTDatasetLoader:
    def __init__(self, dataset_name, batch_size=32, shuffle_train=True, shuffle_val=False, shuffle_test=False, num_workers=4):
        """
        Initialize the MedMNIST dataset loader.

        Args:
            dataset_name (str): The key of the dataset in the MedMNIST INFO dictionary (e.g., 'organmnist3d').
            batch_size (int): Batch size for the data loaders.
            shuffle_train (bool): Whether to shuffle the training dataset.
            shuffle_val (bool): Whether to shuffle the validation dataset.
            shuffle_test (bool): Whether to shuffle the test dataset.
        """
        if dataset_name not in INFO:
            raise ValueError(f"Dataset '{dataset_name}' is not available in MedMNIST INFO.")

        self.info = INFO[dataset_name]
        self.DataClass = getattr(medmnist, self.info["python_class"])
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test
        self.num_workers = num_workers

    def prepare_data_loaders(self):
        """
        Prepare and return data loaders for train, validation, and test splits.

        Returns:
            tuple: A tuple containing train_loader, val_loader, and test_loader.
        """
        train_dataset = self.DataClass(split="train", download=True)
        val_dataset = self.DataClass(split="val", download=True)
        test_dataset = self.DataClass(split="test", download=True)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_train, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle_val, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle_test, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader
