import torch
from torch.utils.data import Dataset
import torchvision

# Tatjana Meier


class MnistPairs(Dataset):
    """Dataset with Mnist pairs."""

    def __init__(self, root, train, download, transform=None, order='right', return_original_labels=False):
        """
        Args:
            root (string): Directory to store the downloaded MNIST dataset.
            train (bool): If True, use the training part of the MNIST dataset.
            download(bool): If True, will download the dataset, if it is not in the root folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            order (str): Indicates which ordering of digits to use, ['right', 'left'].
            return_original_labels (bool): Indicates if it is needed to return the original MNIST labels.
        """
        
        assert order in ['right', 'left'], "Got unexpected order argument. Expected one of ['right', 'left']"
        self.order = order
        
        self.return_original_labels = return_original_labels
        
        self.mnist_dataset = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform)

        # at the moment it doesn't bother the order of the images in the following definition
        # it gets then crucial at the get method
        self.list_of_pairs = []
        for index in range(0, len(self.mnist_dataset), 2):
            self.list_of_pairs.append((self.mnist_dataset[index], self.mnist_dataset[index+1]))

    def __len__(self):
        # MnistPairs should be half the size of the MNIST dataset
        return len(self.mnist_dataset) // 2

    def __getitem__(self, idx):
        # The ith element of the MnistPairs class
        # is a pair of subsequent MNIST dataset samples.
        # That is if MNIST is [a, b, c, d], then MnistPairs
        # with the 'right' order are [[a, b], [c, d]],
        # and [[b, a], [d, c]] for the 'left' order.
        # The label is mod 10 sum of the MNIST labels.

        ith_tuple = self.list_of_pairs[idx]

        first_image = ith_tuple[0][0]
        first_label = ith_tuple[0][1]
        second_image = ith_tuple[1][0]
        second_label = ith_tuple[1][1]
        label = int(first_label + second_label) % 10
        
        # at the moment list of tuples is arranged in right order
        # if left order is desired, change the order of the images

        if self.order == 'left':
            first_image, second_image = second_image, first_image
            first_label, second_label = second_label, first_label
        
        if self.return_original_labels:
            return first_image, second_image, label, first_label, second_label
        
        return first_image, second_image, label
