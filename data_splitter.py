from typing import List, Tuple

from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, train_size: float, val_size: float, test_size: float):
        assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must add up to 1.0"
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def split(self, data: List, labels: List) -> Tuple[List, List, List, List, List, List]:
        assert len(data) == len(labels), "Data and labels must have the same length"
        x_train_val, x_test, y_train_val, y_test = train_test_split(data, labels, test_size=self.test_size,
                                                                    random_state=42)
        test_size = self.val_size / (self.train_size + self.val_size)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=test_size,
                                                          random_state=42)

        return x_train, y_train, x_val, y_val, x_test, y_test
