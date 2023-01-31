import numpy as np
import torch


class LinearRegression:
    """
    A linear regression model that uses analytical solution to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit linear regression model using analytical solution

        Arguments:
            X (np.ndarray): The covariates for training, y (np.ndarray): The corresponding paired values.

        Returns:
            Nothing.
        """
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        xTx = X.T @ X

        if np.abs(np.linalg.det(xTx)) < 1e-10:
            raise Exception("Non-invertible matrix; no analytical solution")

        result = np.linalg.inv(xTx) @ X.T @ y

        self.w, self.b = result[0:-1], result[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fit linear regression model using gradient descent

        Arguments:
            X (np.ndarray): The covariates for training, y (np.ndarray): The corresponding paired values.

        Returns:
            Nothing.
        """
        # X = np.hstack((X, np.ones((X.shape[0], 1))))

        # self.w = np.ones(X.shape[1])
        # for i in range(epochs):
        #     grad = -2 * X.T @ y + 2 * X.T @ X @ self.w
        #     print(f"X shape {X.shape}")
        #     print(f"grad shape {grad.shape}")

        #     self.w -= lr * grad
        X = torch.tensor(X).float()
        y = torch.tensor(y).float()

        self.w = torch.rand(X.shape[1], requires_grad=True).float()
        self.b = torch.rand(1, requires_grad=True).float()

        for i in range(epochs):
            loss = ((X @ self.w + self.b - y) ** 2).sum()
            loss.backward()

            with torch.no_grad():
                self.w -= lr * self.w.grad
                self.b -= lr * self.b.grad
                self.w.grad.zero_()
                self.b.grad.zero_()

        self.w = self.w.detach().numpy()
        self.b = self.b.detach().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X @ self.w + self.b
