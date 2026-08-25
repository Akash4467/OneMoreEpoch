import numpy as np

from onemoreepoch import Tensor
from onemoreepoch.nn import Linear, MSELoss, ReLU, Sequential
from onemoreepoch.optim import SGD
from onemoreepoch.utils import TrainingLogger


# Trains a small MLP to learn y = 2x + 1 and prints progress plus a final prediction
def main() -> None:
    np.random.seed(0)
    x_data = np.random.randn(128, 1)
    y_data = 2 * x_data + 1 + 0.01 * np.random.randn(128, 1)
    x, y = Tensor(x_data), Tensor(y_data)

    model = Sequential(Linear(1, 16), ReLU(), Linear(16, 1))
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.05)
    logger = TrainingLogger(log_every=20)

    for epoch in range(1, 201):
        prediction = model(x)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log_epoch(epoch, loss.item())

    logger.finish()

    test = Tensor(np.array([[3.0]]))
    print(f"model(3.0) = {model(test).item():.4f}  (target: 7.0)")


if __name__ == "__main__":
    main()
