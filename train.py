"""
Trains a single perceptron on a logic gate (AND or OR)
and saves the learned weights + bias into model.json
"""

import json
import numpy as np

# change this to "OR" if you want to train the OR gate instead
GATE = "AND"

# the 4 possible input combinations of two binary inputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# expected output for each of the above inputs
if GATE == "AND":
    y = np.array([0, 0, 0, 1])
else:
    y = np.array([0, 1, 1, 1])


def sigmoid(z):
    # squashes any number into the range 0 to 1, so we can read it as a probability
    return 1 / (1 + np.exp(-z))


# start with zero weights and zero bias, the training loop will fix them
w = np.zeros(2)
b = 0.0

lr = 0.5        # learning rate - how big a step we take on every correction
epochs = 5000   # how many times we go over the whole dataset

history = []    # keeps a record of how the weights changed during training

for epoch in range(epochs):
    for xi, target in zip(X, y):
        z = np.dot(w, xi) + b     # weighted sum of the inputs
        pred = sigmoid(z)         # turn it into a probability
        error = target - pred     # how far off we were

        # nudge the weights and bias in the direction that reduces the error
        w += lr * error * xi
        b += lr * error

    # take a snapshot every 100 epochs so we can plot the progress later
    if epoch % 100 == 0:
        loss = np.mean((y - sigmoid(X @ w + b)) ** 2)
        history.append({
            "epoch": epoch,
            "w1": float(w[0]),
            "w2": float(w[1]),
            "bias": float(b),
            "loss": float(loss),
        })

        # print it once in a while just to see it learning
        if epoch % 1000 == 0:
            print(f"epoch {epoch:5d} | loss {loss:.4f}")

# one last snapshot so the graph ends at the final trained values
final_loss = np.mean((y - sigmoid(X @ w + b)) ** 2)
history.append({
    "epoch": epochs,
    "w1": float(w[0]),
    "w2": float(w[1]),
    "bias": float(b),
    "loss": float(final_loss),
})

# save the trained model so the streamlit app can just load it
model = {"gate": GATE, "weights": w.tolist(), "bias": b, "history": history}
with open("model.json", "w") as f:
    json.dump(model, f, indent=2)

print("\nTraining done!")
print("weights :", w)
print("bias    :", b)

# quick sanity check - does it actually predict the gate correctly?
print("\nCheck:")
for xi, target in zip(X, y):
    p = sigmoid(np.dot(w, xi) + b)
    print(f"{xi} -> {int(p >= 0.5)} (expected {target}, probability {p:.3f})")
