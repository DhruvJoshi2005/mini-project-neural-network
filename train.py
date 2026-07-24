"""
Trains a single perceptron on a logic gate (AND or OR)
and saves the learned weights + bias into model.json

Uses the classic perceptron learning rule with a binary step function,
so it finishes in just a few epochs.
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


def step(z):
    # binary step - if the weighted sum reaches 0 we say 1, otherwise 0
    return 1 if z >= 0 else 0


# start with zero weights and zero bias, the training loop will fix them
w = np.zeros(2)
b = 0.0

lr = 1           # learning rate - with a step function 1 works fine
max_epochs = 20  # safety limit, it normally converges way before this

history = []     # keeps a record of the weights after every epoch

for epoch in range(1, max_epochs + 1):
    mistakes = 0  # how many of the 4 rows we got wrong in this epoch

    for xi, target in zip(X, y):
        z = np.dot(w, xi) + b     # weighted sum of the inputs
        pred = step(z)            # 0 or 1
        error = target - pred     # 0 if correct, +1 or -1 if wrong

        # only touch the weights when we actually made a mistake
        if error != 0:
            w += lr * error * xi
            b += lr * error
            mistakes += 1

    # note down where the weights stand at the end of this epoch
    history.append({
        "epoch": epoch,
        "w1": float(w[0]),
        "w2": float(w[1]),
        "bias": float(b),
        "mistakes": mistakes,
    })
    print(f"epoch {epoch:2d} | w1={w[0]:5.1f}  w2={w[1]:5.1f}  b={b:5.1f} | mistakes={mistakes}")

    # no mistakes in a full pass means we are done, no need to keep going
    if mistakes == 0:
        print(f"\nConverged in {epoch} epochs!")
        break

# save the trained model so the streamlit app can just load it
model = {
    "gate": GATE,
    "weights": w.tolist(),
    "bias": b,
    "epochs_taken": len(history),
    "history": history,
}
with open("model.json", "w") as f:
    json.dump(model, f, indent=2)

print("\nweights :", w)
print("bias    :", b)

# quick sanity check - does it actually predict the gate correctly?
print("\nCheck:")
for xi, target in zip(X, y):
    z = np.dot(w, xi) + b
    print(f"{xi} -> {step(z)} (expected {target})")
