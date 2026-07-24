# Binary Pattern Classifier (Perceptron)

A single perceptron trained on a logic gate (AND / OR), served through a small Streamlit app.

## Files
- `train.py` — trains the perceptron and saves `model.json`
- `app.py` — Streamlit UI that loads `model.json` and predicts
- `model.json` — the saved weights and bias

## How to run
```bash
python train.py        # trains and saves the model
streamlit run app.py   # opens the web app
```

To train the OR gate instead, change `GATE = "AND"` to `GATE = "OR"` in `train.py` and run it again.

## How it works (short version)
1. A perceptron takes the two inputs, multiplies each by a weight, adds a bias:
   `z = w1*x1 + w2*x2 + b`
2. The **binary step function** decides the class: if `z >= 0` output **1**, else **0**.
3. Training repeats: predict → find the error (`target - prediction`) → and only when
   it is wrong, correct the weights with `w = w + lr * error * x`.
4. The moment a full pass over all 4 rows happens with zero mistakes, training stops.

The app also shows `sigmoid(z)` as a probability. Note that `sigmoid(z) >= 0.5` is true
exactly when `z >= 0`, so it always agrees with the step function — it just gives a
nicer 0 to 1 number to look at.

## Sample result (AND gate) — converges in 6 epochs

| Epoch | w1 | w2 | bias | mistakes |
|---|---|---|---|---|
| 1 | 1 | 1 | 0 | 2 |
| 2 | 2 | 1 | -1 | 3 |
| 3 | 2 | 1 | -2 | 3 |
| 4 | 2 | 2 | -2 | 2 |
| 5 | 2 | 1 | -3 | 1 |
| 6 | 2 | 1 | -3 | **0** |

Final: **w1 = 2, w2 = 1, b = -3**

Check it by hand — only `(1,1)` gives `2 + 1 - 3 = 0`, which is `>= 0` so it outputs 1.
Every other input goes negative and outputs 0. That is exactly the AND gate.
