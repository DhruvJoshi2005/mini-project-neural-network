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
2. `sigmoid(z)` squashes `z` into a value between 0 and 1 — that's the probability.
3. If the probability is >= 0.5 we call it class **1**, otherwise class **0**.
4. Training just repeats: predict → measure the error (`target - prediction`) → nudge
   the weights and bias a little in the direction that reduces that error.
   After enough rounds the weights settle on values that get all 4 rows right.

## Sample result (AND gate)
w1 = 12.09, w2 = 12.09, b = -18.30

Both inputs must be 1 for the sum to beat the bias — which is exactly the AND rule.
