# Lottery Ticket Hypothesis: Activation Functions & Pruning

To run this project, first you need to set up a proper environment.

To do this, run the following commands:

```
python3 -m venv lth_env
source lth_env/bin/activate
pip install -r requirements.txt
```

You could also have a python compatible compiler (Such as VSC) do it for you.


---
## Training the Models:

With the way the code is set up, you can train/prune models using all implemented 
activation functions using the command:

```
python main.py --train
```

---
## Analyzing the Data:

To gather the resulting model metrics, use the command:

```
python main.py --analyze
```

---
## Generating Plots

To set up plots/graphs using these result metrics, use the command:

```
python make_plots.py
```

After that, the result plots should be available to see/open inside the project folder.
