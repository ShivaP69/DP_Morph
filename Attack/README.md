# Ruuning Attack model
This code includes multiple attack implementations; however, only the global loss results are reported in the paper. Additional modifications are required for the other attacks.

First install the following packages:

```bash
pip install torch torchvision tqdm numpy matplotlib seaborn pandas scikit-learn scipy kornia opacus pillow
```

For running the attacks:

For DUke:

```bash
 python3 main.py --main_dir "DukeData" --n_classes 9 --OUTPUT_CHANNELS 9 --DPSGD True --epsilon 8 --morphology True --operation "open" 
 
```
For UMN:

```bash
python3 main.py --main_dir "UMNData" --n_classes 2 --OUTPUT_CHANNELS 2 --DPSGD True --epsilon 8 --morphology True --operation "open" 
 ```
Also in the args.py,you should:
```bash
OUTPUT_CHANNELS=9# 9 for Duke dataset and 2 for UMN
```

For details on defining other parameters such as batch size and more, please refer to our paper.


## reproducing the results

To generate the comparison table of global loss-based attack performance, run the following command:

```bash
python3 global_loss_bar.py 
```
Then in the next step:
```bash
python3 csv_to_latex.py
```

As a result, you will obtain a LaTeX table comparing the global-loss-based attack on both UMN and Duke datasets, under different epsilon values, for both DP-morph and non-morph settings.