# lncLocator2

This is a repository of codes of lncLocator 2.0, an end-to-end lncRNA subcellular localization predictor. You can use this program and know more about it through our [website](http://www.csbio.sjtu.edu.cn/bioinf/lncLocator2/).

## Setting the environment

The environment on our computer is as follows:
* Python 3.6.8
* PyTorch 1.3.1
* Pandas 1.0.1
* NumPy 1.18.1
* scikit-learn 0.22.1

## Usage

Run the `main.py` to train and test lncLocator 2:
```
python main.py
```

And you can configure the training yourself. For example, change the directory of dataset by:
```
python main.py --train_dataset=path/to/your/train/dataset --dev_dataset=path/to/your/dev/dataset --test_dataset=path/to/your/test/dataset
```
Check the `config.py` to see what you could adjust.

Training log will be recorded in `train.log` at the root directory of the project. The curves of loss/auroc/accuracy versus epoch will be drawn and saved at the root directory of the project, titled `loss.png`, `auroc.png` and `acc.png` respectively. The prediction result for test set in every epoch will be saved at the root directory of the project, titled `record.csv`.
