# BERT-Keyword-Extractor
Use BERT Token Classification Model to extract keywords from a sentence.

Parameters tuned as specified in BERT by Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.

## Hyper-parameter Tuning

I ran ablation experiments and these are the results. I suggest to use parameters in line 4.
All training was done on batch size of 32.

| Learning Rate 	| Number of Epochs 	| Validation loss 	| Validation Accuracy 	| F1-Score     	|
|---------------	|------------------	|-----------------	|---------------------	|--------------	|
| 3.00E-05      	| 3                	| 0.05294724515   	| 98.30%              	| 0.5318559557 	|
| 5.00E-05      	| 3                	| 0.04899719357   	| 98.47%              	| 0.56218628   	|
| 2.00E-05      	| 3                	| 0.05733459462   	| 98.15%              	| 0.4390547264 	|
| 3.00E-05      	| 4                	| 0.05020467712   	| **98.48%**              	| 0.5528169014 	|
| 5.00E-05      	| 4                	| 0.05194576555   	| 98.43%              	| 0.5780836421 	|
| 2.00E-05      	| 4                	| 0.05373481681   	| 98.25%              	| 0.5019740553 	|

The `main.py` script can be utilized for training and accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --lr LR            initial learning rate
  --epochs EPOCHS    upper epoch limit
  --batch_size N     batch size
  --seq_len N        sequence length
  --save SAVE        path to save the final model
```
Example:

```bash
python main.py --data "maui-semeval2010-train" --lr 2e-5 --batch_size 32 --save "model.pt" --epochs 3           
```

The `keyword-extractor.py` script can be used to extract keywords from a sentence and accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --sentence SEN        sentence to extract keywords
  --path LOAD        path to load model from
```

Example:

```bash
python keyword-extractor.py --sentence "BERT is a great model." --path "model.pt"           
```