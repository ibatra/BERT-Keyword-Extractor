# Deep Keyphrase Extraction using BERT

I have used BERT Token Classification Model to extract keywords from a sentence. Feel free to clone and use it. If you face any problems, kindly post it on issues section.

Special credits to BERT authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, [original repo](https://github.com/google-research/bert) and Huggingface for PyTorch version [original repo](https://github.com/huggingface/pytorch-pretrained-BERT).

## Requirements

You need:

```
pytorch 1.0
python 3.6
pytorch-pretrained-bert 0.4.0
```

## Usage


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

## Training

You can also train it from scratch using BERT's pre-trained model. The `main.py` script can be utilized for training and accepts the following arguments:

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

This model has been trained on SemEval 2010 dataset (scientific publications). You can swap this with your own custom dataset.

## Code explanations

I have provided the explanation of keyphrase extraction in the form of python notebook which you can view [here](https://github.com/ibatra/BERT-Keyword-Extractor/blob/master/BERT-Keyword%20Extractor.ipynb)

## Hyper-parameter Tuning

I ran ablation experiments according to the BERT paper and these are the results. I suggest to use parameters in line 4.
All training was done on batch size of 32.

| Learning Rate 	| Number of Epochs 	| Validation loss 	| Validation Accuracy 	| F1-Score     	|
|---------------	|------------------	|-----------------	|---------------------	|--------------	|
| 3.00E-05      	| 3                	| 0.05294724515   	| 98.30%              	| 0.5318559557 	|
| 5.00E-05      	| 3                	| 0.04899719357   	| 98.47%              	| 0.56218628   	|
| 2.00E-05      	| 3                	| 0.05733459462   	| 98.15%              	| 0.4390547264 	|
| 3.00E-05      	| 4                	| 0.05020467712   	| **98.48%**              	| 0.5528169014 	|
| 5.00E-05      	| 4                	| 0.05194576555   	| 98.43%              	| 0.5780836421 	|
| 2.00E-05      	| 4                	| 0.05373481681   	| 98.25%              	| 0.5019740553 	|

