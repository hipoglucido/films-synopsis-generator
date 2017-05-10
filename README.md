# Films Synopsis Generator

## Summary
The goal of this project is to build a model that is able to generate different film synopsys from a set of predefined genres. The model is LSTM [one to many](http://karpathy.github.io/assets/rnn/diags.jpeg)

For training the net we will be using a dataset of >100K pairs of <genres,synopsis> (data is in spanish).

## Project folder structure
Lets put the code in the `src` folder and all the input and output data in the `data` folder (without pushing any data to the repo).
```
.
├── data
│   ├── input
│   │   ├── datasets
│   │   │   ├── raw
│   │   │   ├── test
│   │   │   └── train
│   │   └── weights
│   └── output
│       ├── tensorboard_logs
│       └── weights
├── proposal.pdf
├── README.md
└── src
    └── data_loading_example.py
```


## Useful links
- RNN basics: [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- LSTM basics: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- At prediction time we will use Beam Search to generate various synopsis from the same set of genres: [Beam Search video](https://www.youtube.com/watch?v=UXW6Cs82UKo)
- Repo that uses LSTM + beamsearch to generate headlines of paragraphs ([link](https://github.com/udibr/headlines))