import sentencepiece as spm
spm.SentencePieceTrainer.train(input='../data/text8_for_sp', model_prefix='m', vocab_size=60000)
