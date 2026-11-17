"""Tokenizers contain vocabulary rules only. No pretrained model is downloaded."""
import json


class Codec:
    def __init__(self, specification=None):
        self.spec = specification or {'kind':'gpt2'}
        if self.spec['kind']=='compact':
            from tokenizers import Tokenizer
            self.engine = Tokenizer.from_str(self.spec['json'])
            self.vocab_size = self.engine.get_vocab_size()
            self.eos = self.engine.token_to_id('<|endoftext|>')
        else:
            import tiktoken
            self.engine = tiktoken.get_encoding('gpt2')
            self.vocab_size,self.eos = self.engine.n_vocab,self.engine.eot_token

    def encode(self,text):
        return self.engine.encode(text).ids if self.spec['kind']=='compact' else self.engine.encode_ordinary(text)

    def decode(self,ids):
        return self.engine.decode(ids,skip_special_tokens=False) if self.spec['kind']=='compact' else self.engine.decode(ids)


def fit_codec(train_text, kind='compact'):
    if kind=='gpt2':return Codec()
    if kind!='compact':raise ValueError('Choose compact or GPT-2 tokenization.')
    from tokenizers import Tokenizer,models,pre_tokenizers,decoders,trainers
    engine=Tokenizer(models.BPE())
    engine.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=True)
    engine.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=4096,min_frequency=2,show_progress=False,
                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=['<|endoftext|>'])
    # Fit exclusively on training text. Validation/test text never reaches this trainer.
    engine.train_from_iterator([train_text],trainer=trainer)
    return Codec({'kind':'compact','json':engine.to_str()})
