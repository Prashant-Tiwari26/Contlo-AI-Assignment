from GPT2Model import GPT2
from transformers import GPT2Tokenizer

def Run():
    model = GPT2()
    model.from_pretrained()
    print(model)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    encoded = tokenizer("the quick brown fox jumps over the ", return_tensors='pt')
    tokens = encoded['input_ids']

    added = model.generate(tokens, 2)
    print(tokenizer.decode(added.squeeze(dim=0)))

if __name__ == '__main__':
    Run()