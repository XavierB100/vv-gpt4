import math
import torch


@torch.no_grad()
def score_text(model, codec, text, device, max_chars=8000):
    """Predict each held-out token once; normalize by the same raw UTF-8 bytes.

    Token loss/perplexity cannot be compared between different vocabularies.
    Each model uses its own context window; report that constraint with the score.
    """
    text=text[:max_chars]
    ids=[codec.eos]+codec.encode(text)
    size=model.config.block_size
    stride=max(1,size//2)
    total,count=0.,0
    was_training=model.training
    model.eval()
    for pos in range(1,len(ids),stride):
        end=min(len(ids),pos+stride)
        start=max(0,end-size-1)
        x=torch.tensor([ids[start:end-1]],device=device)
        y=torch.tensor([ids[start+1:end]],device=device)
        y[:,:pos-start-1]=-1
        loss=model(x,y)[1].item()
        total+=loss*(end-pos)
        count+=end-pos
    model.train(was_training)
    return {'bits_per_byte':total/(max(1,len(text.encode('utf-8')))*math.log(2)),
            'loss':total/max(1,count),'tokens':count,'bytes':len(text.encode('utf-8')),
            'chars':len(text),'context_tokens':size}
