"""Bounded public data download; never downloads model weights."""
import json
import hashlib
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vvgpt.lab import add_partitioned_dataset
from vvgpt.store import Store
from vvgpt.data import fingerprint

ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'runtime/research/stories';folder.mkdir(exist_ok=True)
manifest=folder/'manifest.json'
if manifest.exists():print('Story dataset already prepared.');raise SystemExit(0)
revision='f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
url=f'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{revision}/TinyStoriesV2-GPT4-train.txt'
stories=[];seen=set();pending=[];received=0
# Stop immediately after the required complete unique records; cap network reads at 64 MiB.
with urllib.request.urlopen(url,timeout=60) as response:
    for line in response:
        received+=len(line)
        if received>64*1024*1024:raise RuntimeError('Download bound reached before enough complete stories.')
        text=line.decode('utf-8')
        if '<|endoftext|>' in text:
            before,after=text.split('<|endoftext|>',1);pending.append(before)
            story=''.join(pending).strip();pending=[after]
            digest=fingerprint(story)
            if story and digest not in seen:
                seen.add(digest);stories.append(story)
            if len(stories)>=21000:break
        else:pending.append(text)
assert len(stories)==21000
splits={key:'\n<|endoftext|>\n'.join(items)+'\n<|endoftext|>\n' for key,items in [
 ('train',stories[:20000]),('validation',stories[20000:20500]),('test',stories[20500:21000]) ]}
provenance={'dataset':'roneneldan/TinyStories','revision':revision,'url':url,'license':'CDLA-Sharing-1.0',
 'license_url':'https://cdla.dev/sharing-1-0/','counts':{'train':20000,'validation':500,'test':500},
 'downloaded_bytes':received,'deduplicated':'exact whole-story hash','separator':'<|endoftext|>','tokenizer_requirement':'compact recognizes EOS separators'}
did=add_partitioned_dataset(Store(ROOT/'runtime'),'Simple stories / clean-language study',splits,provenance)
info={'dataset':did,'provenance':provenance,'hashes':{k:fingerprint(v) for k,v in splits.items()},'runs':[],
 'prompts':['Lily put her red ball in a box. Then she went outside. When she came back, she looked in the box and',
 'Tom was hungry, so he went to the kitchen. His mum gave him a bowl of soup. Tom took a spoon and',
 'Ben and Anna found a little bird with a hurt wing. They wanted to help it, so',
 'It was raining outside. Mia wanted to stay dry, so she took her',
 'Sam had two apples. He gave one to his friend. Now Sam had',
 'A little dog was lost in the park. A girl saw him and said,',
 'Ella built a tall tower with her blocks. Her baby brother knocked it down. Ella felt',
 'Max planted a seed in a small pot. Every day he gave it water. After a week,'],
 'seeds':[42,123]}
manifest.write_text(json.dumps(info,indent=2)+'\n')
(folder/'LICENSE.txt').write_text('TinyStories dataset: CDLA-Sharing-1.0\nhttps://cdla.dev/sharing-1-0/\nAuthor dataset: https://huggingface.co/datasets/roneneldan/TinyStories\nRevision '+revision+'\n')
print(json.dumps({'dataset':did,'downloaded_bytes':received,'stories':len(stories),'split_chars':{k:len(v) for k,v in splits.items()}}))
