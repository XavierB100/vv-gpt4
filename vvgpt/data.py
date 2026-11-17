"""Non-destructive preparation and chronological train/validation/test partitions."""
import hashlib
import re
import unicodedata
from collections import Counter

STAMP = re.compile(r'^\[?(\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*(?:[-–]\s*)?([^:\n]{1,80}):\s*(.*)$')


def fingerprint(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def prepare(text, deduplicate=True):
    original = text
    text = unicodedata.normalize('NFC', text.replace('\r\n','\n').replace('\r','\n')).lstrip('\ufeff')
    # Strip only invisible direction controls used by chat exports, not punctuation/emoji/code.
    text = text.translate(dict.fromkeys(map(ord,'\u200e\u200f\u202a\u202c'), None))
    lines = text.splitlines()
    matched = sum(bool(STAMP.match(line)) for line in lines)
    kind = 'whatsapp' if matched >= 2 else 'text'
    messages = []
    if kind == 'whatsapp':
        for line in lines:
            match = STAMP.match(line)
            if match:
                _,_,sender,body = match.groups()
                # Sender labels remain consistent with the generation prompt.
                messages.append(f'{sender.strip()}: {body}')
            elif messages and line.strip():
                # System notices carry a timestamp but no sender; do not join them to someone's words.
                if not re.match(r'^\[?\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}', line):
                    messages[-1] += '\n'+line
        text = '\n'.join(messages)
    else:
        text = '\n'.join(line.rstrip() for line in lines)
    blocks = re.split(r'\n\s*\n', text)
    counts = Counter(re.sub(r'\s+',' ',b).strip() for b in blocks if len(b.strip()) >= 120)
    duplicates = sum(n-1 for n in counts.values())
    seen, kept, removed = set(), [], 0
    for block in blocks:
        key = re.sub(r'\s+',' ',block).strip()
        if deduplicate and kind == 'text' and len(key)>=120 and key in seen:
            removed += 1
            continue
        seen.add(key)
        if block.strip(): kept.append(block.strip())
    text = '\n\n'.join(kept).strip()
    if not text: raise ValueError('No usable text remains after preparation.')
    report = {'format':kind,'original_chars':len(original),'prepared_chars':len(text),
              'messages':len(messages),'duplicate_long_blocks':duplicates,'removed_blocks':removed,
              'source_sha256':fingerprint(original),'prepared_sha256':fingerprint(text)}
    report['warnings'] = []
    if len(text)<20000: report['warnings'].append('Small corpus: expect limited vocabulary and repetition. More varied text matters more than a larger model.')
    if duplicates: report['warnings'].append(f'{duplicates} repeated long blocks found. Short repeated dialogue is retained.')
    return text, report


def split_text(text):
    """80/10/10 contiguous split, preferring line boundaries; no overlapping text."""
    def boundary(fraction):
        at = int(len(text)*fraction)
        newline = text.find('\n',at,min(len(text),at+500))
        return newline+1 if newline!=-1 else at
    first,second = boundary(.8),boundary(.9)
    if not 0<first<second<len(text): raise ValueError('Not enough text for three separate splits.')
    return {'train':text[:first],'validation':text[first:second],'test':text[second:]}
