"""Read-only, reproducible audit of the original local checkpoint families."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vvgpt.lab import LEGACY, load_checkpoint

report={'source':'XavierB100/vv-gpt3','commit':'802d1a8ac5efc3c7bc28f6b1e1ceeed09e7ad2b4','models':[]}
for path in sorted(LEGACY.glob('*.metadata.json')):
    metadata=json.loads(path.read_text())
    name=path.name.removesuffix('.metadata.json')
    result={'name':name,'metadata':metadata,'weights':{}}
    for variant,suffix in [('final',''),('best','_best'),('latest','_latest')]:
        file=LEGACY/f'{name}{suffix}.pt'
        checkpoint=load_checkpoint(file)
        digest=hashlib.sha256()
        for key,value in sorted(checkpoint['model'].items()):
            digest.update(key.encode());digest.update(value.contiguous().numpy().tobytes())
        result['weights'][variant]={'sha256':digest.hexdigest(),'saved_step':checkpoint.get('iter_num'),
                                    'saved_val_loss':checkpoint.get('val_loss'),'file_bytes':file.stat().st_size}
        del checkpoint
    report['models'].append(result)
destination=Path(__file__).resolve().parents[1]/'docs'/'legacy-evidence.json'
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([{'name':m['name'],'final_matches_best':m['weights']['final']['sha256']==m['weights']['best']['sha256'],
                   'final_matches_latest':m['weights']['final']['sha256']==m['weights']['latest']['sha256']} for m in report['models']],indent=2))
