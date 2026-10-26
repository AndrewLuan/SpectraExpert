"""独审队列的组合数量、预算、输入角色、依赖和无test边界；只构造内存DAG。"""
from pathlib import Path
import sys
sys.path.insert(0,'/Users/andrewluan/Desktop/Projects/IRfingerprints/SpectrExpert')
from build_pool_queue import build
from nmr_transfer.h_research_driver import validate_queue
D=Path(__file__).parent
R=Path('/Users/andrewluan/Desktop/Projects/IRfingerprints/SpectrExpert')
for phase,total,fitcount in [('smoke',4,4),('formal',48,16),('diagnostics',12,0)]:
 q=build(D,R,phase);validate_queue(q);assert len(q['jobs'])==total
 fits=[j for j in q['jobs'] if j['kind']=='training'];assert len(fits)==fitcount
 for j in q['jobs']:
  a=j['argv'];assert not any('/test/' in x or '/verify/' in x for x in a)
  if j['kind']=='validation':
   assert any('/original_val/' in x or '/tune/' in x for x in a)
   assert len(j['deps'])==(2 if phase=='formal' else 0)
  if phase=='formal' and j['kind']=='training':
   for flag,value in {'--epochs':'64','--samples':'8192','--batch-size':'32','--anchor':'0.1','--lr':'0.001'}.items():assert a[a.index(flag)+1]==value
   initial=a[a.index('--initial-checkpoint')+1]
   assert 'factor_R0O1F0' in initial
   assert ('/aug/' in initial)==('--augment' in a)
 print(phase,len(q['jobs']),'PASS')
