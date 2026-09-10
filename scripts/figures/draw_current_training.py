"""功能概述：生成当前共享谱编码器训练和检索流程的可编辑论文示意图。
主要职责：text/box/arrow 记录统一点坐标布局并绘制矢量对象；main 输出 PDF、SVG、PNG
及供原生 PPTX 导出器消费的 JSON。布局只描述实现，不生成虚构数据或性能结果。
代码库关系：依据 raman_top10_experiment 的 64×96 多正样本目标及 domain_method，
替代历史 spectrum_graph_v2 图；当前图由 domain_method 的 figure 环境引入。
输入输出与副作用：无网络或训练数据读取；写入 output/figures/revised，执行对齐审计；
最低字号为 7 pt，所有几何与文字使用同一布局源以保持论文与 PPTX 一致。
"""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/figures/revised'
W, H = 515., 268.
INK, BLUE, TEAL, AMBER = '#263746', '#E5EEF7', '#E2F0EC', '#F5EBD9'
items = []
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7.2,
                     'pdf.fonttype':42,'svg.fonttype':'none'})
fig, ax = plt.subplots(figsize=(W/72,H/72))
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')

def text(x,y,s,size=7.2,bold=False,align='left'):
    items.append(dict(kind='text',x=x,y=y,text=s,size=size,bold=bold,align=align))
    ax.text(x,y,s,fontsize=size,weight='bold' if bold else 'normal',ha=align,va='center',
            color=INK,linespacing=1.35)

def box(x,y,w,h,s,color=BLUE):
    items.append(dict(kind='box',x=x,y=y,w=w,h=h,color=color))
    ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='#9EAFBC',linewidth=.6))
    text(x+w/2,y+h/2,s,align='center')

def arrow(x1,y1,x2,y2):
    items.append(dict(kind='arrow',x1=x1,y1=y1,x2=x2,y2=y2))
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=7,
                               linewidth=.7,color=INK,shrinkA=0,shrinkB=0))

def rule(y):
    items.append(dict(kind='rule',y=y))
    ax.plot([6,W-6],[y,y],color='#CCD4DA',lw=.65)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    text(6,10,'a  Training inputs and shared spectral encoder',8.4,True)
    box(6,25,118,28,'32 augmented theory queries\nSource-dependent proxy views')
    box(6,63,118,28,'32 real experimental queries\nExact-identity pairs',TEAL)
    box(6,101,118,28,'96 theoretical references\n64 matched + 32 extra')
    box(159,35,174,76,'Shared masked Transformer\n3,501 bins · 500–4,000 cm⁻¹\n16-bin patches · stride 8\n3 layers · 4 heads · width 128\nCoverage-weighted pooling')
    arrow(124,39,156,49);arrow(124,77,156,73);arrow(124,115,156,101)
    box(368,38,140,31,'64 query embeddings U\nUnit vectors in 64 dimensions',TEAL)
    box(368,83,140,31,'96 reference embeddings V\nSame encoder and projection')
    arrow(333,53,365,53);arrow(333,98,365,98)
    text(159,123,'Proxy fit: training pairs only; fixed or randomized views',7)
    rule(138)
    text(6,150,'b  Identity-aware contrastive objective',8.4,True)
    text(6,167,'Scores = UVᵀ / τ,  τ = 0.1',7.6,True)
    text(6,181,'Positive set: all references with the same full identity')
    text(6,195,'L = ½ (L query→reference + L reference→query)',7.6,True)
    text(293,166,'Sum positive probability mass before taking −log')
    text(293,180,'Reverse average uses supported references only')
    text(293,194,'Extras: random, or sampled from top-16 wrong matches')
    rule(207)
    text(6,219,'c  Frozen inference and constrained selection',8.4,True)
    box(6,233,81,27,'Experimental\nquery + mask',TEAL)
    box(105,233,110,27,'Frozen spectrum index\n10,000 structures')
    box(233,233,98,27,'Top 10 structures\n+ query peaks')
    box(349,233,80,27,'LLM selection\nOptional evidence',AMBER)
    box(447,233,61,27,'One candidate\nindex',TEAL)
    for a,b in [(87,105),(215,233),(331,349),(429,447)]:arrow(a+2,246,b-2,246)
    fig.canvas.draw()
    sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig,json_out=str(OUT/'training.alignment.json'),strict=True)
    fig.savefig(OUT/'raman_model_training_current.pdf')
    fig.savefig(OUT/'raman_model_training_current.svg')
    fig.savefig(OUT/'raman_model_training_current.png',dpi=600)
    (OUT/'layout.json').write_text(json.dumps(dict(width=W,height=H,items=items),indent=2))
    plt.close(fig)

if __name__ == '__main__':
    main()
