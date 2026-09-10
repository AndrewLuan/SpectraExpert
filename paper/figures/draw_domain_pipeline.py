"""功能概述：为十页最终稿绘制矢量版域适配与候选选择流程图。
主要职责：分开训练和推理，标明真实配对、代理视图、多正样本损失和测试边界。
代码库关系：生成同目录 domain_shift_pipeline.pdf，由 domain_method.tex 引用。
输入输出与副作用：无数据读取或网络调用；仅写确定性的论文示意图。
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7.15,2.85))
ax.set(xlim=(0,10),ylim=(0,4)); ax.axis('off')
navy='#1e3a5f'; blue='#dbeafe'; green='#dcfce7'; rose='#fce7f3'; amber='#fef3c7'
def box(x,y,w,h,label,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.08',facecolor=color,edgecolor=navy,linewidth=.7))
 ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=7,color=navy,linespacing=1.35)
def arrow(x1,y1,x2,y2):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=8,color=navy,linewidth=.8))
ax.text(.02,3.83,'TRAIN  |  Experimental groups only; no test measurements',fontsize=8.2,weight='bold',color=navy)
box(.05,2.8,2.15,.63,'Theory + source proxy\nfixed / randomized views',blue)
box(.05,1.92,2.15,.63,'Real experimental spectra\npaired by molecular identity',green)
box(2.9,2.36,2.1,.7,'Shared masked Transformer\ncoverage-weighted pooling',blue)
box(5.73,2.36,2.1,.7,'Multi-positive contrastive loss\nmatch full molecular identities',rose)
box(8.46,2.36,1.43,.7,'Theory embeddings\nsame encoder',blue)
arrow(2.22,3.12,2.87,2.87);arrow(2.22,2.23,2.87,2.57)
arrow(5.04,2.7,5.69,2.7);arrow(8.42,2.7,7.87,2.7)
ax.text(6.84,2.03,'32 extra references: random or hard negatives',ha='center',fontsize=6.7,color=navy)
ax.plot([.05,9.9],[1.67,1.67],color='#cbd5e1',linewidth=.9)
ax.text(.02,1.43,'INFER  |  Frozen encoder and 10,000 known library identities',fontsize=8.2,weight='bold',color=navy)
box(.05,.32,1.65,.72,'Experimental query\ncurve + coverage',green)
box(2.28,.32,1.76,.72,'Frozen retrieval\ntop 10 structures',blue)
box(4.63,.32,1.96,.72,'Peaks + candidates\noptional rule evidence',amber)
box(7.18,.32,1.21,.72,'LLM\nselection',rose)
box(8.96,.32,.94,.72,'One valid\nindex',green)
for a,b in [(1.74,2.24),(4.08,4.59),(6.63,7.14),(8.43,8.92)]:arrow(a,.68,b,.68)
ax.text(5.03,.03,'Separate retrieval coverage from correct selection; no structural modification',ha='center',fontsize=7,color=navy)
fig.subplots_adjust(left=.005,right=.995,top=.995,bottom=.015)
fig.savefig(Path(__file__).with_name('domain_shift_pipeline.pdf'))
plt.close(fig)
