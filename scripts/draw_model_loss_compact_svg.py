"""功能概述：生成紧凑的论文模型与损失机制示意图。
主要职责：以原生 SVG 绘制共享编码器、匹配矩阵、正负样本关系与监督分布。
代码库关系：对应 spectrum_graph_v2 和 structure_losses 的当前训练配置。
输入输出与副作用：仅生成指定 SVG；图标与矩阵数值均为示意，不读取训练数据。
"""
import argparse
from html import escape
from pathlib import Path
import math
import re


def draw(output):
    p = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Raman–structure retrieval: dual encoder and four training mechanisms</title>
<desc>Compact schematic of two shared-weight spectrum encoder passes, a molecular graph encoder, and cross-modal, augmentation, structure-relation and auxiliary losses. All illustrated spectra, molecules and similarity matrices are schematic.</desc>
<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker></defs>
<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202932;font-size:18px} .h{font-size:23px;font-weight:700}.s{font-size:16px}.f{font-size:20px;font-family:Arial,Helvetica,sans-serif}.b{font-weight:700}.tiny{font-size:14px}</style>
<rect width="1600" height="1000" fill="white"/>''']
    blue, orange, green, purple = '#3977b5', '#cc8740', '#479575', '#8a6ab1'

    def text(x,y,s,c='',anchor='start',color=None):
        label = re.sub(r'\bL(cross|aug|struct|aux|total)\b', r'L<tspan baseline-shift="sub" font-size="72%">\1</tspan>', escape(s))
        p.append(f'<text x="{x}" y="{y}" class="{c}" text-anchor="{anchor}"'+(f' style="fill:{color}"' if color else '')+f'>{label}</text>')

    def box(x,y,w,h,fill='white',stroke='#b7c0c8',r=8,dash=False):
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"'+(' stroke-dasharray="5 4"' if dash else '')+'/>')

    def line(d,color='#58636d',dash=False,both=False,arrow=True):
        p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8"'+(' stroke-dasharray="5 4"' if dash else '')+(' marker-end="url(#arr)"' if arrow else '')+(' marker-start="url(#arr)"' if both else '')+'/>')

    def circ(x,y,r,color,stroke='white'):
        p.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="{stroke}" stroke-width="1.5"/>')

    def wave(x,y,w=90,h=36,variant=0):
        box(x-5,y-5,w+10,h+12,'#f0f7fd','#abc6de',5)
        coords=[]
        for i in range(100):
            u=i/99
            v=sum(a*math.exp(-((u-c-variant*.008)/s)**2) for c,s,a in [(.15,.023,.4),(.37,.035,.94),(.61,.022,.55),(.82,.025,.75)])
            coords.append(f'{x+w*u:.1f},{y+h-h*v:.1f}')
        p.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{blue}" stroke-width="2"/>')

    def mol(x,y,scale=1):
        pts=[(x+23*scale*math.cos(i*math.pi/3),y+23*scale*math.sin(i*math.pi/3)) for i in range(6)]
        for i in range(6):
            a,b=pts[i],pts[(i+1)%6]
            line(f'M{a[0]} {a[1]}L{b[0]} {b[1]}',orange,arrow=False)
        line(f'M{x+23*scale} {y}L{x+42*scale} {y-17*scale}',orange,arrow=False)
        for i,(a,b) in enumerate(pts):circ(a,b,5*scale,orange if i!=2 else blue)
        circ(x+42*scale,y-17*scale,7*scale,'#d76665')

    def vector(x,y,color,n=7,w=11,h=27):
        for i in range(n):
            box(x+i*(w+2),y,w,h,color,'white',2)
            p.append(f'<rect x="{x+i*(w+2)}" y="{y}" width="{w}" height="{h}" fill="white" opacity="{.12+(i*3%7)*.09}"/>')

    def matrix(x,y,n=5,cell=19,kind='identity'):
        for i in range(n):
            for j in range(n):
                if kind=='identity':v=1 if i==j or (i in [1,2] and j in [1,2]) else .08
                elif kind=='views':v=1 if abs(i-j)==n//2 else .08
                else:v=max(.06,1-abs(i-j)*.29-((i+j)%3)*.07)
                if kind in ['masked','views'] and i==j:
                    fill='#f0f0f0'
                else:fill=f'rgb({int(246-v*176)},{int(248-v*111)},{int(251-v*68)})'
                box(x+j*cell,y+i*cell,cell-1,cell-1,fill,'white',1)
                if kind in ['masked','views'] and i==j:
                    line(f'M{x+j*cell+5} {y+i*cell+5}l{cell-11} {cell-11}', '#9aa1a7',arrow=False)

    def panel(x,y,w,h,title,fill):
        box(x,y,w,h,fill,'#aeb8c1',13,True)
        text(x+16,y+28,title,'h')

    # A: two actual view paths enter one shared spectrum backbone.
    text(22,30,'(a) Paired theoretical data → shared retrieval space','h')
    text(1578,29,'spectrum_graph_v2  |  B = 32  |  τ = 0.1','s','end')
    box(18,44,1564,263,'#fafbfc','#c6cdd3',12)
    wave(40,80,104,46)
    text(92,153,'Raman x','b','middle')
    text(92,175,'3501 points','s','middle')
    text(92,196,'500–4000 cm⁻¹','tiny','middle')
    line('M157 105H185V77H210')
    line('M185 105V156H210')
    text(183,225,'Augment + z-score','s','middle')
    wave(221,60,76,30,0); text(309,82,'x₁','b')
    wave(221,139,76,30,1); text(309,161,'x₂','b')
    line('M334 76H371');line('M334 155H371')
    box(376,59,275,126,'#e5f0fa',blue)
    text(513,85,'Spectrum Transformer','b','middle')
    text(513,111,'16-point patches / stride 8','s','middle')
    text(513,135,'436 tokens → 3 layers × 4 heads','s','middle')
    text(513,161,'Mean pool → h₁, h₂ ∈ ℝ¹²⁸','s','middle')
    text(514,205,'Shared weights for both views','s','middle',blue)
    line('M651 93H688');line('M651 152H688')
    box(693,65,122,112,'white',blue)
    text(754,95,'Projection','b','middle');text(754,122,'+ L2 norm','s','middle');text(754,150,'128 → 64','s','middle')
    line('M815 92H850');line('M815 153H850')
    vector(856,75,blue);text(963,96,'s₁','b')
    vector(856,137,blue);text(963,158,'s₂','b')
    text(911,192,'Unit vectors ∈ ℝ⁶⁴','s','middle')
    mol(79,262,.85);text(142,251,'SMILES','b');text(142,277,'→ graph','s')
    line('M246 258H371')
    text(299,244,'7 atom / 3 bond','tiny','middle')
    text(299,281,'feature types','tiny','middle')
    box(376,219,275,71,'#fbefdf',orange)
    text(513,245,'Edge-conditioned GIN × 2','b','middle')
    text(513,272,'0/1/2-hop mean + sum → hɢ¹²⁸','s','middle')
    line('M651 255H688')
    box(693,225,122,58,'white',orange)
    text(754,248,'Projection','s','middle');text(754,270,'+ L2 norm','s','middle')
    line('M815 254H850');vector(856,241,orange);text(963,263,'g','b')
    # Shared latent space: two views cluster around the same structure.
    box(1004,59,226,231,'white','#b7c0c8',8)
    text(1117,84,'Shared latent space','b','middle')
    for xx,yy,c in [(1042,111,blue),(1070,126,blue),(1061,147,orange),(1170,120,blue),(1190,141,orange),(1169,152,blue),(1104,211,orange),(1082,225,blue),(1117,234,blue)]:
        if c==orange:box(xx-7,yy-7,14,14,c,'white',2)
        else:circ(xx,yy,7,c)
    for d in ['M1042 111L1061 147L1070 126','M1170 120L1190 141L1169 152','M1082 225L1104 211L1117 234']:line(d,'#85939f',True,arrow=False)
    text(1117,274,'● spectra     ■ structures','s','middle')
    box(1250,59,312,231,'#f3f7ed','#9caf89',8)
    text(1406,84,'Structure-derived supervision','b','middle')
    mol(1290,119,.6)
    line('M1326 116H1360',green)
    text(1372,113,'Canonical identity','s');text(1372,134,'→ positive-pair mask','s')
    vector(1270,160,green,6,9,19)
    text(1372,170,'Morgan bits → Tanimoto','s')
    text(1372,190,'r = 2 · 2048 bits','tiny')
    text(1270,222,'Descriptors + element presence','s')
    text(1270,247,'Identity → (b, c); Morgan → (d)','s')
    text(1270,274,'Chemical attributes → (e)','s')

    # B: bidirectional matching with multi-positive block.
    panel(18,321,774,261,'(b) Cross-modal alignment', '#f5f9fd')
    text(775,350,'Lcross','b','end',blue)
    wave(44,385,72,29);mol(80,465,.6)
    line('M131 401H173',blue);line('M131 465H173',orange)
    vector(182,387,blue,5,9,24);vector(182,451,orange,5,9,24)
    text(209,434,'Sᵥ','s','middle');text(209,496,'G','s','middle')
    line('M246 401H267V432H294');line('M246 465H267V432')
    matrix(303,381,5,20)
    text(353,373,'g₁ … gB','tiny','middle');text(289,432,'sᵢ','tiny','end')
    line('M309 491H397',blue,both=True);text(352,514,'S → G / G → S','tiny','middle')
    text(430,388,'Same molecule = positive','b',color=blue)
    text(430,413,'Includes repeated canonical identities','s')
    box(433,431,17,17,'#4689b7','#4689b7',2);text(460,445,'Positive pairs: increase probability','s')
    box(433,459,17,17,'#e8edf3','#b8c7d5',2);text(460,473,'Other pairs: contrastive negatives','s')
    text(430,510,'Both views × both directions → average','s')
    text(39,550,'Lcross = ¼ Σᵥ [ C(SᵥGᵀ / τ) + C(GSᵥᵀ / τ) ]','f')
    text(39,573,'C(A) = −meanᵢ meanⱼ∈P(i) log softmax(Aᵢ)ⱼ;  v ∈ {1, 2}','s')

    # C: explicit pull/push geometry and diagonal masking.
    panel(808,321,774,261,'(c) Spectral augmentation consistency','#f8f5fb')
    text(1565,350,'Laug','b','end',purple)
    wave(838,382,66,29);wave(838,458,66,29,1)
    text(871,441,'same molecule','tiny','middle')
    line('M923 397H969',blue);line('M923 473H969',blue)
    circ(1000,408,11,blue);circ(1074,462,11,blue)
    line('M1011 416L1061 452',green,both=True)
    text(1069,406,'pull together','s',color=green)
    circ(1125,488,9,purple);circ(1180,433,9,purple);circ(1046,493,9,purple)
    line('M1013 410L1165 431',purple,True);line('M1010 416L1113 480',purple,True)
    text(1116,514,'push other identities away','tiny','middle',purple)
    matrix(1269,380,6,18,'views')
    text(1400,400,'Z = [S₁; S₂]','s')
    text(1400,427,'2B × 2B scores','s')
    text(1400,454,'× exclude self','s')
    text(1400,482,'P(i): same identity','s')
    text(1270,514,'Positive: another view of the same molecule','tiny')
    text(829,550,'Laug = −meanᵢ meanₚ∈P(i) log [ exp(zᵢᵀzₚ / τ) / Σₖ≠ᵢ exp(zᵢᵀzₖ / τ) ]','f')
    text(829,573,'Preserve molecular identity under spectral perturbations.','s')

    # D: fixed molecular similarities are transferred to three latent relations.
    panel(18,596,774,268,'(d) Preserve graded structural similarity','#f5faf5')
    text(775,625,'Lstruct','b','end',green)
    mol(61,673,.58);mol(114,694,.47)
    text(88,734,'Structures','s','middle')
    line('M147 686H178',green)
    vector(188,660,green,6,8,18);vector(188,691,green,6,8,18)
    text(221,734,'Morgan bits','s','middle')
    line('M253 686H283',green)
    matrix(293,649,5,15,'soft')
    text(330,742,'Tanimoto T','s','middle')
    line('M373 686H417',green)
    text(395,670,'softmax','tiny','middle')
    # Tiny probability distributions communicate soft supervision.
    for i,h in enumerate([38,28,14,7,4]):box(430+i*12,710-h,9,h,green,'white',1)
    text(460,741,'Target p','s','middle')
    line('M501 687H546',green)
    text(525,673,'KL','b','middle',green)
    for xx,label,kind in [(555,'S̄Gᵀ','soft'),(634,'S̄S̄ᵀ','masked'),(711,'GGᵀ','masked')]:
        matrix(xx,654,5,12,kind);text(xx+30,738,label,'s','middle')
    text(648,758,'Three learned relation matrices','tiny','middle')
    text(39,792,'Lstruct = D(S̄Gᵀ) + ½ D(S̄S̄ᵀ) + ½ D(GGᵀ)','f')
    text(39,818,'D(A) = meanᵢ KL[softmax(Tᵢ / 0.1) ∥ softmax(Aᵢ / 0.1)]','s')
    text(39,846,'S̄ = norm[(S₁ + S₂)/2]; diagonal masked for S̄S̄ᵀ and GGᵀ only.','s')

    # E: chemical semantics are predicted from spectrum backbone, not latent graph.
    panel(808,596,774,268,'(e) Predict chemical attributes from spectra','#fcf8f1')
    text(1565,625,'Laux','b','end',orange)
    vector(831,677,blue,7,9,30)
    text(871,733,'(h₁ + h₂) / 2','s','middle')
    line('M914 692H938V663H959');line('M938 692V743H959')
    box(966,642,132,43,'#e8f1fa',blue,5);text(1032,668,'Descriptor head','s','middle')
    box(966,723,132,43,'#e8f1fa',blue,5);text(1032,749,'Element head','s','middle')
    line('M1098 663H1127');line('M1098 744H1127')
    for i,h in enumerate([15,30,22,35,11,27]):box(1136+i*13,682-h,9,h,blue,'white',1)
    text(1174,704,'15 predictions d̂','tiny','middle')
    for i,lab in enumerate(['C','N','O','F','H']):
        circ(1142+i*21,738,9,blue if i in [0,2,4] else '#d9e1e7')
        text(1142+i*21,762,lab,'tiny','middle')
    line('M1221 664H1283',orange,both=True);text(1250,650,'SmoothL1','tiny','middle')
    line('M1237 738H1283',orange,both=True);text(1262,714,'BCE logits','tiny','middle')
    box(1292,643,269,123,'#fff1df',orange,6)
    text(1426,668,'Structure-derived targets','b','middle')
    text(1306,696,'d*: 15 standardized descriptors','s')
    text(1306,723,'e*: C / N / O / F / H presence','s')
    text(1306,750,'Training-set normalization','tiny')
    text(829,804,'Laux = SmoothL1(d̂, d*) + BCEWithLogits(ê, e*)','f')
    text(829,831,'Counts, mass, rings, bond types and charge constrain spectral features.','s')
    text(829,851,'No direct fingerprint prediction head in this run (λfp = 0).','tiny')

    box(18,877,1564,49,'#eef2f6','#aebcc9',8)
    text(800,910,'Ltotal = Lcross + 0.25 Laug + 0.5 Lstruct + 0.1 Laux','h','middle')
    text(28,954,'INFERENCE','b')
    text(159,954,'Raman query → spectrum encoder → s⁶⁴','s')
    line('M487 948H533')
    text(548,954,'Cosine scores → 10,000 cached candidate graphs','s')
    line('M963 948H1010')
    text(1027,954,'Ranked Top-K molecular candidates','b')
    text(28,985,'Training: 9,500 theoretical spectra  ·  Validation: 500 theoretical  ·  Test: 200 held-out experimental spectra','s')
    text(1578,985,'Icons, geometry and matrix values are schematic.','tiny','end')
    p.append('</svg>')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(p),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    draw(parser.parse_args().output)
