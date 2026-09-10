"""功能概述：生成论文用 Raman 双塔模型与训练目标 SVG 插图。
主要职责：text/box/arrow 构造可编辑 SVG 元素，spectrum/molecule 绘制示意输入；main
按当前 spectrum_graph_v2 的 128/64 维配置、损失权重和检索协议组装单页矢量图。
代码库关系：图示对应 multimodal_retriever、structure_losses 与 run_closed_raman.sh；
光谱和分子图仅为示意，不代表实验样本。公式中的 CE 使用平均正样本 log probability。
输入输出与副作用：只写显式 --output 文件；无模型加载、远程访问或数据读取。
"""
import argparse
from html import escape
from pathlib import Path
import math
import re


def main(output):
    parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1680" height="1260" viewBox="0 0 1680 1260">
<title>Raman spectrum–graph retrieval: architecture and training objectives</title>
<desc>Two augmented Raman views and a molecular graph are encoded into a shared 64 dimensional space. Training combines symmetric cross-modal alignment, augmentation consistency, Morgan structure relations, and auxiliary supervision. Inference ranks 10000 molecular candidates from a spectrum-only query.</desc>
<!-- 可编辑矢量模型插图；波形与分子图为示意，具体层数、维度与损失权重对应本次训练。 -->
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="#536275"/></marker>
</defs>
<style>
 text { font-family: Arial, Helvetica, sans-serif; fill:#192b3d; }
 .title {font-size:32px;font-weight:700;}
 .section {font-size:25px;font-weight:700;}
 .head {font-size:23px;font-weight:700;}
 .body {font-size:21px;}
 .small {font-size:18px;fill:#506176;}
 .formula {font-family: 'DejaVu Sans',Arial,sans-serif;font-size:23px;}
 .note {font-size:17px;fill:#506176;}
</style>
<rect width="1680" height="1260" fill="white"/>''']

    def text(x, y, value, cls='body', anchor='start', color=None):
        fill = f' style="fill:{color}"' if color else ''
        value = re.sub(r'\bL(cross|aug|struct|aux|total)\b',
                       r'L<tspan baseline-shift="sub" font-size="70%">\1</tspan>', escape(value))
        parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"{fill}>{value}</text>')

    def box(x, y, w, h, fill='#f7f9fc', stroke='#d6dfe8', radius=10):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')

    def arrow(path, dashed=False):
        dash = ' stroke-dasharray="6 5"' if dashed else ''
        parts.append(f'<path d="{path}" fill="none" stroke="#536275" stroke-width="2" marker-end="url(#arrow)"{dash}/>')

    def spectrum(x, y, width, height):
        coords=[]
        for i in range(250):
            u=i/249
            v=sum(a*math.exp(-((u-c)/s)**2) for c,s,a in [(0.16,.016,.45),(.31,.026,.9),(.47,.014,.62),(.60,.028,.34),(.82,.022,.7)])
            coords.append(f'{x+width*u:.2f},{y+height-height*v:.2f}')
        parts.append(f'<path d="M{x},{y+height+2} H{x+width}" stroke="#b6c9dc" fill="none"/>')
        parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="#276b9b" stroke-width="2.4"/>')

    def molecule(x,y):
        points=[(x+35*math.cos(math.pi*i/3),y+35*math.sin(math.pi*i/3)) for i in range(6)]
        parts.append('<polygon points="'+' '.join(f'{a:.1f},{b:.1f}' for a,b in points)+'" fill="none" stroke="#a96b35" stroke-width="2.5"/>')
        for a,b in points:
            parts.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="#a96b35"/>')
        parts.append(f'<path d="M{x+35},{y} h32" stroke="#a96b35" stroke-width="2.5"/>')
        text(x+72,y+7,'O','body',color='#b35449')

    text(48,52,'Raman spectrum–graph retrieval','title')
    text(1632,50,'MODEL & TRAINING OBJECTIVES','small','end')
    parts.append('<path d="M48 74 H1632" stroke="#d6dfe8"/>')
    text(48,114,'a','section')
    text(82,114,'Dual-encoder architecture','section')
    text(1632,114,'Training: paired theoretical Raman spectra and molecular structures','small','end')

    box(48,144,1584,198,'#f3f8fc','#d7e5ef')
    box(48,356,1584,148,'#fcf8f3','#eadfce')
    # Spectrum branch: two views share one encoder.
    box(68,165,266,154,'white','#aac9df')
    text(201,194,'Raman spectrum','head','middle')
    spectrum(91,204,215,52)
    text(201,284,'500–4000 cm⁻¹ · 3501 points','note','middle')
    text(201,307,'Augment → standardize → x₁, x₂','note','middle')
    arrow('M334 237 H399')
    box(404,165,380,154,'white','#aac9df')
    text(594,194,'Shared spectrum encoder','head','middle')
    text(594,224,'Patch 16 / stride 8 → 436 tokens','body','middle')
    text(594,253,'Transformer × 3 · 4 heads · d = 128','body','middle')
    text(594,283,'Mean pooling → h₁, h₂ ∈ ℝ¹²⁸','body','middle')
    text(594,308,'Projection head + L2 normalization','note','middle')
    arrow('M784 240 H875')
    box(880,190,250,101,'#e0eef9','#91b8d5')
    text(1005,222,'Spectral embeddings','head','middle')
    text(1005,255,'s₁, s₂ ∈ ℝ⁶⁴','formula','middle')
    text(1005,280,'Unit norm · shared latent space','note','middle')
    box(1210,165,400,154,'white','#aac9df')
    text(1410,194,'Auxiliary prediction heads','head','middle')
    text(1410,224,'Input: (h₁ + h₂) / 2','body','middle')
    text(1410,255,'15 standardized descriptors','body','middle')
    text(1410,285,'5 element-presence logits','body','middle')
    text(1410,307,'C / N / O / F / H','note','middle')
    arrow('M785 173 V153 H1410 V163',True)

    box(68,373,266,113,'white','#d9b993')
    molecule(125,425)
    text(240,410,'SMILES','head','middle')
    text(240,439,'↓','body','middle')
    text(240,466,'Molecular graph','small','middle')
    arrow('M334 430 H399')
    box(404,373,380,113,'white','#d9b993')
    text(594,403,'Graph encoder','head','middle')
    text(594,433,'Edge-conditioned GIN · 2 layers','body','middle')
    text(594,461,'0/1/2-hop mean + sum → hɢ ∈ ℝ¹²⁸','small','middle')
    arrow('M784 430 H875')
    box(880,380,250,101,'#f6e9da','#d9b993')
    text(1005,413,'Graph embedding','head','middle')
    text(1005,444,'g ∈ ℝ⁶⁴','formula','middle')
    text(1005,468,'Projection + L2 normalization','note','middle')
    box(1210,373,400,113,'white','#d9b993')
    text(1410,403,'Structure-derived targets','head','middle')
    text(1410,433,'Morgan bits · descriptors · identity','body','middle')
    text(1410,463,'Fixed labels for training losses','small','middle')
    arrow('M201 486 V497 H1410 V487',True)
    # Embedding alignment shown between latent boxes, not an inference input.
    parts.append('<path d="M1005 295 V375" stroke="#536275" stroke-width="2" stroke-dasharray="5 5" marker-start="url(#arrow)" marker-end="url(#arrow)"/>')
    text(1040,339,'Alignment','small')

    box(48,524,1584,88,'#f6f8fb','#d6dfe8')
    text(70,555,'INFERENCE','small')
    text(70,587,'Spectrum-only query','head')
    arrow('M321 571 H372')
    text(395,579,'s ∈ ℝ⁶⁴','formula')
    arrow('M520 571 H568')
    text(588,561,'Cosine scores: sᵀgⱼ','head')
    text(588,588,'10,000 pre-encoded candidate graphs','small')
    arrow('M996 571 H1060')
    text(1083,579,'Ranked Top-K candidate SMILES','head')
    text(48,658,'b','section')
    text(82,658,'Training objectives','section')
    text(1632,658,'Batch B = 32 · contrastive temperature τ = 0.1','small','end')

    # Each loss card includes its actual code definition.
    cards=[(48,681,776,183),(846,681,786,183),(48,885,776,197),(846,885,786,197)]
    for x,y,w,h in cards:box(x,y,w,h,'#fff','#d6dfe8')
    text(70,715,'Cross-modal alignment','head')
    text(802,715,'weight 1.0','small','end')
    text(70,753,'Lcross = ¼ Σᵥ [ C(SᵥGᵀ / τ) + C(GSᵥᵀ / τ) ]','formula')
    text(70,787,'C(A) = −meanᵢ meanⱼ∈P(i) log softmax(Aᵢ)ⱼ','body')
    text(70,817,'v ∈ {1, 2}; P(i): columns with the same canonical identity.','small')
    text(70,845,'Align each spectrum with its paired molecular structure.','small')

    text(868,715,'Spectral augmentation consistency','head')
    text(1610,715,'weight 0.25','small','end')
    text(868,753,'Laug = −meanᵢ meanₚ∈P(i) log pᵢₚ','formula')
    text(868,787,'pᵢⱼ = exp(zᵢᵀzⱼ / τ) / Σₖ≠ᵢ exp(zᵢᵀzₖ / τ)','body')
    text(868,817,'Z = [S₁; S₂]; exclude self; same-identity views are positives.','small')
    text(868,845,'Keep molecular identity stable under spectral perturbations.','small')

    text(70,919,'Molecular structure relations','head')
    text(802,919,'weight 0.5','small','end')
    text(70,957,'Lstruct = D(S̄Gᵀ) + ½D(S̄S̄ᵀ) + ½D(GGᵀ)','formula')
    text(70,991,'D(A) = meanᵢ KL[ softmax(Tᵢ / 0.1) ∥ softmax(Aᵢ / τ) ]','small')
    text(70,1022,'S̄: normalized mean of s₁ and s₂; T: Morgan Tanimoto.','small')
    text(70,1049,'Mask the diagonal for S̄S̄ᵀ and GGᵀ only.','small')
    text(70,1073,'Morgan radius 2 · 2048 bits · no chirality encoding','note')

    text(868,919,'Auxiliary chemical supervision','head')
    text(1610,919,'weight 0.1','small','end')
    text(868,957,'Laux = SmoothL1(d̂, d*) + BCEWithLogits(ê, e*)','formula')
    text(868,991,'d*: 15 descriptors, standardized using the training split.','small')
    text(868,1022,'e*: presence of C / N / O / F / H.','small')
    text(868,1049,'Predictions use the mean of the two spectral backbone features.','small')
    text(868,1073,'Direct fingerprint prediction is disabled (λfp = 0).','note')

    box(48,1104,1584,90,'#edf3f8','#b5c8d9')
    text(840,1142,'Ltotal = Lcross + 0.25 Laug + 0.5 Lstruct + 0.1 Laux','title','middle')
    text(840,1176,'9,500 theoretical training spectra · 500 theoretical validation spectra · seed 0','body','middle')
    text(48,1223,'Checkpoint selection: validation Top-1 Tanimoto with same-identity candidates excluded.','small')
    text(48,1248,'The 200 experimental test spectra are held out from training and model selection; input drawings are schematic.','note')
    parts.append('</svg>')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(parts),encoding='utf-8')
    print(output.resolve())


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
