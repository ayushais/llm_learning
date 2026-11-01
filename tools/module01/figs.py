import json, re
from html import escape as E
import os
HERE=os.path.dirname(os.path.abspath(__file__))
S=HERE+"/"
D=json.load(open(S+"data.json"))
W=480
def vis(s): return s.replace(" ","·") if s.startswith(" ") else s
NUM={1:1,3:3,4:4,"proj":5,5:6,6:7,7:8,"mlp":9,8:10,9:11,10:12,11:13,12:14,13:15}
def fig(n,svg,cap,label):
    n=NUM[n]
    return f'<figure class="fig" id="fig{n}">{svg}<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>\n'
def svg(h,body,label,w=W):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{E(label)}">{body}</svg>'
def t(x,y,s,cls="",anchor="start",size=12,extra=""):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" class="{cls}" {extra}>{E(s)}</text>'
F={}

# 1 next-token distribution
rows=D["milk"]; mx=max(v for _,v in rows); y0=10; body=""
for i,(tk,v) in enumerate(rows):
    y=y0+i*24; w=v/mx*270
    body+=t(100,y+14,repr(tk)[1:-1] if False else tk.replace(" ","·"),"mono","end")
    body+=f'<rect class="bar" x="106" y="{y+3}" width="{w:.1f}" height="14" rx="3"><title>{E(tk)!r}: {v:.3f}</title></rect>'
    body+=t(112+w,y+14,f"{v*100:.1f}%","mut")
yy=y0+10*24+14
body+=t(100,yy+8,"…","mut","end")+t(106,yy+8,f"and ~50,000 other tokens share the remaining {D['milk_rest']*100:.0f}%","mut")
F[1]=fig(1,svg(yy+22,body,"Top ten next-token probabilities after 'I put the milk in the'"),
 "GPT-2's actual next-token distribution after <code>I put the milk in the</code>. The model doesn't output one answer; it outputs a probability for every one of 50,257 tokens. Plausible fridge-like words dominate, but no single word wins. (<code>·</code> marks a leading space, which belongs to the token.)","")

# 3 tokenization chips
x=8;y=8;body="";H=0
for tk,i in D["tok"]:
    disp=tk.replace(" ","·"); w=max(34,8.2*len(disp)+16)
    if x+w>W-8: x=8;y+=62
    body+=f'<rect class="chip" x="{x}" y="{y}" width="{w:.0f}" height="28" rx="6"/>'+t(x+w/2,y+19,disp,"mono","middle",12)
    body+=t(x+w/2,y+46,str(i),"mut","middle",11)
    x+=w+6
H=y+62
F[3]=fig(3,svg(H,body,"The sentence split into GPT-2 tokens and their integer ids"),
 "How GPT-2 sees <code>The cat sat on the mat. Unbelievably, tokenization is weird!</code>: boxes are tokens, numbers are vocabulary ids. Common words are single tokens (with their leading space); rare words like <i>Unbelievably</i> split into pieces. The model only ever sees the numbers.","")

# 4 residual stream
b=f'''<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--acc)"/></marker></defs>
<line x1="110" y1="30" x2="110" y2="300" stroke="var(--fg)" stroke-width="3"/>
{t(110,18,"x₀ = embedding","","middle")}
<path d="M110 70H210" class="ln" marker-end="url(#ah)"/>
<rect class="box" x="212" y="52" width="150" height="36" rx="8"/>{t(287,75,"norm → attention","","middle")}
<path d="M362 70H400V120H118" class="ln" marker-end="url(#ah)"/>
<circle cx="110" cy="120" r="9" class="box"/>{t(110,124,"+","","middle")}
{t(220,110,"output added back to the stream","mut","start",11)}
{t(122,150,"x₁ = x₀ + Attn(Norm(x₀))","mut")}
<path d="M110 175H210" class="ln" marker-end="url(#ah)"/>
<rect class="box" x="212" y="157" width="150" height="36" rx="8"/>{t(287,180,"norm → MLP","","middle")}
<path d="M362 175H400V225H118" class="ln" marker-end="url(#ah)"/>
<circle cx="110" cy="225" r="9" class="box"/>{t(110,229,"+","","middle")}
{t(122,255,"x₂ = x₁ + MLP(Norm(x₁))","mut")}
{t(110,320,"repeat for every layer, then unembed","","middle")}
<path d="M110 300V306" stroke="var(--fg)" stroke-width="3"/>'''
F[4]=fig(4,svg(330,b,"Residual stream: each sublayer reads the stream and adds its output back"),
 "The residual stream. The thick vertical line is the vector for one token position as it moves up through the layers. Each sublayer reads a normalised copy, computes something, and <i>adds</i> the result back; nothing is overwritten. This is why depth can be stacked easily and why researchers can ask what each layer contributed (Elhage et al. 2021).","")

# 5 attention flow
b=f'''<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--acc)"/></marker></defs>
{t(10,22,"current token","mut")}{t(10,38,"(“to”)","mut")}
<rect class="box" x="10" y="48" width="70" height="30" rx="6"/>{t(45,68,"query q","","middle")}
{t(150,22,"earlier tokens","mut")}
'''
names=["Mary","and","John","…"]
for i,n in enumerate(names):
    y=48+i*40
    b+=f'<rect class="box" x="150" y="{y}" width="70" height="30" rx="6"/>'+t(185,y+19,f"key k({n})" if n!="…" else "keys …","","middle",11)
    b+=f'<rect class="box" x="360" y="{y}" width="80" height="30" rx="6"/>'+t(400,y+19,f"value v({n})" if n!="…" else "values …","","middle",11)
    b+=f'<path d="M80 63L150 {y+15}" class="ln" marker-end="url(#ah)" style="opacity:.5"/>'
    b+=t(262,y+19,"w","","middle",12)+f'<rect x="240" y="{y+4}" width="44" height="22" rx="4" fill="none" stroke="var(--bd)"/>'
    b+=f'<path d="M220 {y+15}H240M284 {y+15}H360" class="ln" marker-end="url(#ah)"/>' if False else f'<path d="M220 {y+15}H240" class="ln"/><path d="M284 {y+15}H358" class="ln" marker-end="url(#ah)" style="opacity:.6"/>'
b+=t(262,40,"weight","mut","middle",11)
b+=f'<rect class="box" x="150" y="225" width="290" height="34" rx="8"/>'+t(295,247,"output = Σ weight × value   (a blend of values)","","middle")
b+=t(150,278,"weights = softmax( q·k / √d ) over the allowed","mut","start",11)+t(150,292,"(earlier) tokens; they sum to 1","mut","start",11)
F[5]=fig(5,svg(304,b,"Attention: a query is compared with keys, the scores become weights, and the weights blend the values"),
 "One attention step for one token. The current token’s query is scored against every earlier token’s key (dot product, scaled by √d). Softmax turns the scores into weights that sum to 1, and the output is the weighted sum of the tokens’ values. Keys say “what do I contain”, queries say “what am I looking for”, values say “what do I pass on”.","")

# 6 real attention bars last row
labs=D["labs"]; row=D["attn"][-1]; rows=list(zip(labs,row)); body=""; 
for i,(tk,v) in enumerate(rows):
    y=8+i*20; hi=(tk==" Mary")
    body+=t(86,y+14,tk.replace(" ","·"),"mono","end")
    if v>=0.005: body+=f'<rect class="{"bar2" if hi else "bar"}" x="92" y="{y+3}" width="{v*300:.1f}" height="12" rx="3"><title>{E(tk)}: {v:.2f}</title></rect>'
    if v>=0.05: body+=t(98+v*300,y+14,f"{v:.2f}","mut")
    if hi: body+=t(98+v*300+34,y+14,"← the answer","","start",12)
F[6]=fig(6,svg(8+len(rows)*20+6,body,"Attention weights of the final token in head 9.6"),
 "Real attention weights from GPT-2, layer 9, head 6, for the last token of <code>When Mary and John went to the store, John gave a drink to</code>. This head puts nearly half its weight on <b>Mary</b>, the person who should receive the drink, and most of the rest on the first token (a common “parking spot” when nothing is relevant). This head was identified as a name-mover in the indirect-object circuit studied by Wang et al. (2022).","")

# 7 heatmap
A=D["attn"]; n=len(labs); cs=22; ox=76; oy=70; body=""
for j,tk in enumerate(labs):
    x=ox+j*cs+cs/2
    body+=t(x,oy-6,tk.replace(" ","·"),"mono","start",10,f'transform="rotate(-55 {x} {oy-6})"')
    body+=t(ox-4,oy+j*cs+cs/2+4,tk.replace(" ","·"),"mono","end",10)
for i in range(n):
    for j in range(n):
        if j<=i:
            v=A[i][j]; body+=f'<rect x="{ox+j*cs}" y="{oy+i*cs}" width="{cs-1}" height="{cs-1}" rx="2" class="cell" style="opacity:{0.06+0.94*min(1,v/0.6):.2f}"><title>{E(labs[i])} → {E(labs[j])}: {v:.2f}</title></rect>'
        else:
            body+=f'<rect x="{ox+j*cs}" y="{oy+i*cs}" width="{cs-1}" height="{cs-1}" rx="2" class="mask"/>'
h=oy+n*cs+10
F[7]=fig(7,svg(h,body,"Attention heatmap for layer 9 head 6 with causal mask"),
 "The same head for every token, as a grid. Row = the token doing the looking, column = the token being looked at, darker = more weight. The shaded upper triangle is the <b>causal mask</b>: a token can never look at later tokens. That restriction is what lets one forward pass train on every position of a sequence simultaneously. Look at the last row (Figure 7).","")

# 8 params
P=[("MLP",56.7),("token embedding",38.6),("attention",28.3),("position embedding",0.8)]; body=""
for i,(k,v) in enumerate(P):
    y=8+i*30; w=v/56.7*260
    body+=t(130,y+17,k,"","end")+f'<rect class="bar" x="138" y="{y+3}" width="{w:.1f}" height="20" rx="4"><title>{k}: {v}M</title></rect>'+t(144+w,y+18,f"{v}M","mut")
body+=t(138,8+4*30+14,"(layer norms: < 0.1M; output layer shares the token embedding)","mut","start",11)
F[8]=fig(8,svg(8+4*30+26,body,"Parameter count by component for GPT-2 small"),
 "Where GPT-2 small’s 124.4M parameters live (measured in the notebook). Embeddings are a big slice for a small model; in large models the transformer blocks dominate.","")


# proj: where q, k, v come from
rows=[("W_Q","q","what am I looking for?"),("W_K","k","how do others find me?"),("W_V","v","what do I hand over if picked?")]
b=f'''<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--acc)"/></marker></defs>
<rect class="box" x="10" y="30" width="34" height="150" rx="6"/>{t(27,24,"x","","middle",13)}
{t(27,198,"768","mut","middle",11)}{t(27,212,"numbers","mut","middle",11)}
'''
for i,(w,o,r) in enumerate(rows):
    y=30+i*55
    b+=f'<path d="M44 105L100 {y+18}" class="ln" style="opacity:.5"/>'
    b+=f'<rect class="box" x="100" y="{y}" width="84" height="36" rx="8"/>'+t(142,y+23,"× "+w,"","middle",13)
    b+=f'<path d="M184 {y+18}H218" class="ln" marker-end="url(#ah)"/>'
    sc={"q":"--s1","k":"--s2","v":"--s3"}[o]
    b+=f'<rect class="box" x="222" y="{y+6}" width="22" height="24" rx="4" style="stroke:var({sc});stroke-width:2.5"/>'+t(233,y+23,o,"","middle",13,'style="font-weight:600"')
    b+=t(256,y+16,r,"","start",12)+t(256,y+30,"(64 numbers)","mut","start",10.5)
b+=t(110,208,"learned in training; same matrices at every position;","mut","start",11)+t(110,222,"recomputed from the current text on every pass","mut","start",11)
F["proj"]=fig("proj",svg(232,b,"One position's vector is multiplied by three learned matrices to give a query, a key and a value"),
 "Where query, key and value come from, for one head of GPT-2 small. The position’s current vector <i>x</i> is multiplied by three learned matrices. The results are three short vectors with three different jobs. Every position does this, and each of the 12 heads has its own three matrices.","")

# mlp
b=f'''<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--acc)"/></marker></defs>
<rect class="box" x="12" y="62" width="34" height="46" rx="5"/>{t(29,52,"x","","middle",12)}{t(29,124,"768","mut","middle",11)}
<path d="M46 85H86" class="ln" marker-end="url(#ah)"/>{t(66,76,"W_in","mut","middle",10.5)}
<rect class="box" x="90" y="8" width="60" height="154" rx="5"/>{t(120,86,"3072","","middle",12)}{t(120,178,"hidden units","mut","middle",11)}
<path d="M150 85H190" class="ln" marker-end="url(#ah)"/>{t(170,76,"GELU","mut","middle",10.5)}
<rect class="box" x="194" y="8" width="60" height="154" rx="5" style="fill:var(--code)"/>{t(224,86,"active","","middle",11)}{t(224,100,"units","","middle",11)}
<path d="M254 85H294" class="ln" marker-end="url(#ah)"/>{t(274,76,"W_out","mut","middle",10.5)}
<rect class="box" x="298" y="62" width="34" height="46" rx="5"/>{t(315,52,"Δx","","middle",12)}{t(315,124,"768","mut","middle",11)}
{t(350,76,"added back to","mut","start",11.5)}{t(350,91,"the residual stream","mut","start",11.5)}{t(350,112,"same weights at every","mut","start",11)}{t(350,126,"position; no mixing","mut","start",11)}{t(350,140,"between positions","mut","start",11)}
<line x1="12" y1="196" x2="468" y2="196" class="grid"/>
{t(12,216,"The memory view (illustrative example, not measured):","","start",12,'font-weight="600"')}
<rect class="box" x="12" y="226" width="200" height="50" rx="8"/>{t(112,246,"unit detects: context looks like","","middle",11.5)}{t(112,262,"“…Eiffel Tower is located in”","mono","middle",11)}
<path d="M212 251H262" class="ln" marker-end="url(#ah)"/>
<rect class="box" x="266" y="226" width="200" height="50" rx="8"/>{t(366,246,"unit writes: push the prediction","","middle",11.5)}{t(366,262,"toward “Paris”","mono","middle",11)}'''
F["mlp"]=fig("mlp",svg(290,b,"MLP: expand to 3072 units, apply an activation, project back to 768; and the key-value memory interpretation"),
 "Top: the MLP for one position. The vector is expanded to 3072 hidden units, a nonlinearity switches units on or off, and W<sub>out</sub> projects back to 768 numbers that are added to the residual stream. Bottom: the key–value memory reading of Geva et al. (2021): a unit whose input pattern matches the context writes a vector that favours certain next tokens. The example is illustrative.","")

# 9 training stages
stages=[("1  Pretraining","web text, books, code","predict the next token","billions to trillions of tokens; most of the compute","base model"),
        ("2  Supervised fine-tuning","(prompt, ideal answer) pairs","same loss, on the answer tokens","thousands to ~100k examples (InstructGPT: ~13k)","follows instructions"),
        ("3  Preference tuning / RL","human or AI rankings; checkable answers","RLHF (reward model + PPO), DPO, RL with verifiable rewards","far less data than pretraining","assistant")]
body=f'<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--acc)"/></marker></defs>'
for i,(a,d,o,s,r) in enumerate(stages):
    y=8+i*108
    body+=f'<rect class="box" x="8" y="{y}" width="464" height="86" rx="10"/>'
    body+=t(20,y+22,a,"","start",14,'font-weight="600"')+t(20,y+42,"data: "+d,"","start",12)+t(20,y+60,"objective: "+o,"","start",12)+t(20,y+78,"scale: "+s,"mut","start",11)
    body+=t(462,y+22,"→ "+r,"mut","end",12)
    if i<2: body+=f'<path d="M240 {y+86}V{y+108}" class="ln" marker-end="url(#ah)"/>'
F[9]=fig(9,svg(8+3*108,body,"Three training stages: pretraining, supervised fine-tuning, preference tuning"),
 "The three stages of building an assistant. Each stage starts from the weights of the previous one. Stage 1 teaches language and knowledge, stage 2 teaches the format of being helpful, stage 3 shapes which answers are preferred.","")

# 10 surprisal
S_=D["surp"]; mx=max(v for _,v in S_); bw=40; body=""
for i,(tk,v) in enumerate(S_):
    x=24+i*50; h=v/mx*130
    body+=f'<rect class="bar" x="{x}" y="{175-h:.1f}" width="{bw}" height="{h:.1f}" rx="3"><title>{E(tk)}: {v:.2f} nats</title></rect>'
    body+=t(x+bw/2,169-h,f"{v:.1f}","mut","middle",11)+t(x+bw/2,193,tk.replace(" ","·"),"mono","middle",11)
body+=f'<line x1="16" y1="175" x2="472" y2="175" class="axis"/>'+t(8,12,"surprise (−ln P, nats)","mut","start",11)
F[10]=fig(10,svg(205,body,"Per-token surprise for The quick brown fox sentence"),
 "How surprised GPT-2 is by each word of <code>The quick brown fox jumps over the lazy dog.</code> (the first token has no prediction). Adjectives like <i>quick</i>, <i>brown</i>, <i>lazy</i> are unpredictable (~7–9 nats ≈ probability 1 in a thousand or less); after <i>jumps</i> the word <i>over</i> is cheap. Training pushes the average of these bars down across trillions of tokens.","")

# 11 scaling law schematic
body=f'''<line x1="50" y1="20" x2="50" y2="190" class="axis"/><line x1="50" y1="190" x2="460" y2="190" class="axis"/>
<path d="M60 40L450 160" stroke="var(--s1)" stroke-width="2.5" fill="none"/>
{t(255,215,"compute (log scale) →  or parameters, or data","mut","middle",11)}
{t(14,105,"loss (log)","mut","middle",11,'transform="rotate(-90 14 105)"')}
{t(70,150,"straight line on log–log axes = power law","","start",12)}
{t(70,167,"each 10× in compute buys a constant factor in loss","mut","start",11)}'''
for x,y in ((100,52),(200,81),(300,111),(400,141)): body+=f'<circle cx="{x}" cy="{y}" r="4" fill="var(--s1)" stroke="var(--bg)" stroke-width="2"/>'
F[11]=fig(11,svg(228,body,"Schematic of a scaling law: loss falls as a straight line on log-log axes"),
 "<b>Schematic, not measured data.</b> Kaplan et al. (2020) found that test loss falls as a smooth power law in parameters, data and compute over many orders of magnitude, which plots as a straight line on log–log axes. Chinchilla (2022) refined how a fixed compute budget should be split between model size and training tokens.","")

# 12 generation loop
g=D["greedy"]; prompt=["In"," a"," shocking"," finding",","," scientists"," discovered"]
body=f'<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--acc)"/></marker></defs>'
for i in range(3):
    y=8+i*62; ctx="… scientists discovered"+"".join(g[:i])
    body+=f'<rect class="box" x="8" y="{y}" width="250" height="40" rx="8"/>'+t(18,y+25,ctx,"mono","start",11.5)
    body+=f'<path d="M258 {y+20}H290" class="ln" marker-end="url(#ah)"/><rect class="box" x="292" y="{y}" width="80" height="40" rx="8"/>'+t(332,y+25,"GPT-2","","middle")
    body+=f'<path d="M372 {y+20}H404" class="ln" marker-end="url(#ah)"/><rect class="chip hl" x="406" y="{y+4}" width="64" height="32" rx="6"/>'+t(438,y+25,g[i].replace(" ","·"),"mono","middle")
    if i<2: body+=f'<path d="M438 {y+36}V{y+62}" class="ln" style="stroke-dasharray:3 3"/>'
body+=t(8,8+3*62+8,"(greedy decoding: always take the most likely token)","mut","start",11)
F[12]=fig(12,svg(8+3*62+18,body,"Generation loop: feed the text, get one token, append it, repeat"),
 "Autoregressive generation, real GPT-2 output. Each pass produces <i>one</i> token, which is appended to the input for the next pass. The model never writes a sentence at once.","")

# 13 temperature
T=D["temp"]; col={"0.5":"var(--s1)","1.0":"var(--s2)","2.0":"var(--s3)"}; ox,oy,pw,ph=40,16,370,150
mx=max(T["0.5"]); body=f'<line x1="{ox}" y1="{oy+ph}" x2="{ox+pw}" y2="{oy+ph}" class="axis"/><line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+ph}" class="axis"/>'
for g_ in (0,0.1,0.2,0.3,0.4,0.5):
    if g_<=mx*1.001: yy=oy+ph-g_/mx*ph; body+=f'<line x1="{ox}" y1="{yy:.1f}" x2="{ox+pw}" y2="{yy:.1f}" class="grid"/>'+t(ox-6,yy+4,f"{g_:.2f}","mut","end",10)
for k,v in T.items():
    pts=" ".join(f"{ox+i*(pw/14):.1f},{oy+ph-p/mx*ph:.1f}" for i,p in enumerate(v))
    body+=f'<polyline points="{pts}" fill="none" stroke="{col[k]}" stroke-width="2.5" stroke-linejoin="round"/>'
    i=14; body+=f'<circle cx="{ox+pw}" cy="{oy+ph-v[i]/mx*ph:.1f}" r="4" fill="{col[k]}" stroke="var(--bg)" stroke-width="2"/>'
    for i,p in enumerate(v): body+=f'<circle cx="{ox+i*(pw/14):.1f}" cy="{oy+ph-p/mx*ph:.1f}" r="8" fill="transparent"><title>T={k}, rank {i+1}: {p:.3f}</title></circle>'
ly=[("T=0.5 sharp",col["0.5"]),("T=1 as trained",col["1.0"]),("T=2 flat",col["2.0"])]
for i,(l,c) in enumerate(ly): body+=f'<rect x="{ox+10+i*120}" y="{oy+ph+28}" width="12" height="12" rx="2" fill="{c}"/>'+t(ox+28+i*120,oy+ph+39,l,"","start",11)
body+=t(ox+pw/2,oy+ph+22,"tokens ranked by probability (1 = most likely)","mut","middle",10.5)
body+=t(ox,oy-4,"probability","mut","start",10.5)
F[13]=fig(13,svg(oy+ph+50,body,"Next-token probability by rank at three temperatures"),
 "Temperature reshapes one and the same distribution (GPT-2 after <code>The best programming language is</code>; most likely tokens: Python, Java, the). Dividing the logits by T&lt;1 concentrates probability on the leader; T&gt;1 flattens the curve so unlikely tokens get sampled more often.","")

# ---- patch html
p=os.path.join(HERE,"..","..","docs","01_big_picture.html"); h=open(p).read()
h=re.sub(r"\n?<figure class=\"fig\" id=\"fig\d+\">.*?</figure>\n","",h,flags=re.S)  # idempotent
css="""
:root{--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a}
@media(prefers-color-scheme:dark){:root{--s1:#3987e5;--s2:#d95926;--s3:#199e70}}
.fig{margin:1.8rem 0}
figcaption{font:.85rem/1.5 system-ui,sans-serif;color:var(--mut);margin-top:.4rem}
svg .bar,svg .cell{fill:var(--s1)} svg .bar2{fill:var(--s2)}
svg .chip{fill:var(--card);stroke:var(--bd)} svg .chip.hl{stroke:var(--s2);stroke-width:2}
svg .ln{stroke:var(--acc);stroke-width:1.5;fill:none}
svg .axis{stroke:var(--mut);stroke-width:1} svg .grid{stroke:var(--bd);stroke-width:1}
svg .mask{fill:var(--bd);opacity:.35}
svg text.mut{fill:var(--mut)} svg text.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
"""
if "--s1:" not in h: h=h.replace("</style>",css+"</style>",1)
def after(anchor,key,before=False):
    global h
    assert h.count(anchor)==1,(anchor,h.count(anchor))
    h=h.replace(anchor,(F[key]+anchor) if before else (anchor+"\n"+F[key]))
after("<h2>2. The pipeline</h2>",1,True)
after("<!--fig:tokens-->",3)
after("writes small updates back.</p>",4)
after("<!--fig:proj-->","proj"); after("<!--fig:flow-->",5); after("<!--fig:real-->",6); h=h.replace(F[6],F[6]+F[7],1); after("<!--fig:mlp-->","mlp")
after("so it costs nothing extra.</p></div>",8)
after("<h2>4. How the model is trained</h2>",9)
after("<!--fig:surprisal-->",10)
after("Module 6 covers both.</p>",11)
after("<h2>5. Generation</h2>",12)
after("</ul>\n<p>Naively",13,True) if False else None
h=re.sub(r"</ul>\s*<p>Naively re-running",lambda m:"</ul>\n"+F[13]+"<p>Naively re-running",h,count=1)
# wrap pipeline svg
if 'id="pipe"' not in h:
    h=re.sub(r'(<svg viewBox="0 0 640 330".*?</svg>)',r'<figure class="fig" id="pipe">\1<figcaption><b>Figure 2.</b> The whole pipeline for one forward pass. The dashed loop on the right is generation: the sampled token is appended and the process repeats.</figcaption></figure>',h,count=1,flags=re.S)
open(p,"w").write(h); print(h.count("<figure"))
