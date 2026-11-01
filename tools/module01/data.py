import os, torch, json, math
from transformers import AutoTokenizer, AutoModelForCausalLM
tok=AutoTokenizer.from_pretrained("gpt2"); m=AutoModelForCausalLM.from_pretrained("gpt2",attn_implementation="eager").eval()
D={}
def logits(t):
    with torch.no_grad(): return m(tok(t,return_tensors="pt").input_ids).logits[0,-1]
p=torch.softmax(logits("I put the milk in the"),-1); top=torch.topk(p,10)
D["milk"]=[(tok.decode([i]),float(v)) for v,i in zip(top.values,top.indices)]; D["milk_rest"]=float(1-top.values.sum())
l=logits("The best programming language is")
D["temp"]={T:[float(x) for x in torch.softmax(l/T,-1).sort(descending=True).values[:15]] for T in (0.5,1.0,2.0)}
D["temp_top"]=[tok.decode([i]) for i in torch.topk(l,5).indices]
t="When Mary and John went to the store, John gave a drink to"
x=tok(t,return_tensors="pt")
with torch.no_grad(): o=m(**x,output_attentions=True)
labs=[tok.decode([i]) for i in x.input_ids[0]]; D["labs"]=labs
# find head whose last row puts most weight on ' Mary'
mi=labs.index(" Mary"); best=[]
for L in range(12):
    for H in range(12): best.append((float(o.attentions[L][0,H,-1,mi]),L,H))
best.sort(reverse=True); D["best"]=best[:5]
bl,bL,bH=best[0][1],best[0][2],0
L,H=9,6; D["head"]=(L,H); D["attn"]=o.attentions[L][0,H].tolist()
D["attn_prev"]=o.attentions[5][0,1].tolist()
ids=x.input_ids[0].tolist()
text="The cat sat on the mat. Unbelievably, tokenization is weird!"
D["tok"]=[(tok.decode([i]),i) for i in tok.encode(text)]
ids=tok("In a shocking finding, scientists discovered",return_tensors="pt").input_ids
g=[]
for _ in range(4):
    with torch.no_grad(): n=m(ids).logits[0,-1].argmax()
    g.append(tok.decode([n])); ids=torch.cat([ids,n.view(1,1)],1)
D["greedy"]=g
s="The quick brown fox jumps over the lazy dog."; ii=tok(s,return_tensors="pt").input_ids
with torch.no_grad(): lg=m(ii).logits[0,:-1]
nll=-torch.log_softmax(lg,-1)[torch.arange(ii.shape[1]-1),ii[0,1:]]
D["surp"]=[(tok.decode([i]),float(n)) for i,n in zip(ii[0,1:],nll)]
json.dump(D,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"data.json"),"w"))
print(D["best"],D["head"],D["greedy"],D["temp_top"],D["milk_rest"])
