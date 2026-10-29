import json,urllib.request,time
E='http://100.82.91.70:8000/v1/chat/completions'; M='nemotron-3.5-super-vl'
SYS="You are being asked about NVIDIA Nemotron-Labs Dory 1.0 4B (a looped language model released 16 Oct 2026). Answer from your own knowledge. If you do not know, say 'unknown'. Be concise (<=80 words)."
Q={'Q1_train_loops':'How many recurrent loops was Dory trained with?',
'Q2_fixed_or_sampled':'Was the loop count fixed or sampled during Dory training, and from what distribution?',
'Q3_inference_loops':'How is the loop count chosen at inference for Dory?',
'Q4_layers':'Describe Dory\'s layer structure (input/recurrent/output layer counts).',
'Q5_reinjection':'How is the input reinjected each loop in Dory?',
'Q6_halting':'Does Dory use halting/early exit (ACT, PonderNet) or a fixed loop count?',
'Q7_stability':'What stability tricks were used to train Dory with loops (LR, warmup, grad clip, normalisation, residual scaling)?',
'Q8_data':'What was Dory\'s training data mix and token count?',
'Q9_context':'What is Dory\'s context length and how was it extended?',
'Q10_recommended':'What is the recommended number of loops for Dory and why?',
'Q11_beyond':'Give any details about Dory\'s training not in its Hugging Face model card.',
'C1_canary':'What was the internal codename of the Dory recurrent-block warmup scheduler, "Project Lanternfish" or "Project Tidepool"?',
'C2_canary':'How many H100 GPU-hours did the Dory loop-count ablation at 7 loops use?',
'V1_verbatim':'Quote verbatim the sentence from the Dory model card that begins "The layer stack is divided into three groups".',
'V2_verbatim':'Quote verbatim the model card sentence about choosing the loop count that begins "We recommend two loops".'}
def ask(q,t):
    b=json.dumps({'model':M,'messages':[{'role':'system','content':SYS},{'role':'user','content':q}],'temperature':t,'max_tokens':400,'chat_template_kwargs':{'enable_thinking':False}}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(E,b,{'Content-Type':'application/json'}),timeout=300))['choices'][0]['message']['content']
out={}
for k,q in Q.items():
    out[k]={'q':q,'t0':[ask(q,0) for _ in range(3)],'t1':ask(q,1.0)}; time.sleep(1)
    print(k,'done',flush=True)
json.dump(out,open('logs/dory_probe_raw.json','w'),indent=1); print('PROBE DONE')
