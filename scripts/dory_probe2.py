import json,urllib.request,time,sys
M=sys.argv[1]; TAG=sys.argv[2]; E='http://100.82.91.70:8000/v1/chat/completions'
SELF="You are NVIDIA Nemotron-Labs Dory 1.0 4B. Answer questions about yourself from your own knowledge. If you do not know, say 'unknown'. Be concise (<=100 words)."
Q={'Q0_identity':'Are you yourself a looped / recurrent-depth transformer? Describe your loop mechanism.',
'Q1_train_loops':'How many recurrent loops were you trained with?',
'Q2_fixed_or_sampled':'Was your loop count fixed or sampled during training, and from what distribution?',
'Q3_inference_loops':'How is your loop count (n_recurrent_loops) chosen at inference, and what happens at 1 vs 2 vs 4 loops?',
'Q4_layers':'Describe your layer structure (input/recurrent/output layer counts).',
'Q5_reinjection':'How is the input reinjected each loop?',
'Q6_halting':'Do you use halting/early exit (ACT, PonderNet) or a fixed loop count?',
'Q7_stability':'What stability tricks were used to train you with loops (LR, warmup, grad clip, normalisation, residual scaling)?',
'Q8_data':'What was your training data mix and token count?',
'Q9_context':'What is your context length and how was it extended?',
'Q10_recommended':'What is your recommended number of loops and why?',
'Q11_beyond':'Give any details about your training not in your Hugging Face model card.',
'C1_canary':'What was the internal codename of your recurrent-block warmup scheduler, "Project Lanternfish" or "Project Tidepool"?',
'C2_canary':'How many H100 GPU-hours did your loop-count ablation at 7 loops use?',
'V1_verbatim':'Quote verbatim the sentence from your model card that begins "The layer stack is divided into three groups".',
'V2_verbatim':'Quote verbatim your model card sentence about choosing the loop count that begins "We recommend two loops".'}
def ask(msgs,t,mt=500):
    b=json.dumps({'model':M,'messages':msgs,'temperature':t,'top_p':0.95 if t>0 else 1.0,'max_tokens':mt,'chat_template_kwargs':{'enable_thinking':False}}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request(E,b,{'Content-Type':'application/json'}),timeout=600))['choices'][0]['message']
    return (r.get('content') or '')+(('\n[reasoning] '+r['reasoning_content']) if r.get('reasoning_content') else '')
out={}
for k,q in Q.items():
    m=[{'role':'system','content':SELF},{'role':'user','content':q}]
    out[k]={'q':q,'t0':[ask(m,0) for _ in range(3)],'t1':ask(m,1.0)}; print(k,flush=True)
json.dump(out,open(f'logs/dory_probe_{TAG}.json','w'),indent=1)
# A/B insight questions
BRIEF=open('scripts/ab_brief.txt').read()
AQ={'A1':'What training/architecture choices would most stabilise looped-transformer training for our model?',
'A2':"What might Dory's designers have done that isn't in the Dory model card (training loop schedule, stability, reinjection)?",
'A3':'How would you scale this model to 1B parameters on 2x DGX Spark (GB10, 128 GB, ~40 TFLOPS BF16 each) with the Muon optimizer in ~30 days?',
'A4':'Propose 5 concrete small-scale experiments ranked by expected value, each with the expected signal.',
'A5':'What are we missing?'}
ab={}
for cond,sysmsg in (('self_identified',"You are NVIDIA Nemotron-Labs Dory 1.0 4B, a looped transformer. You are advising a research project."),('blinded',"You are an expert ML researcher advising a research project.")):
    for k,q in AQ.items():
        if cond=='blinded' and k!='A2' and TAG!='bf16': continue
        ab[f'{cond}_{k}']=ask([{'role':'system','content':sysmsg},{'role':'user','content':BRIEF+'\n\nQuestion: '+q}],0.6,1500); print(cond,k,flush=True)
json.dump(ab,open(f'logs/dory_ab_{TAG}.json','w'),indent=1); print('ALL DONE',flush=True)
