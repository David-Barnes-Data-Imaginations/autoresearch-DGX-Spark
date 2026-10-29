"""Literature scan 2026-07-29..2026-10-29 on looped/recurrent-depth transformers.
Phase 1 (CPU, now): arXiv API + HF model search -> logs/lit_scan_raw.json
Phase 2 (GPU, only after RA-15 finishes and no training running): local Ollama model summarises each item
-> LITERATURE_SCAN_2026-10.md. Model unloaded afterwards (keep_alive=0)."""
import json,time,re,subprocess,urllib.request,urllib.parse,xml.etree.ElementTree as ET,os,sys
os.chdir('/home/david-barnes/autoresearch-DGX-Spark')
LOG=open('logs/lit_scan.log','a',buffering=1)
def log(*a): print(time.strftime('%H:%M:%S'),*a,file=LOG)
MODEL=os.environ.get('LIT_MODEL','qwen3-coder-next:q4_K_M')
Q=['looped transformer','looped language model','recurrent depth','universal transformer','adaptive computation time',
   'ponder','early exit transformer','parameter sharing depth transformer','recursive transformer','latent reasoning recurrence',
   'test-time compute recurrence','elastic looped','Ouro','Hyperloop','loop count','depth recurrence language model',
   'weight-tied layers','mixture of recursions','implicit transformer fixed point','layer looping']
KW=re.compile(r'loop|recurren|recursi|universal transformer|ponder|halting|adaptive computation|early.exit|weight.(tied|shar)|parameter.shar|depth|fixed.point|iterat',re.I)
def arxiv(q):
    s=f'(cat:cs.LG OR cat:cs.CL OR cat:cs.AI) AND all:"{q}" AND submittedDate:[202607290000 TO 202610292359]'
    u='http://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':s,'max_results':60,'sortBy':'submittedDate'})
    x=ET.fromstring(urllib.request.urlopen(u,timeout=60).read()); ns={'a':'http://www.w3.org/2005/Atom'}
    for e in x.findall('a:entry',ns):
        yield {'src':'arxiv','title':' '.join(e.find('a:title',ns).text.split()),'url':e.find('a:id',ns).text,
               'date':e.find('a:published',ns).text[:10],'authors':', '.join(a.find('a:name',ns).text for a in e.findall('a:author',ns)[:6]),
               'abstract':' '.join(e.find('a:summary',ns).text.split())}
def phase1():
    items={}
    for q in Q:
        try:
            n=0
            for it in arxiv(q):
                if KW.search(it['title']+' '+it['abstract']) and it['url'] not in items: items[it['url']]=it; n+=1
            log(f'arxiv "{q}": +{n} (total {len(items)})')
        except Exception as ex: log('arxiv fail',q,ex)
        time.sleep(3.5)
    for q in ['looped','recurrent-depth','dory','ouro','huginn']:
        try:
            for m in json.load(urllib.request.urlopen(f'https://huggingface.co/api/models?search={q}&sort=createdAt&direction=-1&limit=20',timeout=30)):
                d=m.get('createdAt','')[:10]
                if d>='2026-07-29': items['hf:'+m['id']]={'src':'hf-model','title':m['id'],'url':'https://huggingface.co/'+m['id'],'date':d,'authors':m['id'].split('/')[0],'abstract':'(model repo; tags: '+', '.join(m.get('tags',[])[:12])+')'}
            log(f'hf "{q}" total {len(items)}')
        except Exception as ex: log('hf fail',q,ex)
    json.dump(list(items.values()),open('logs/lit_scan_raw.json','w'),indent=1); log('PHASE1 DONE items',len(items))
def gpu_busy():
    return subprocess.run(['pgrep','-f','python training/nano_mythos_train.py'],capture_output=True).returncode==0
def ask(prompt):
    body=json.dumps({'model':MODEL,'prompt':prompt,'stream':False,'options':{'temperature':0.2,'num_ctx':8192},'keep_alive':'10m'}).encode()
    r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:11434/api/generate',body,{'Content-Type':'application/json'}),timeout=600)
    return json.load(r)['response']
CTX=("Our project: small (~10M param) looped transformer 'nano_mythos': prelude 2 layers, 1-layer shared recurrent block looped up to 8 times "
     "with ACT halting + Poisson loop-depth sampling, LTI input reinjection (Parcae), MoD token routing, Hyperloop multi-stream residual, Muon optimizer. "
     "We are now testing FIXED loop counts (Dory-style, 2-16 loops), deeper recurrent blocks and prelude/coda ratios, ELT-style early-exit self-distillation, "
     "and planning a 1B-parameter run on two DGX Spark GPUs.")
def phase2():
    items=json.load(open('logs/lit_scan_raw.json'))
    while gpu_busy() or 'RA15 RUNS COMPLETE' not in open('logs/ra15_full.log',errors='ignore').read():
        log('waiting: RA-15 sweep still running'); time.sleep(300)
    time.sleep(60)
    while gpu_busy(): log('waiting: training on GPU'); time.sleep(300)
    log('PHASE2 start model',MODEL,'items',len(items)); out=[]
    for i,it in enumerate(items):
        p=(f"{CTX}\n\nItem: {it['title']} ({it['date']}, {it['authors']})\n{it['url']}\nAbstract: {it['abstract'][:2500]}\n\n"
           "Reply ONLY with JSON: {\"relevance\":0-10 (10 = directly about looped/recurrent-depth/loop-count/halting for LMs),"
           "\"summary\":\"2-4 sentences, only facts from the abstract\",\"key_numbers\":\"numbers stated in the abstract or ''\","
           "\"ideas\":\"1-2 concrete small-scale experiments for our project, or ''\"}")
        try:
            r=ask(p); j=json.loads(re.search(r'\{.*\}',r,re.S).group(0)); it.update(j)
        except Exception as ex: it['relevance']=-1; it['summary']=f'(LLM failed: {ex})'
        out.append(it); log(f'{i+1}/{len(items)} rel={it.get("relevance")} {it["title"][:70]}')
        if gpu_busy(): log('training started; pausing'); 
        while gpu_busy(): time.sleep(120)
    out.sort(key=lambda x:-(x.get('relevance') or 0) if isinstance(x.get('relevance'),(int,float)) else 0)
    with open('LITERATURE_SCAN_2026-10.md','w') as f:
        f.write(f"# Literature scan: looped / recurrent-depth transformers, 2026-07-29 → 2026-10-29\n\nGenerated {time.strftime('%Y-%m-%d %H:%M')} by lit_scan.py: arXiv API (cs.LG/cs.CL/cs.AI, {len(Q)} queries, keyword-filtered) + HF model search; summaries by local {MODEL} via Ollama from abstracts only (unverified — check the paper before relying on a number). Hermes not used: its configured model runs on Spark 2.\n\n")
        for tier,lo in (('High relevance (7-10)',7),('Medium (4-6)',4),('Low (0-3)',-99)):
            f.write(f'## {tier}\n\n')
            for it in out:
                r=it.get('relevance',0) if isinstance(it.get('relevance'),(int,float)) else 0
                if (r>=lo) and (lo==-99 and r<4 or lo==4 and r<7 or lo==7):
                    if lo==-99: f.write(f"- [{it['title']}]({it['url']}) ({it['date']}) rel {r}\n"); continue
                    f.write(f"### {it['title']}\n{it['authors']} · {it['date']} · {it['url']} · relevance {r}\n\n{it.get('summary','')}\n\n")
                    if it.get('key_numbers'): f.write(f"Key numbers: {it['key_numbers']}\n\n")
                    if it.get('ideas'): f.write(f"Ideas for us: {it['ideas']}\n\n")
    try: urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:11434/api/generate',json.dumps({'model':MODEL,'keep_alive':0}).encode()),timeout=60)
    except Exception: pass
    log('PHASE2 DONE; model unloaded; wrote LITERATURE_SCAN_2026-10.md')
if __name__=='__main__':
    if 'phase2' not in sys.argv: phase1()
    if 'phase1' not in sys.argv: phase2()
