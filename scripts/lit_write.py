import json,sys,time
sys.argv=['x','phase1']
src=open('scripts/lit_scan.py').read().split("if __name__")[0]
exec(src)
out=json.load(open('logs/lit_scan_done.json'))
for x in out:
    try: x['relevance']=float(x.get('relevance',0))
    except Exception: x['relevance']=0.0
out.sort(key=lambda x:-x['relevance'])
top=[x for x in out if x['src']=='arxiv'][:25]
digest='\n'.join(f"- {x['title']} ({x['url']}) rel {x['relevance']:.0f}: {x.get('summary','')} IDEAS: {x.get('ideas','')}" for x in top)
syn=ask(CTX+"\n\nHere are the most relevant recent items:\n"+digest+"\n\nWrite markdown with two sections: '## Top 5 most relevant' (5 bullets: title+URL, why it matters to us, one concrete small-scale test) and '## Suggested new avenues' (3-6 env-gated experiments for a ~10M-param looped transformer, each one line with expected signal). Use only the items given.",3000)
with open('LITERATURE_SCAN_2026-10.md','w') as f:
    f.write(f"# Literature scan: looped / recurrent-depth transformers, 2026-07-29 → 2026-10-29\n\nGenerated {time.strftime('%Y-%m-%d %H:%M')} by scripts/lit_scan.py: arXiv API (cs.LG/cs.CL/cs.AI, {len(Q)} queries, keyword-filtered) + HF model search, 200 items. Summariser: {MODEL} (NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) via the Spark 2 vLLM endpoint used by Hermes ({ENDPOINT}), approved by David 2026-10-29. Loop-related arXiv items were read as abstract + up to 14k chars of arXiv HTML full text (see 'read'). Summaries are LLM-generated and unverified: check the paper before relying on a number.\n\n")
    f.write(syn+'\n\n')
    for tier,lo,hi in (('High relevance (8-10)',8,99),('Medium (5-7)',5,8),('Low (0-4)',-99,5)):
        f.write(f'## {tier}\n\n')
        for it in out:
            r=it['relevance']
            if not (lo<=r<hi): continue
            if lo==-99: f.write(f"- [{it['title']}]({it['url']}) ({it['date']}) rel {r:.0f}\n"); continue
            f.write(f"### {it['title']}\n{it['authors']} · {it['date']} · {it['url']} · relevance {r:.0f} · read: {it.get('read','')}\n\n{it.get('summary','')}\n\n")
            if it.get('key_numbers'): f.write(f"Key numbers: {it['key_numbers']}\n\n")
            if it.get('ideas'): f.write(f"Ideas for us: {it['ideas']}\n\n")
log('WRITE DONE (rewritten by lit_write.py after phase2 process exited before writing)')
print(sum(1 for x in out if x['relevance']>=8), sum(1 for x in out if 5<=x['relevance']<8))
