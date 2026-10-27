import re, glob, sys, os
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','literature')
entries=[]
for f in sorted(glob.glob(os.path.join(ROOT,'bib/*.bib'))):
    txt=open(f).read()
    # split at entry starts
    parts=re.split(r'(?m)^(?=@)', txt)
    for p in parts:
        m=re.match(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', p)
        if not m or m.group(1).lower() in ('comment','string','preamble'): continue
        body=p.strip()
        ax=re.search(r'(?:arxiv\.org/(?:abs|pdf)/|eprint\s*=\s*[{"])(\d{4}\.\d{4,5})', body, re.I)
        doi=re.search(r'doi\s*=\s*[{"]([^}"]+)', body, re.I)
        title=re.search(r'title\s*=\s*[{"]+(.+?)[}"]+\s*,', body, re.I|re.S)
        entries.append(dict(key=m.group(2), file=f, body=body,
            ax=ax.group(1) if ax else None, doi=doi.group(1).lower() if doi else None,
            title=re.sub(r'\W+',' ',title.group(1).lower()).strip() if title else None))
seen={}; out=[]; dups=[]
for e in entries:
    ids=[('k',e['key'])]+([('a',e['ax'])] if e['ax'] else [])+([('d',e['doi'])] if e['doi'] else [])+([('t',e['title'])] if e['title'] else [])
    hit=next((seen[i] for i in ids if i in seen), None)
    if hit: dups.append((e['key'],e['file'],hit['key'],hit['file'])); continue
    for i in ids: seen[i]=e
    out.append(e)
open(os.path.join(ROOT,'references.bib'),'w').write('% Merged from literature/bib/*.bib; deduplicated by key, arXiv ID, DOI, title.\n% Generated 2026-10-27. Edit per-theme files, then re-run scripts/merge_bib.py.\n\n'+'\n\n'.join(e['body'] for e in out)+'\n')
print(len(entries),'entries ->',len(out),'unique')
for d in dups: print('DUP',*d)
noid=[e['key'] for e in out if not e['ax'] and not e['doi']]
print('no arXiv/DOI:',len(noid),noid)
