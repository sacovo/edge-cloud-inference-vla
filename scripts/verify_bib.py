"""Check every arXiv ID in literature/bib/*.bib (the source files) against the arXiv API; flag title mismatches."""
import re, os, sys, glob, time, urllib.request, xml.etree.ElementTree as ET, difflib
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','literature')
txt='\n'.join(open(f).read() for f in sorted(glob.glob(os.path.join(ROOT,'bib','*.bib'))))
items=[]
for p in re.split(r'(?m)^(?=@)', txt):
    m=re.match(r'@\w+\s*\{\s*([^,\s]+)',p)
    if not m: continue
    ax=re.search(r'(?:arxiv\.org/(?:abs|pdf)/|eprint\s*=\s*[{"])(\d{4}\.\d{4,5})',p,re.I)
    t=re.search(r'\btitle\s*=\s*[{"]+(.+?)[}"]+\s*,',p,re.I|re.S)
    if ax and t: items.append((m.group(1),ax.group(1),t.group(1)))
# Published title differs from arXiv title (checked by hand 2026-10-27)
ALIASES={'duisterhof2021tinyrl'}
norm=lambda s: re.sub(r'[^a-z0-9 ]','',re.sub(r'\s+',' ',s.lower())).strip()
ns={'a':'http://www.w3.org/2005/Atom'}; bad=0
for i in range(0,len(items),50):
    chunk=items[i:i+50]
    url='http://export.arxiv.org/api/query?max_results=100&id_list='+','.join(c[1] for c in chunk)
    root=ET.fromstring(urllib.request.urlopen(url,timeout=60).read())
    got={}
    for e in root.findall('a:entry',ns):
        aid=re.search(r'(\d{4}\.\d{4,5})',e.find('a:id',ns).text).group(1)
        got[aid]=e.find('a:title',ns).text
    for key,aid,title in chunk:
        real=got.get(aid)
        r=difflib.SequenceMatcher(None,norm(title),norm(real or '')).ratio()
        a,b=norm(title),norm(real or '')
        if r<0.8 and not (b.startswith(a) or a.startswith(b)) and key not in ALIASES:
            bad+=1; print(f'MISMATCH {key} {aid}\n  bib:   {title}\n  arXiv: {real}')
    time.sleep(3)
print(f'checked {len(items)} arXiv entries, {bad} flagged')
sys.exit(1 if bad else 0)
