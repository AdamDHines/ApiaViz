"""Download calibration sources with URL, UTC timestamp, length and SHA256."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[2]/'apiaviz/output/uv-calibration/sources'


def fetch(name,url):
    ROOT.mkdir(parents=True,exist_ok=True)
    if Path(name).name!=name:raise ValueError('Use a plain filename')
    target=ROOT/name
    record=target.with_name(target.name+'.provenance.json')
    if target.exists() and record.exists():
        info=json.loads(record.read_text())
        assert info['url']==url
        assert hashlib.sha256(target.read_bytes()).hexdigest()==info['sha256']
        print('CACHED',name,flush=True)
        return
    request=urllib.request.Request(url,headers={'User-Agent':'ApiaViz research calibration (Python urllib)'})
    with urllib.request.urlopen(request,timeout=90) as response:
        temporary=target.with_suffix(target.suffix+'.part')
        digest=hashlib.sha256(); size=0
        with temporary.open('wb') as f:
            while chunk:=response.read(1024*1024):
                f.write(chunk);digest.update(chunk);size+=len(chunk)
        info=dict(url=url,resolved_url=response.url,downloaded_utc=datetime.now(timezone.utc).isoformat(),
                  bytes=size,sha256=digest.hexdigest(),content_type=response.headers.get('Content-Type'))
    temporary.replace(target)
    record.write_text(json.dumps(info,indent=2)+'\n')
    print('SAVED',name,size,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name',nargs='?');parser.add_argument('url',nargs='?')
    parser.add_argument('--manifest',type=Path,help='Restore the named sources from a saved sources.lock.json')
    args=parser.parse_args()
    if args.manifest:
        if args.name or args.url: parser.error('Use either name/url or --manifest')
        for name,record in json.loads(args.manifest.read_text()).items():
            fetch(name,record['url'])
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==record['sha256'], name
    else:
        if not args.name or not args.url: parser.error('Provide name and URL')
        fetch(args.name,args.url)
