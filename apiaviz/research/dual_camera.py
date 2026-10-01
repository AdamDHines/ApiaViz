"""Co-registered spectral/visible camera with an owned isolated Mitsuba worker.

Only image data reaches policies. No geometry, labels or contact information is
exposed by scan(). Visible bands have exactly zero response below 400 nm.
"""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

import numpy as np

from .spectral_input import file_sha

ROOT=Path(__file__).resolve().parents[2]
BANDS=['uv','blue','green','visible_red','visible_green','visible_blue']


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def canonical_position(position,decimals=None):
    p=np.asarray(position,dtype=float)
    if p.shape!=(2,) or not np.isfinite(p).all(): raise ValueError('Finite XY required')
    if decimals is not None:
        if decimals!=8: raise ValueError('Only the versioned 10nm camera grid is supported')
        p=np.round(p,decimals)
    return [float(v)+0. for v in p]


def position_key(position,decimals=None):
    return digest(canonical_position(position,decimals))


def render_seed(key,config):
    policy=config.get('seed_policy','position-v1')
    if policy=='position-v1': return (config['seed']+int(key[:8],16))%(2**32)
    if policy=='common-v2': return config['seed']%(2**32)
    raise ValueError('Unknown camera seed policy')


def visible_weights(wavelengths):
    """Declared analytical visible camera proxy; not a calibrated RGB camera."""
    wl=np.asarray(wavelengths)
    curves=np.array([np.exp(-.5*((wl-peak)/width)**2)*wl for peak,width in ((610,35),(540,30),(460,25))])
    # The 400 nm knot must also be zero: linear interpolation from 395 to a
    # positive 400 nm value would otherwise leak a narrow ultraviolet band.
    curves[:,wl<=400]=0.
    return curves/np.trapz(curves,wl,axis=1)[:,None]


def visible_image(raw,white=1.):
    if not np.isfinite(white) or white<=0: raise ValueError('Positive visible white level required')
    # Fixed camera response, shared across all methods and views. Not fitted to
    # outcomes. Raw linear radiances remain separate in the six-band cache.
    return raw/(raw+white)


def sensor_views(raw,headings,config,*,uv=False):
    """Shared acquisition contract for teaching, recall, avoidance and reports."""
    from .uv_input import sample_receptors
    revision=config.get('retinal_sampling','bilinear-legacy')
    if revision=='solid-angle-box-before-response-v1':
        from .retinal_camera import integrate_receptors
        linear=integrate_receptors(raw[:,:,:3] if uv else raw[:,:,3:],headings,
            elevation=config['elevation_deg'],shape=tuple(config['retina_shape']))
        return linear if uv else visible_image(linear,config['visible_white'])
    if revision!='bilinear-legacy':raise ValueError('Unknown retinal acquisition revision')
    data=raw[:,:,:3] if uv else visible_image(raw[:,:,3:],config['visible_white'])
    return sample_receptors(data,headings,elevation=config['elevation_deg'],shape=tuple(config['retina_shape']))


def load_cached(cache,key,render_hash):
    path=Path(cache)/f'{key}.json'
    record=json.loads(path.read_text())
    if (record.get('schema') not in ('apiaviz-dual-frame-v1','apiaviz-dual-frame-v2') or record.get('linear') is not True
            or record['render_hash']!=render_hash or record['channels']!=BANDS):
        raise ValueError('Camera cache protocol/channel mismatch')
    decimals=record.get('pose_decimals')
    if record['schema']=='apiaviz-dual-frame-v2' and decimals!=8:
        raise ValueError('Missing canonical camera pose convention')
    if position_key(record['position'],decimals)!=key: raise ValueError('Camera pose hash mismatch')
    array=path.with_suffix('.npz')
    if file_sha(array)!=record['array_sha256']: raise ValueError('Camera cache checksum mismatch')
    with np.load(array,allow_pickle=False) as archive:
        data=archive['responses']
    if data.ndim!=3 or data.shape[-1]!=6 or data.dtype.kind!='f' or not np.isfinite(data).all() or np.any(data<0):
        raise ValueError('Invalid linear six-band frame')
    if list(data.shape)!=record['shape']: raise ValueError('Camera shape mismatch')
    return data,record


class DualCamera:
    def __init__(self,environment,render_config,*,progress=None,timeout=300.):
        self.environment=Path(environment).resolve()
        self.cache=self.environment/'camera'; self.cache.mkdir(exist_ok=True)
        self.config=render_config
        self.render_hash=digest(render_config)
        self.progress=progress; self.timeout=timeout
        self.frames=OrderedDict(); self.references={}
        self.renders=self.cache_hits=0
        self.process=None; self.log=None
        self._buffer=b''

    def __enter__(self):
        self.log=(self.environment/'spectral-worker.log').open('a')
        command=['pixi','run','--manifest-path',str(ROOT/'scripts/uv_mitsuba/pixi.toml'),'--locked',
            'python',str(ROOT/'scripts/uv_mitsuba/trial_camera.py'),'--environment',str(self.environment)]
        self.process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.log,text=True,bufsize=1,start_new_session=True)
        try:
            ready=self._read()
            if ready.get('render_hash')!=self.render_hash: raise ValueError('Renderer handshake mismatch')
        except BaseException:
            self.__exit__(None,None,None); raise
        return self

    def _read(self):
        deadline=time.monotonic()+self.timeout
        while time.monotonic()<deadline:
            if b'\n' not in self._buffer:
                if not select.select([self.process.stdout],[],[],max(0.,deadline-time.monotonic()))[0]: break
                block=os.read(self.process.stdout.fileno(),65536)
                if not block: raise RuntimeError('Spectral worker exited; inspect spectral-worker.log')
                self._buffer+=block
                continue
            line,self._buffer=self._buffer.split(b'\n',1)
            line=line.decode()
            if not line.startswith('APIAVIZ '):
                self.log.write(line); self.log.flush(); continue
            result=json.loads(line[len('APIAVIZ '):])
            if not result.get('ok'): raise RuntimeError(result.get('error','Spectral render failed'))
            return result
        raise TimeoutError('Spectral worker timed out; inspect spectral-worker.log')

    def frame(self,position):
        position=canonical_position(position,self.config.get('pose_decimals'))
        key=self.key(position)
        if key in self.frames:
            self.cache_hits+=1; self.frames.move_to_end(key)
            return self.frames[key]
        if not (self.cache/f'{key}.json').exists():
            if self.process is None: raise FileNotFoundError(f'Camera cache miss: {key}')
            self.process.stdin.write(json.dumps(dict(position=np.asarray(position).tolist()))+'\n')
            self.process.stdin.flush(); result=self._read()
            if result['key']!=key: raise ValueError('Renderer returned wrong pose')
            self.renders+=1
        else: self.cache_hits+=1
        data,record=load_cached(self.cache,key,self.render_hash)
        self.references[key]=file_sha(self.cache/f'{key}.json')
        self.frames[key]=data
        if len(self.frames)>64: self.frames.popitem(last=False)
        if self.progress: self.progress(self.renders,self.cache_hits)
        return data

    def key(self,position):
        return position_key(position,self.config.get('pose_decimals'))

    def scan(self,position,headings,*,uv=False):
        return sensor_views(self.frame(position),headings,self.config,uv=uv)

    def __exit__(self,*unused):
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    try:
                        self.process.stdin.write('{"stop":true}\n'); self.process.stdin.flush()
                        self.process.wait(timeout=5)
                    except (BrokenPipeError,subprocess.TimeoutExpired):
                        os.killpg(self.process.pid,signal.SIGTERM)
                        try: self.process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(self.process.pid,signal.SIGKILL); self.process.wait()
            finally:
                self.process.stdin.close(); self.process.stdout.close()
                self.process=None
        if self.log: self.log.close(); self.log=None
