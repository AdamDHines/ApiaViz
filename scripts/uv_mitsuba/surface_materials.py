"""Source-informed spectral surfaces. Empirical spectra, simulated spatial detail.

This does not claim Blender Noise equivalence or measured spectral BRDF/BTDF.
The same achromatic spatial modulation applies at every wavelength. No RGB-to-UV
spectral upsampling or learned texture is performed.
"""
from render import mi,dr,np,spectrum,WAVELENGTHS


def noise(p):
    """Deterministic 3D cubic value noise and exact world-coordinate gradient."""
    cell=dr.floor(p);f=p-cell;s=f*f*(3-2*f);ds=6*f*(1-f)
    value=mi.Float(0);gradient=mi.Vector3f(0)
    for x in (0,1):
        for y in (0,1):
            for z in (0,1):
                a=mi.UInt32(mi.Int32(cell.x)+x)*73856093 ^ mi.UInt32(mi.Int32(cell.y)+y)*19349663 ^ mi.UInt32(mi.Int32(cell.z)+z)*83492791
                a=(a^(a>>16))*2246822519;a=(a^(a>>13))*3266489917;a=a^(a>>16)
                v=mi.Float(a & 0xffffff)/16777215
                wx=s.x if x else 1-s.x;wy=s.y if y else 1-s.y;wz=s.z if z else 1-s.z
                value+=v*wx*wy*wz
                gradient+=v*mi.Vector3f((1 if x else -1)*ds.x*wy*wz,
                    (1 if y else -1)*ds.y*wx*wz,(1 if z else -1)*ds.z*wx*wy)
    return value,gradient


def fractal(p,scale,octaves):
    value=mi.Float(0);gradient=mi.Vector3f(0);total=0.
    for i in range(octaves):
        amplitude=.5**i;frequency=scale*2**i
        v,g=noise(p*frequency)
        value+=amplitude*v;gradient+=amplitude*frequency*g;total+=amplitude
    return value/total,gradient/total


def register():
    class SurfaceTexture(mi.Texture):
        def __init__(self,props):
            super().__init__(props)
            self.scale=float(props['scale']);self.octaves=int(props['octaves'])
            self.dark=float(props.get('dark',0.));self.nested=props.get('nested',None)
            self.ramp_low=float(props.get('ramp_low',0.));self.ramp_high=float(props.get('ramp_high',1.))
            if not (self.scale>0 and 1<=self.octaves<=6 and 0<=self.dark<=1 and self.ramp_low<self.ramp_high):
                raise ValueError('Invalid spectral texture')
        def value_gradient(self,si):
            v,g=fractal(si.p,self.scale,self.octaves)
            t=(v-self.ramp_low)/(self.ramp_high-self.ramp_low)
            gain=(1-self.dark)/(self.ramp_high-self.ramp_low)
            return self.dark+(1-self.dark)*dr.clip(t,0,1),dr.select((t>0)&(t<1),g*gain,mi.Vector3f(0))
        def eval(self,si,active=True):
            value,_=self.value_gradient(si)
            return value*(self.nested.eval(si,active) if self.nested is not None else mi.Spectrum(1))
        def eval_1(self,si,active=True):return self.value_gradient(si)[0]
        def eval_3(self,si,active=True):return mi.Color3f(self.eval_1(si,active))
        def eval_1_grad(self,si,active=True):
            _,g=self.value_gradient(si)
            return mi.Vector2f(dr.dot(g,si.dp_du),dr.dot(g,si.dp_dv))
        def mean(self):return (self.dark+1)/2*(float(self.nested.mean()) if self.nested is not None else 1)
        def is_spatially_varying(self):return True
        def to_string(self):return 'SourceInformedSurfaceTexture[]'
    mi.register_texture('source_surface_texture',lambda p:SurfaceTexture(p))


def bsdf(values,source):
    if source['schema']!='source-informed-surfaces-v1':raise ValueError('Unknown surface schema')
    reflectance=dict(type='source_surface_texture',scale=source['colour_scale'],
        octaves=1+int(source['colour_detail']),dark=source['dark_ratio'],
        ramp_low=source['ramp_positions'][0],ramp_high=source['ramp_positions'][-1],
        nested=spectrum(WAVELENGTHS,values))
    leaf=source['transmission_mix']>0
    material=dict(type='principledthin' if leaf else 'principled',base_color=reflectance,
                  roughness=source['roughness'],spec_trans=0.)
    if leaf:material['diff_trans']=2*source['transmission_mix']
    # Principledthin is already two-sided. The opaque branch needs the adapter.
    if not leaf:material=dict(type='twosided',nested=material)
    if source['bump_distance_m']>0:
        material=dict(type='bumpmap',scale=source['bump_distance_m']*source['bump_strength'],
            texture=dict(type='source_surface_texture',scale=source['bump_scale'],
                         octaves=1+int(source['bump_detail'])),nested=material)
    return mi.load_dict(material)
