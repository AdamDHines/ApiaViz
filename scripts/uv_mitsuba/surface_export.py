"""Read supported Blender material parameters; do not imply shader equivalence."""
def describe(material):
    nodes=list(material.node_tree.nodes)
    def one(kind):
        found=[n for n in nodes if n.bl_idname==kind]
        if len(found)!=1:raise ValueError(f'{material.name}: expected one {kind}')
        return found[0]
    principled=one('ShaderNodeBsdfPrincipled');ramp=one('ShaderNodeValToRGB')
    colour_noise=ramp.inputs[0].links[0].from_node
    if colour_noise.bl_idname!='ShaderNodeTexNoise':raise ValueError('Unsupported colour texture')
    colours=[list(e.color[:3]) for e in ramp.color_ramp.elements]
    luminance=[sum(a*b for a,b in zip(c,[.2126,.7152,.0722])) for c in colours]
    result=dict(schema='source-informed-surfaces-v1',material=material.name,
        roughness=float(principled.inputs['Roughness'].default_value),
        source_linear_rgb=colours,ramp_positions=[float(e.position) for e in ramp.color_ramp.elements],
        dark_ratio=min(luminance)/max(max(luminance),1e-12),
        colour_scale=float(colour_noise.inputs['Scale'].default_value),
        colour_detail=float(colour_noise.inputs['Detail'].default_value),
        bump_scale=0.,bump_distance_m=0.,bump_strength=0.,bump_detail=0.,transmission_mix=0.,
        approximation='World-space smooth value noise replaces Blender Noise; spectral shape retained with wavelength-neutral contrast. Proxy reflectance is the bright envelope, not the spatial mean. Principled roughness/thin-leaf response is an engineering approximation, not measured spectral BRDF/BTDF.')
    bumps=[n for n in nodes if n.bl_idname=='ShaderNodeBump']
    if bumps:
        bump=one('ShaderNodeBump');noise=bump.inputs['Height'].links[0].from_node
        if noise.bl_idname!='ShaderNodeTexNoise':raise ValueError('Unsupported bump texture')
        result.update(bump_scale=float(noise.inputs['Scale'].default_value),
            bump_detail=float(noise.inputs['Detail'].default_value),
            bump_distance_m=float(bump.inputs['Distance'].default_value),
            bump_strength=float(bump.inputs['Strength'].default_value))
    if any(n.bl_idname=='ShaderNodeBsdfTranslucent' for n in nodes):
        result['transmission_mix']=float(one('ShaderNodeMixShader').inputs[0].default_value)
    return result
