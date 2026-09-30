import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from apiaviz.research.collision_geometry import SCHEMA, sha256, protocol_geometry, RockGeometry
from scripts.prepare_mesh_protocol import prepare


class MeshProtocolTests(unittest.TestCase):
    def test_fresh_protocol_freezes_physics_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); env=root/'env'; env.mkdir()
            (env/'grassland.blend').write_bytes(b'synthetic test bytes, never rendered')
            (env/'protocol.json').write_text(json.dumps(dict(route=[[2,2],[3,2]],headings=[0,0],world_bounds_m=[-5,5,-5,5])))
            digest=sha256(env/'grassland.blend')
            meta=dict(scene_sha256=digest,obstacles=[dict(x=0,y=0,conservative_radius_m=1)])
            (env/'world.json').write_text(json.dumps(meta))
            encoder=root/'encoder'; encoder.mkdir()
            (encoder/'encoder-19.pt').write_bytes(b'fixture checkpoint, never loaded')
            mesh=root/'collision.json'
            mesh.write_text(json.dumps(dict(schema=SCHEMA,scene_sha256=digest,rocks=[
                dict(name='rock',triangles_xy_m=[[[0,0],[1,0],[0,1]]])])) )
            p=dict(worlds=[dict(name='fixture',environment='old-machine',protocol_sha256=sha256(env/'protocol.json'))],
                seeds=[19],checkpoint_sha256={'19':sha256(encoder/'encoder-19.pt')},
                scene_sha256={'fixture':digest},trials=1,analysis={},integration_mode='motor_feedback',
                scenarios=[dict(name='aligned',lateral=0.,heading=0.,kick=0.)],
                controllers=[dict(name='familiarity',phase=1)])
            source=root/'source.json'; source.write_text(json.dumps(p)); before=source.read_bytes()
            args=SimpleNamespace(source=source,output=root/'new-v1',world=[['fixture',str(env),str(mesh)]],
                                 encoder_environment=encoder,body_radius_m=.005)
            prepare(args)
            self.assertEqual(source.read_bytes(),before)
            q=json.loads((args.output/'protocol.json').read_text())
            self.assertEqual(q['physics']['contact_response'],'block')
            self.assertEqual(q['evaluation_safety']['schema'],'navigation-safety-v1')
            self.assertFalse(q['physics']['worlds']['fixture']['release_preflight']['aligned']['adjusted'])
            self.assertEqual(q['source_protocol']['sha256'],sha256(source))
            self.assertIsInstance(protocol_geometry(q,q['worlds'][0],meta),RockGeometry)
            self.assertEqual(json.loads((args.output/'manifest.json').read_text())['status'],'prepared')
            self.assertTrue((args.output/'worlds/fixture/manifest.json').exists())
            with self.assertRaises(FileExistsError): prepare(args)
            mesh.write_text(mesh.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'): protocol_geometry(q,q['worlds'][0],meta)


if __name__ == '__main__': unittest.main()
