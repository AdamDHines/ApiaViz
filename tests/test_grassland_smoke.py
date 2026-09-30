import unittest

import numpy as np
import torch

from apiaviz.research.grassland_smoke import make_route, position_key, sample_panorama
from apiaviz.research.openloop import acquisition_views


class GrasslandGeometryTests(unittest.TestCase):
    def test_physical_azimuth_and_elevation(self):
        # An analytic direction-colour field catches horizontal reversal, vertical
        # reversal, off-by-half-pixel errors and the seam independently of Blender.
        h,w=152,720
        az=np.radians(180-(np.arange(w)+.5)*360/w)
        el=60.5-(np.arange(h)+.5)*76/h
        image=np.empty((h,w,3),dtype=float)
        image[:,:,0]=127.5*(1+np.cos(az))[None,:]
        image[:,:,1]=127.5*(1+np.sin(az))[None,:]
        image[:,:,2]=((el+15.5)/76*255)[:,None]
        headings=np.array([0,90,180,270,359.9])
        views=sample_panorama(image,headings).numpy()
        target=np.radians(headings[:,None]+np.linspace(-148,148,74)[None,:])
        np.testing.assert_allclose(views[:,0,0],(1+np.cos(target))/2,atol=5e-6)
        np.testing.assert_allclose(views[:,1,0],(1+np.sin(target))/2,atol=5e-6)
        np.testing.assert_allclose(views[0,2,:,0],(np.linspace(60,-15,18)+15.5)/76,atol=1e-7)
        self.assertTrue(torch.equal(sample_panorama(image,[0]),sample_panorama(image,[360])))

    def test_route_and_shared_acquisition_budget(self):
        p,h=make_route()
        np.testing.assert_allclose(np.linalg.norm(np.diff(p,axis=0),axis=1),.1,atol=1e-12)
        self.assertTrue(np.all(h%2==0))
        self.assertEqual(len(p),len(h)+1)
        teach,head=acquisition_views(p,h,viewpoints=9,width=.2,lookahead=0)
        self.assertEqual(len(teach),9*len(h))
        np.testing.assert_allclose(teach[4::9],p[:-1],atol=5e-7)
        np.testing.assert_array_equal(head.reshape(-1,9),np.repeat(h[:,None],9,axis=1))
        self.assertEqual(position_key([1.,2.]),position_key(np.array([1.,2.])))
        self.assertNotEqual(position_key([1.,2.]),position_key([1.,2.0000001]))


if __name__=='__main__':
    unittest.main()
