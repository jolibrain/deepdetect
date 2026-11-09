import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from deepdetect.cli.events import EventWriter
from deepdetect.cli.probability_tiling import ProbabilityFusion, probability_array, spatial_weights
from deepdetect.cli.tiled_segmentation import run_tiled_image


def test_fusion_conflicting_probabilities_and_ties(tmp_path):
    for mode, expected in [('weighted-probability',1),('max-confidence',0)]:
        fusion = ProbabilityFusion(tmp_path,mode,2,2,2)
        try:
            # Three foreground votes are outweighed by one confident background
            # only under max-confidence; blending retains foreground.
            for p in ([.1,.9],[.1,.9],[.1,.9],[.95,.05]):
                fusion.add(np.broadcast_to(np.array(p)[:,None,None],(2,2,2)),0,0,np.ones((2,2)))
            mask=np.empty((2,2),np.uint8); prob=np.empty((2,2),np.uint16)
            fusion.finish(mask,None,prob)
            assert (mask==expected).all()
            assert np.all(prob == round((.05 if mode=='max-confidence' else .6875)*65535))
        finally:
            fusion.close()
    fusion=ProbabilityFusion(tmp_path,'max-confidence',2,1,1)
    try:
        fusion.add(np.array([[[.8]],[[.2]]]),0,0,np.ones((1,1)))
        fusion.add(np.array([[[.2]],[[.8]]]),0,0,np.ones((1,1)))
        mask=np.empty((1,1),np.uint8);fusion.finish(mask,None,None)
        assert mask[0,0]==0
    finally:
        fusion.close()


class Fixture:
    def __init__(self):self.calls=0
    def predict(self,paths,**kwargs):
        self.calls+=1
        assert kwargs['output_parameters']['confidences']==['0','1']
        predictions=[]
        for p in paths:
            with Image.open(p) as im:
                assert im.size==(4,4)
                values=np.asarray(im)[:,:,0]/255
            predictions.append(dict(uri=str(p),imgsize=dict(width=4,height=4),
                confidences={'0':(1-values).flatten().tolist(),'1':values.flatten().tolist()},
                vals=(values>.5).astype(int).flatten().tolist()))
        return {'predictions':predictions[::-1]}


@pytest.mark.parametrize('shape,source_size', [((11,7),8),((3,2),8),((9,9),4)])
def test_source_resize_multi_fusion_and_cleanup(tmp_path,shape,source_size):
    path=tmp_path/'white image.png';Image.new('RGB',shape,'white').save(path)
    service=Fixture();out=tmp_path/'out'
    result=run_tiled_image(service=service,image_path=path,output_dir=out,predict_parameters={},
        tile_width=4,tile_height=4,source_size=source_size,overlap=source_size//2,
        batch_size=2,nclasses=2,preview_max_side=20,confidence_maps=True,
        writer=EventWriter(stream=io.StringIO()),padding='edge',
        merge_modes=('weighted-probability','max-confidence'))
    assert service.calls==(result.tiles+1)//2
    for mode in ('weighted-probability','max-confidence'):
        with Image.open(out/mode/'white image_mask.png') as im:
            assert im.size==shape;assert (np.asarray(im)==1).all()
        with Image.open(out/mode/'white image_foreground_probability.png') as im:
            assert (np.asarray(im)==65535).all()
    assert not list(out.glob('.tile-fusion-*'))


@pytest.mark.parametrize('maps', [{}, {'0':[.2]*4,'1':[.2]*4}, {'0':[float('nan')]*4,'1':[1]*4}, {'0':[0]*3,'1':[1]*3}])
def test_reject_bad_probabilities(maps):
    with pytest.raises(ValueError):
        probability_array(dict(imgsize={'width':2,'height':2},confidences=maps),2,2,2)


def test_spatial_weight_edges_and_symmetry():
    w=spatial_weights(8,6)
    assert (w>0).all()
    np.testing.assert_allclose(w,w[::-1,::-1])
    assert w[4,3]>w[0,0]


def test_partial_failure_cleans_accumulators(tmp_path):
    class Broken(Fixture):
        def predict(self,paths,**kwargs):
            result=super().predict(paths,**kwargs)
            result['predictions'][0]['confidences']['1'][0]=float('nan')
            return result
    path=tmp_path/'scan.png';Image.new('RGB',(10,6),'white').save(path)
    with pytest.raises(ValueError,match='finite'):
        run_tiled_image(service=Broken(),image_path=path,output_dir=tmp_path/'out',predict_parameters={},
            tile_width=4,tile_height=4,overlap=2,batch_size=1,nclasses=2,preview_max_side=8,
            confidence_maps=True,writer=EventWriter(stream=io.StringIO()),
            merge_modes=('weighted-probability','max-confidence'),padding='edge')
    assert not list((tmp_path/'out').glob('.tile-fusion-*'))


def test_spatially_weighted_overlap_preserves_normalized_vector(tmp_path):
    fusion=ProbabilityFusion(tmp_path,'weighted-probability',2,1,3)
    try:
        fusion.add(np.array([[[.9,.9]],[[.1,.1]]],np.float32),0,0,np.array([[1,.25]]))
        fusion.add(np.array([[[.2,.2]],[[.8,.8]]],np.float32),1,0,np.array([[.75,1]]))
        mask=np.empty((1,3),np.uint8);prob=np.empty((1,3),np.uint16)
        fusion.finish(mask,None,prob)
        np.testing.assert_array_equal(mask,[[0,1,1]])
        np.testing.assert_allclose(prob/65535,[[.1,.625,.8]],atol=1/65535)
    finally:fusion.close()


def test_irregular_overlap_preserves_source_pixel_coordinates(tmp_path):
    yy,xx=np.mgrid[:7,:11]
    values=(xx*17+yy*5).astype(np.uint8)
    path=tmp_path/'ramp.png';Image.fromarray(np.repeat(values[:,:,None],3,axis=2)).save(path)
    run_tiled_image(service=Fixture(),image_path=path,output_dir=tmp_path/'out',predict_parameters={},
        tile_width=4,tile_height=4,source_size=4,overlap=2,batch_size=3,nclasses=2,
        preview_max_side=20,confidence_maps=True,writer=EventWriter(stream=io.StringIO()),
        merge_modes=('weighted-probability','max-confidence'),padding='edge')
    for mode in ('weighted-probability','max-confidence'):
        with Image.open(tmp_path/'out'/mode/'ramp_foreground_probability.png') as im:
            np.testing.assert_allclose(np.array(im),values.astype(np.uint16)*257,atol=1,rtol=0)
        with Image.open(tmp_path/'out'/mode/'ramp_mask.png') as im:
            np.testing.assert_array_equal(np.array(im),values>127)
