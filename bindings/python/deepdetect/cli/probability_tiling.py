"""Source-scale tiling with reusable, bounded-memory probability fusion."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image

from .tiled_segmentation import (
    TiledImageResult, _atomic_png, _ownership_bounds, _preview_mask, _tile_starts,
)
from .utils import predictions_by_uri
from .visualize import _segmentation_palette

MERGE_MODES = ('center', 'weighted-probability', 'max-confidence')


def spatial_weights(height, width):
    def axis(n):
        return np.maximum(np.sin(np.pi * (np.arange(n) + .5) / n) ** 2, 1e-3)
    return (axis(height)[:, None] * axis(width)[None, :]).astype(np.float32)


def probability_array(prediction, nclasses, width, height):
    if prediction.get('imgsize') != {'width': width, 'height': height}:
        raise ValueError('tiled segmentation prediction has unexpected imgsize')
    maps = prediction.get('confidences', {})
    if not isinstance(maps, dict) or any(str(i) not in maps for i in range(nclasses)):
        raise ValueError('probability fusion requires a confidence map for every class')
    p = np.asarray([maps[str(i)] for i in range(nclasses)], dtype=np.float32)
    if p.shape != (nclasses, height * width):
        raise ValueError('class confidence maps have unexpected lengths')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('class confidence maps must be finite probabilities in [0, 1]')
    mass = p.sum(axis=0)
    if not np.allclose(mass, 1, atol=1e-3, rtol=0):
        raise ValueError('class confidence maps must sum to one')
    return (p / mass).reshape(nclasses, height, width)


class ProbabilityFusion:
    """One accumulator per mode; all receive the same probability vectors."""
    def __init__(self, root, mode, nclasses, height, width):
        if mode not in MERGE_MODES:
            raise ValueError(f'unknown tile merge mode: {mode}')
        self.mode = mode
        self.values = np.memmap(Path(root)/f'{mode}-values.bin', mode='w+', dtype='float32',
                                shape=(nclasses, height, width))
        self.weights = np.memmap(Path(root)/f'{mode}-weights.bin', mode='w+', dtype='float32',
                                 shape=(height, width))
        self.values[:] = 0
        self.weights[:] = -1 if mode == 'max-confidence' else 0

    def add(self, p, x, y, weights):
        h, w = p.shape[1:]
        target = self.values[:, y:y+h, x:x+w]
        current = self.weights[y:y+h, x:x+w]
        if self.mode == 'max-confidence':
            confidence = p.max(axis=0)
            take = confidence > current  # stable first tile wins exact ties
            target[:, take] = p[:, take]
            current[take] = confidence[take]
        else:
            target += p * weights
            current += weights

    def finish(self, mask, confidence, foreground):
        height = mask.shape[0]
        for y in range(0, height, 128):
            block = np.array(self.values[:, y:y+128])
            weight = self.weights[y:y+128]
            if (weight < 0).any() or (self.mode != 'max-confidence' and (weight <= 0).any()):
                raise ValueError('tile fusion left uncovered pixels')
            if self.mode != 'max-confidence':
                block /= weight[None]
            mask[y:y+128] = block.argmax(axis=0)
            if confidence is not None:
                confidence[y:y+128] = np.rint(block.max(axis=0)*65535).astype(np.uint16)
            if foreground is not None:
                foreground[y:y+128] = np.rint(block[1]*65535).astype(np.uint16)

    def close(self):
        self.values._mmap.close()
        self.weights._mmap.close()


def run_probability_tiles(*, service, image_path, output_dir, predict_parameters,
                          tile_width, tile_height, source_size, overlap, batch_size,
                          nclasses, preview_max_side, confidence_maps, writer,
                          warmup, merge_modes, padding):
    if not merge_modes or len(set(merge_modes)) != len(merge_modes) or any(m not in MERGE_MODES for m in merge_modes):
        raise ValueError('tile merge modes must be unique supported modes')
    if padding not in ('constant', 'edge'):
        raise ValueError('tile padding must be constant or edge')
    sw, sh = (source_size, source_size) if source_size is not None else (tile_width, tile_height)
    if min(sw, sh, tile_width, tile_height, batch_size, preview_max_side) < 1 or not 0 <= overlap < min(sw, sh):
        raise ValueError('invalid tile dimensions, overlap, batch size, or preview size')
    if not 1 <= nclasses <= 256:
        raise ValueError('tiled segmentation requires between 1 and 256 classes')
    with Image.open(image_path) as opened:
        source = opened.convert('RGB')
    width, height = source.size
    xs, ys = _tile_starts(width, sw, overlap), _tile_starts(height, sh, overlap)
    xb, yb = _ownership_bounds(xs, sw, width), _ownership_bounds(ys, sh, height)
    tiles = [(x,y,xi,yi) for yi,y in enumerate(ys) for xi,x in enumerate(xs)]
    parameters = copy.deepcopy(predict_parameters)
    parameters.setdefault('output_parameters', {})['confidences'] = [str(i) for i in range(nclasses)]
    # Inputs are already resized; prevent connector settings from changing the grid.
    parameters.setdefault('input_parameters', {}).update(width=tile_width, height=tile_height)
    spatial = spatial_weights(sh,sw)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    elapsed = 0.
    writer.emit('tile_plan', image=str(image_path), width=width, height=height,
                source_width=sw, source_height=sh, tile_width=tile_width, tile_height=tile_height,
                overlap=overlap, tiles=len(tiles), merge_modes=list(merge_modes), padding=padding)
    with tempfile.TemporaryDirectory(prefix='.tile-fusion-', dir=output_dir) as temporary:
        accumulators = []
        try:
            for mode in merge_modes:
                accumulators.append(ProbabilityFusion(temporary, mode, nclasses, height, width))
            for start in range(0,len(tiles),batch_size):
                batch = tiles[start:start+batch_size]
                paths = []
                for i,(x,y,_,_) in enumerate(batch,start):
                    crop = np.asarray(source.crop((x,y,min(x+sw,width),min(y+sh,height))))
                    crop = np.pad(crop, ((0,sh-crop.shape[0]),(0,sw-crop.shape[1]),(0,0)), mode=padding)
                    image = Image.fromarray(crop)
                    if image.size != (tile_width,tile_height):
                        image = image.resize((tile_width,tile_height),Image.Resampling.BILINEAR)
                    path = Path(temporary)/f'tile-{i:08d}.png'
                    image.save(path)
                    paths.append(path)
                if start == 0:
                    for _ in range(warmup):
                        service.predict(paths,**parameters)
                before = time.perf_counter()
                result = service.predict(paths,**parameters)
                elapsed += time.perf_counter()-before
                predictions = predictions_by_uri(paths,result.get('predictions',[]))
                for (x,y,xi,yi),prediction in zip(batch,predictions):
                    p = probability_array(prediction,nclasses,tile_width,tile_height)
                    if (sw,sh) != (tile_width,tile_height):
                        p = np.stack([np.asarray(Image.fromarray(v).resize((sw,sh),Image.Resampling.BILINEAR)) for v in p])
                        p /= p.sum(axis=0,keepdims=True)
                    vh,vw = min(sh,height-y),min(sw,width-x)
                    p = p[:,:vh,:vw]
                    for accumulator in accumulators:
                        weight = spatial[:vh,:vw]
                        if accumulator.mode == 'center':
                            weight = np.zeros((vh,vw),dtype=np.float32)
                            weight[yb[yi]-y:yb[yi+1]-y,xb[xi]-x:xb[xi+1]-x] = 1
                        accumulator.add(p,x,y,weight)
                for path in paths:
                    path.unlink()
                writer.emit('tile_progress',image=str(image_path),processed=min(start+batch_size,len(tiles)),total_tiles=len(tiles))
            for accumulator in accumulators:
                destination = output_dir/accumulator.mode if len(merge_modes)>1 else output_dir
                destination.mkdir(parents=True,exist_ok=True)
                mask = np.empty((height,width),dtype=np.uint8)
                conf = np.empty((height,width),dtype=np.uint16) if confidence_maps else None
                foreground = np.empty((height,width),dtype=np.uint16) if confidence_maps and nclasses==2 else None
                accumulator.finish(mask,conf,foreground)
                result_image = Image.fromarray(mask,mode='P')
                result_image.putpalette(_segmentation_palette())
                artifacts = {'mask':destination/f'{image_path.stem}_mask.png'}
                _atomic_png(result_image,artifacts['mask'])
                scale = min(1.,preview_max_side/max(width,height))
                size = (max(1,round(width*scale)),max(1,round(height*scale)))
                preview = _preview_mask(mask,size,nclasses)
                base = source.resize(size,Image.Resampling.BILINEAR).convert('RGBA')
                color = preview.convert('RGBA')
                color.putalpha(Image.fromarray(np.where(np.asarray(preview)==0,0,120).astype(np.uint8)))
                artifacts['overlay'] = destination/f'{image_path.stem}_overlay_preview.png'
                _atomic_png(Image.alpha_composite(base,color).convert('RGB'),artifacts['overlay'])
                for name,array in [('confidence',conf),('foreground_probability',foreground)]:
                    if array is not None:
                        artifacts[name] = destination/f'{image_path.stem}_{name}.png'
                        _atomic_png(Image.fromarray(array),artifacts[name])
                metadata = dict(image=str(image_path),width=width,height=height,source_size=[sw,sh],
                    model_input=[tile_width,tile_height],overlap=overlap,tiles=len(tiles),
                    merge=accumulator.mode,padding=padding,image_interpolation='Pillow bilinear',
                    probability_interpolation='Pillow float32 bilinear',inference_seconds=elapsed,
                    class_histogram={str(i):int((mask==i).sum()) for i in range(nclasses)},
                    artifacts={k:str(v) for k,v in artifacts.items()})
                (destination/f'{image_path.stem}_tiling.json').write_text(json.dumps(metadata,indent=2))
                for kind,path in artifacts.items():
                    writer.emit('artifact',kind=kind,image=str(image_path),path=str(path),merge=accumulator.mode)
                writer.emit('prediction',**{k:v for k,v in metadata.items() if k!='artifacts'})
        finally:
            for accumulator in accumulators:
                accumulator.close()
    return TiledImageResult(tiles=len(tiles),inference_seconds=elapsed)
