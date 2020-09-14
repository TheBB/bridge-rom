import os
from pathlib import Path
import shutil
from subprocess import run, PIPE
import tempfile

import click
import h5py
from jinja2 import Template
import lrspline as lr
import numpy as np
import splipy.surface_factory as sf
from splipy.io import G2
from tqdm import tqdm

from typing import Union, Iterable, Optional, List, Tuple


IFEM = '/home/eivind/repos/IFEM/Apps/Elasticity/Linear/build/bin/LinEl'
INPUT = [
    'geometry.g2',
    'topology.xinp',
    'topologysets.xinp',
]
OUTPUT = [
    'bridge.hdf5',
    'bridge.xinp',
    'sol.out',
    'lhs.out',
    'rhs.out',
]


Solution = List[Tuple[lr.LRSplineObject, lr.LRSplineObject]]


def nel(span, meshwidth):
    return int(max(1, np.ceil(span / meshwidth)))

def read_lr(data):
    if data.startswith(b'# LRSPLINE SURFACE'):
        return lr.LRSplineSurface(data)
    return lr.LRSplineVolume(data)

def move_meshlines(source, target):
    if isinstance(source, lr.LRSplineSurface):
        for meshline in source.meshlines:
            target.insert(meshline)
    else:
        for meshrect in source.meshrects:
            target.insert(meshrect)
    target.generate_ids()


class BridgeCase:

    # Discretization
    order: int
    ndim: int
    meshwidth: float
    npatches: int

    # Geometry
    span: float
    diameter: float

    # Parameters
    load: float = 0.0
    load_left: float = 10.0
    load_right: float = 0.0

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def directory(self, name: str, index: Union[int, str]) -> Path:
        if isinstance(index, int):
            index = f'{index:03}'
        path = Path('data') / name / index
        path.mkdir(mode=0o775, exist_ok=True, parents=True)
        return path

    def load_solution(self, index: int, step: Optional[int] = None) -> Solution:
        retval = []
        path = self.directory('raw', index)
        with h5py.File(path / 'bridge.hdf5', 'r') as f:
            step = len(f) - 1 if step is None else step
            group = f[f'{step}/Elasticity-1']
            npatches = len(group['basis'])

            for patchid in range(1, npatches + 1):
                geompatch = read_lr(group[f'basis/{patchid}'][:].tobytes())
                coeffs = group[f'fields/displacement/{patchid}'][:]
                solpatch = geompatch.clone()
                solpatch.controlpoints = coeffs.reshape(len(solpatch), -1)
                retval.append((geompatch, solpatch))
        return retval

    def setup(self):
        patchspan = self.span / self.npatches
        nel_length = nel(patchspan, self.meshwidth)
        nel_diam = nel(self.diameter, self.meshwidth)

        patch = sf.square() * (patchspan, self.diameter)
        patch.raise_order(self.order - 2, self.order - 2)
        patch.refine(nel_length - 1, nel_diam - 1)
        patches = [patch + (i * patchspan, 0) for i in range(self.npatches)]

        with G2('geometry.g2') as f:
            f.write(patches)

        with open('topology.xinp', 'w') as f:
            f.write("<?xml version='1.0' encoding='utf-8' standalone='no'?>\n")
            f.write('<topology>\n')
            for i in range(1, self.npatches):
                f.write(f'  <connection master="{i}" slave="{i+1}" midx="2" sidx="1" orient="0"/>\n')
            f.write('</topology>\n')

        with open('topologysets.xinp', 'w') as f:
            f.write("<?xml version='1.0' encoding='utf-8' standalone='no'?>\n")
            f.write('<topologysets>\n')
            f.write('  <set name="support" type="edge">\n')
            f.write('    <item patch="1">1</item>\n')
            f.write(f'    <item patch="{self.npatches}">2</item>\n')
            f.write('  </set>\n')
            f.write('  <set name="surface" type="edge">\n')
            for i in range(1, self.npatches + 1):
                f.write(f'<item patch="{i}">4</item>\n')
            f.write('  </set>\n')
            f.write('</topologysets>\n')

    def run(self, index: int, **kwargs):
        context = self.__dict__.copy()
        context.update(kwargs)

        with open('bridge.xinp', 'r') as f:
            template = Template(f.read())
        with tempfile.TemporaryDirectory() as tempdir_path:
            root = Path(tempdir_path)
            with open(root / 'bridge.xinp', 'w') as f:
                f.write(template.render(**context))
            for fn in INPUT:
                shutil.copy(fn, root)

            result = run([IFEM, 'bridge.xinp', '-2D', '-hdf5', '-adap', '-cgl2'], cwd=root, stdout=PIPE, stderr=PIPE)
            try:
                result.check_returncode()
            except:
                print(result.stderr)
                raise

            target = self.directory('raw', index)
            for fn in OUTPUT:
                shutil.copy(root / fn, target)

    def merge(self, nsols: int):
        solutions = [self.load_solution(i) for i in range(nsols)]
        rootpatches = [g.clone() for g, _ in solutions[0]]

        for sol in tqdm(solutions, 'Merging'):
            for tgt, (src, _) in zip(rootpatches, sol):
                move_meshlines(src, tgt)

        for sol in tqdm(solutions, 'Back-merging'):
            for src, (tgt1, tgt2) in zip(rootpatches, sol):
                move_meshlines(src, tgt1)
                move_meshlines(src, tgt2)

        for sol in solutions:
            for root, (g, _) in zip(rootpatches, sol):
                np.testing.assert_allclose(root.controlpoints, g.controlpoints)


@click.command()
@click.option('--order', '-o', default=2)
@click.option('--ndim', '-n', default=2)
@click.option('--meshwidth', '-h', default=1.0)
@click.option('--npatches', '-p', default=1)
@click.option('--span', '-s', default=10.0)
@click.option('--diameter', '-d', default=1.0)
@click.option('--load', '-l', default=1e6)
def main(**kwargs):
    case = BridgeCase(**kwargs)
    case.setup()
    case.run(0, load_left=2.0, load_right=3.0)
    case.run(1, load_left=7.0, load_right=8.0)
    case.merge(2)


if __name__ == '__main__':
    main()
