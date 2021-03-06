from io import BytesIO
from itertools import chain
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
from numpy.linalg import norm, solve
import quadpy
from scipy.linalg import eigh
from scipy.sparse import csc_matrix, eye
import splipy.surface_factory as sf
import splipy.volume_factory as vf
from splipy.io import G2
from tqdm import tqdm

from typing import Union, Iterable, Optional, List, Tuple, Union


IFEM = '/home/eivind/repos/IFEM/Apps/Elasticity/Linear/build/bin/LinEl'
INPUT = [
    'topology.xinp',
    'topologysets.xinp',
]
OUTPUT = [
    'bridge.xinp',
    'topology.xinp',
    'topologysets.xinp',
    'bridge.hdf5',
    'sol.out',
    'lhs.out',
    'rhs.out',
]


Solution = List[Tuple[lr.LRSplineObject, lr.LRSplineObject]]


def nel(span, meshwidth):
    return int(max(1, np.ceil(span / meshwidth)))

def move_meshlines(source, target):
    if isinstance(source, lr.LRSplineSurface):
        for meshline in source.meshlines:
            target.insert(meshline)
    else:
        for meshrect in source.meshrects:
            target.insert(meshrect)
    target.generate_ids()

def affine(points, left, right):
    return left + (points + 1) / 2 * (right - left)

def dictzip(**kwargs):
    for values in zip(*kwargs.values()):
        yield dict(zip(kwargs.keys(), values))

def permute_rows(test, control):
    mismatches = [
        i for i, (testrow, controlrow) in enumerate(zip(test, control))
        if not np.allclose(testrow, controlrow)
    ]
    permutation = np.arange(len(test), dtype=np.int32)
    for i in mismatches:
        permutation[i] = next(j for j in mismatches if np.allclose(test[j], control[i]))
    assert len(set(permutation)) == len(test) == len(control)
    np.testing.assert_allclose(test[permutation,:], control)
    return permutation

def load_lr(stream):
    data = stream.read()
    if isinstance(data, str):
        data = data.encode('utf-8')
    with_renum = lr.LRSplineObject.read_many(BytesIO(data), renumber=True)
    without_renum = lr.LRSplineObject.read_many(BytesIO(data), renumber=False)
    assert len(with_renum) == len(without_renum)
    if any((a.controlpoints != b.controlpoints).any() for a, b in zip(with_renum, without_renum)):
        print('Reading changed ordering!')
    return with_renum


class BridgeCase:

    # Operation
    nprocs: int
    nice: int
    dump: bool = False

    # Discretization
    order: int
    ndim: int
    meshwidth: float
    npatches: int
    ngauss: int

    # Geometry
    span: float
    diameter: float

    # Parameters
    load: float = 0.0
    load_left: float = 0.0
    load_right: float = 10.0
    load_width: float = 0.5

    maxstep: int = 10
    beta: int = 5

    with_dirichlet: bool = True
    with_neumann: bool = True

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def directory(self, name: str, index: Optional[Union[int, str]] = None) -> Path:
        if index is None:
            path = Path('data') / name
        else:
            if isinstance(index, int):
                index = f'{index:03}'
            path = Path('data') / name / index
        path.mkdir(mode=0o775, exist_ok=True, parents=True)
        return path

    def load_solution(self, path: int, step: Optional[int] = None) -> Solution:
        retval = []
        with h5py.File(path / 'bridge.hdf5', 'r') as f:
            step = len(f) - 1 if step is None else step
            group = f[f'{step}/Elasticity-1']
            npatches = len(group['basis'])

            for patchid in range(1, npatches + 1):
                geompatch = load_lr(BytesIO(group[f'basis/{patchid}'][:].tobytes()))[0]
                coeffs = group[f'fields/displacement/{patchid}'][:]
                solpatch = geompatch.clone()
                solpatch.controlpoints = coeffs.reshape(len(solpatch), -1)
                retval.append((geompatch, solpatch))
        return retval

    def setup(self):
        patchspan = self.span / self.npatches
        nel_length = nel(patchspan, self.meshwidth)
        nel_diam = nel(self.diameter, self.meshwidth)

        if self.ndim == 2:
            patch = sf.square() * (patchspan, self.diameter)
            patch.raise_order(self.order - 2)
            patch.refine(nel_length - 1, nel_diam - 1)
            patches = [patch + (i * patchspan, 0) for i in range(self.npatches)]
        elif self.ndim == 3:
            patch = vf.cube() * (patchspan, self.diameter, self.diameter)
            patch.raise_order(self.order - 2)
            patch.refine(nel_length - 1, nel_diam - 1, nel_diam - 1)
            patches = [patch + (i * patchspan, 0, 0) for i in range(self.npatches)]

        with G2('geometry.g2') as f:
            f.write(patches)

        with open('topology.xinp', 'w') as f:
            f.write("<?xml version='1.0' encoding='utf-8' standalone='no'?>\n")
            f.write('<!-- this file is autogenerated -->\n')
            f.write('<topology>\n')
            for i in range(1, self.npatches):
                f.write(f'  <connection master="{i}" slave="{i+1}" midx="2" sidx="1" orient="0" />\n')
            f.write('</topology>\n')

        interface = 'edge' if self.ndim == 2 else 'face'
        topface = 4 if self.ndim == 2 else 6
        with open('topologysets.xinp', 'w') as f:
            f.write("<?xml version='1.0' encoding='utf-8' standalone='no'?>\n")
            f.write('<!-- this file is autogenerated -->\n')
            f.write('<topologysets>\n')
            f.write(f'  <set name="support" type="{interface}">\n')
            f.write('    <item patch="1">1</item>\n')
            f.write(f'    <item patch="{self.npatches}">2</item>\n')
            f.write('  </set>\n')
            f.write(f'  <set name="surface" type="{interface}">\n')
            for i in range(1, self.npatches + 1):
                f.write(f'    <item patch="{i}">{topface}</item>\n')
            f.write('  </set>\n')
            f.write('</topologysets>\n')

    def run_ifem(self, target: Path, context: dict, geometry: Optional[Union[str, Path]] = None,
                 ignore: bool = False, rhs_only: bool = False, nprocs: int = None):
        if geometry is None:
            geometry = 'geometry.g2'
        if nprocs is None:
            nprocs = self.nprocs

        context = context.copy()
        context['geometry'] = Path(geometry).name

        with open('bridge.xinp', 'r') as f:
            template = Template(f.read())
        with tempfile.TemporaryDirectory() as tempdir_path:
            root = Path(tempdir_path)
            with open(root / 'bridge.xinp', 'w') as f:
                f.write(template.render(**context))
            for fn in INPUT + [geometry]:
                shutil.copy(fn, root)

            dim = '-2D' if self.ndim == 2 else '-3D'
            args = [IFEM, 'bridge.xinp', dim, '-hdf5', '-adap', '-cgl2']
            if nprocs > 1:
                args = ['mpirun', '-np', str(nprocs)] + args
            if self.nice != 0:
                args = ['nice', f'-n{self.nice}'] + args
            if ignore:
                args.append('-ignoresol')
            if rhs_only:
                args.append('-rhsonly')
            # print(args)
            result = run(args, cwd=root, stdout=PIPE, stderr=PIPE)
            for fn in OUTPUT + [context['geometry']]:
                if (root / fn).exists():
                    shutil.copy(root / fn, target)

        try:
            result.check_returncode()
        except:
            print('Error from IFEM:')
            print('------------------------------------------------')
            print(result.stderr.decode())
            print('------------------------------------------------')
            raise

    def run_single(self, path: Path, geometry: Optional[Union[str, Path]] = None,
                   ignore: bool = False, rhs_only: bool = False, nprocs: int = None, **kwargs):
        context = self.__class__.__dict__.copy()
        context.update(self.__dict__)
        context.update(kwargs)
        self.run_ifem(path, context, geometry, ignore=ignore, rhs_only=rhs_only, nprocs=nprocs)

    def run(self, nsols: int, **kwargs):
        quadrule = quadpy.c1.gauss_legendre(nsols)
        params = {
            'load_left': affine(quadrule.points, 0.0, self.span - self.load_width),
            'load_right': affine(quadrule.points, self.load_width, self.span),
        }

        for i, params in tqdm(enumerate(dictzip(**params)), 'Solving', total=nsols):
            self.run_single(self.directory('raw', i), **kwargs, **params)

    def merge(self, nsols: int):
        solutions = [self.load_solution(self.directory('raw', i)) for i in range(nsols)]
        rootpatches = [g.clone() for g, _ in solutions[0]]

        for sol in tqdm(solutions, 'Merging'):
            for tgt, (src, _) in zip(rootpatches, sol):
                move_meshlines(src, tgt)

        for sol in tqdm(solutions, 'Back-merging'):
            for src, (tgt1, tgt2) in zip(rootpatches, sol):
                move_meshlines(src, tgt1)
                move_meshlines(src, tgt2)

        for i, sol in tqdm(enumerate(solutions), 'Permuting', total=nsols):
            perms = [
                permute_rows(g.controlpoints, root.controlpoints)
                for root, (g, _) in zip(rootpatches, sol)
            ]

            for (root, (g, _), perm) in zip(rootpatches, sol, perms):
                np.testing.assert_allclose(root.controlpoints, g.controlpoints[perm,:])

            path = self.directory('merged', i)
            with open(path / 'geometry.lr', 'wb') as f:
                for (g, _) in sol:
                    g.write(f)
            with open(path / 'solution.lr', 'wb') as f:
                for (_, s) in sol:
                    s.write(f)
            np.savez(path / 'permutation.npz', *perms)

        path = self.directory('merged')
        with open(path / 'geometry.lr', 'wb') as f:
            for root in rootpatches:
                root.write(f)

    def fullscale(self):
        geometry = self.directory('merged') / 'geometry.lr'
        target = self.directory('merged', 'fullscale')
        context = {
            'with_dirichlet': False,
            'with_neumann': False,
            'maxstep': 0,
            'dump_matrix': True,
        }

        # Must run with one process to get dump
        self.run_ifem(target, context, geometry, nprocs=1, ignore=True)

    def extract(self, nsols: int):
        # Check that no renumbering has taken place within IFEM
        fullpatches = self.load_solution(self.directory('merged', 'fullscale'))
        with open(self.directory('merged') / 'geometry.lr', 'rb') as f:
            rootpatches = load_lr(f)
        for root, (full, _) in zip(rootpatches, fullpatches):
            np.testing.assert_allclose(root.controlpoints, full.controlpoints)

        numbering, ndofs = self.load_numbering()
        data = np.zeros((nsols, ndofs, self.ndim))
        for i in tqdm(range(nsols), 'Extracting'):
            path = self.directory('merged', i)
            with open(path / 'solution.lr', 'rb') as f:
                sol = load_lr(f)
            perms = np.load(path / 'permutation.npz')
            for n, s, perm in zip(numbering, sol, perms.values()):
                data[i,n,:] = s.controlpoints[perm,:]
        np.save(self.directory('merged') / 'snapshots.npy', data.reshape(nsols, -1))

    def load_superlu(self, directory: Path):
        with open(directory / 'lhs.out') as f:
            next(f)
            m, n, nnz = map(int, next(f).split())
            data = np.array(list(map(float, next(f).split())), dtype=float)
            n_indptr = int(next(f))
            indptr = np.array(list(map(int, next(f).split())), dtype=int)
            n_indices = int(next(f))
            indices = np.array(list(map(int, next(f).split())), dtype=int)

        assert len(indptr) == n_indptr == m+1 == n+1
        assert nnz == n_indices == len(indices)
        return csc_matrix((data, indices, indptr), shape=(m, n))

    def load_vector(self, directory: Path, filename: str):
        with open(directory / filename) as f:
            next(f)
            return np.array(list(chain.from_iterable(map(float, l.split()) for l in f)))

    def load_sol(self, directory: Path):
        return self.load_vector(directory, 'sol.out')

    def load_rhs(self, directory: Path):
        return self.load_vector(directory, 'rhs.out')

    def load_numbering(self):
        with h5py.File(self.directory('merged', 'fullscale') / 'bridge.hdf5', 'r') as f:
            group = f['0/Elasticity-1/l2g-node']
            numbering = [group[f'{i+1}'][:] - 1 for i in range(len(group))]
        ndofs = max(map(np.max, numbering)) + 1
        return numbering, ndofs

    def load_fullscale_geometry(self):
        with open(self.directory('merged') / 'geometry.lr') as f:
            return load_lr(f)

    def verify_numbering(self):
        patches = self.load_fullscale_geometry()
        numbering, _ = self.load_numbering()
        nodes = dict()

        assert len(patches) == len(numbering) == self.npatches
        for patch, numbers in zip(patches, numbering):
            controlpoints = patch.controlpoints
            assert len(numbers) == len(controlpoints)
            for node, number in zip(controlpoints, numbers):
                assert np.linalg.norm(nodes.setdefault(number, node) - node) < 1e-10

    def load_snapshots(self):
        return np.load(self.directory('merged') / 'snapshots.npy')

    def load_fullscale_superlu(self):
        return self.load_superlu(self.directory('merged', 'fullscale'))

    def project(self, nred: int):
        data = self.load_snapshots()
        hi_mass = eye(data.shape[1])

        corr = data @ hi_mass @ data.T
        eigvals, eigvecs = eigh(corr, turbo=False)
        eigvals, eigvecs = eigvals[..., ::-1], eigvecs[..., ::-1]
        np.savetxt(self.directory('reduced') / 'spectrum.csv', eigvals / eigvals[0])

        proj = data.T @ eigvecs[:, :nred] / np.sqrt(eigvals[:nred])
        np.save(self.directory('reduced', nred) / 'proj.npy', proj.T)

    def load_project(self, nred: int):
        return np.load(self.directory('reduced', nred) / 'proj.npy')

    def run_rhs(self, nsols: int, **kwargs):
        quadrule = quadpy.c1.gauss_legendre(nsols)
        params = {
            'load_left': affine(quadrule.points, 0.0, self.span - self.load_width),
            'load_right': affine(quadrule.points, self.load_width, self.span),
        }

        geometry = self.directory('merged') / 'geometry.lr'
        for i, params in tqdm(enumerate(dictzip(**params)), 'Integrating', total=nsols):
            self.run_single(self.directory('rhs', i), geometry, **kwargs, **params,
                            with_dirichlet=False, ignoresol=True, rhs_only=True, dump_rhs=True, nprocs=1)

    def compare(self, nsols: int, nred: int):
        hi_lhs = self.load_fullscale_superlu()
        proj = self.load_project(nred)
        lo_lhs = proj @ hi_lhs @ proj.T
        data = self.load_snapshots()
        hi_mass = eye(data.shape[1])

        errors = []
        for i in tqdm(range(nsols), 'Comparing'):
            hi_sol = data[i]
            hi_rhs = self.load_rhs(self.directory('rhs', i))
            rc_sol = solve(lo_lhs, proj @ hi_rhs) @ proj
            diff = hi_sol - rc_sol
            error = np.sqrt(diff.T @ hi_mass @ diff) / np.sqrt(hi_sol.T @ hi_mass @ hi_sol)
            errors.append(error)

        print(f'Average error: {np.mean(errors):.2e}')
        print(f'Maximal error: {np.max(errors):.2e}')


@click.command()
@click.option('--order', '-o', default=2)
@click.option('--ndim', '-n', default=2)
@click.option('--meshwidth', '-h', default=1.0)
@click.option('--npatches', '-p', default=1)
@click.option('--span', '-s', default=10.0)
@click.option('--diameter', '-d', default=1.0)
@click.option('--load', '-l', default=1e6)
@click.option('--maxstep', default=10)
@click.option('--beta', default=5)
@click.option('--nsols', default=10)
@click.option('--nred', default=5)
@click.option('--nprocs', default=1)
@click.option('--nice', default=0)
@click.option('--ngauss', default=3)
def main(nsols: int, nred: int, **kwargs):
    case = BridgeCase(**kwargs)
    case.setup()
    case.run(nsols)
    case.merge(nsols)
    case.fullscale()
    case.extract(nsols)
    case.verify_numbering()
    case.project(nred)
    case.run_rhs(nsols)
    case.compare(nsols, nred)


if __name__ == '__main__':
    main()
