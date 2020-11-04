from io import BytesIO
from itertools import chain
from pathlib import Path
import shutil
from subprocess import run, PIPE
import tempfile
import os

from typing import Optional, List, Tuple, Union

import click
import h5py
from jinja2 import Template
import lrspline as lr
import numpy as np
from numpy.linalg import solve
import quadpy
from scipy.linalg import eigh
from scipy.sparse import csc_matrix, eye
from tqdm import tqdm


IFEM = '/home/eivindf/repos/IFEM/Apps/Elasticity/Linear/build/bin/LinEl'
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

def permute_rows(test, control, atol=1e-10):
    mismatches = [
        i for i, (testrow, controlrow) in enumerate(zip(test, control))
        if not np.allclose(testrow, controlrow, atol=atol)
    ]
    permutation = np.arange(len(test), dtype=np.int32)
    for i in mismatches:
        permutation[i] = next(j for j in mismatches if np.allclose(test[j], control[i], atol=atol))
    assert len(set(permutation)) == len(test) == len(control)
    np.testing.assert_allclose(test[permutation,:], control, atol=atol)
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
    nthreads: int
    nice: int
    dump: bool = False
    datadir: Path = Path('data')

    # Discretization
    ndim: int = 3
    ngauss: int

    # Parameters
    load: float = 2.7e6 / 0.15 / 1.2 / 2 / 18
    load_center: float = 0.0

    maxstep: int = 10
    beta: int = 5

    with_dirichlet: bool = True
    with_neumann: bool = True

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def directory(self, name: str, index: Optional[Union[int, str]] = None) -> Path:
        if index is None:
            path = self.datadir / name
        else:
            if isinstance(index, int):
                index = f'{index:03}'
            path = self.datadir / name / index
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
        pass

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

            env = os.environ.copy()
            env.update({
                'OMP_NUM_THREADS': str(self.nthreads),
            })

            dim = '-2D' if self.ndim == 2 else '-3D'
            args = [IFEM, 'bridge.xinp', dim, '-hdf5', '-adap', '-cgl2', '-petsc']
            if nprocs > 1:
                args = ['mpirun', '-np', str(nprocs)] + args
            if self.nice != 0:
                args = ['nice', f'-n{self.nice}'] + args
            if ignore:
                args.append('-ignoresol')
            if rhs_only:
                args.append('-rhsonly')
            with open(target / 'invocation.txt', 'w') as f:
                f.write(' '.join(args))
            result = run(args, cwd=root, stdout=PIPE, stderr=PIPE, env=env)
            for fn in OUTPUT + [context['geometry']]:
                if (root / fn).exists():
                    shutil.copy(root / fn, target)
            with open(target / 'stdout.txt', 'wb') as f:
                f.write(result.stdout)
            with open(target / 'stderr.txt', 'wb') as f:
                f.write(result.stderr)

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
            'load_center': affine(quadrule.points, -42.175, 152.175),
        }

        objs = list(enumerate(dictzip(**params)))
        for i, params in tqdm(objs, 'Solving', total=len(objs)):
            self.run_single(self.directory('raw', i), **kwargs, **params)

    def merge(self, nsols: int):
        solutions = [self.load_solution(self.directory('raw', i), step=3) for i in range(nsols)]
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
                np.testing.assert_allclose(root.controlpoints, g.controlpoints[perm,:], atol=1e-10)

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
            'ngauss': 10,
            'dump_matrix': True,
        }

        # Must run with one process to get dump
        self.run_ifem(target, context, geometry, nprocs=1, ignore=True)

    def run_fullscale(self, nsols: int):
        geometry = self.directory('merged') / 'geometry.lr'
        quadrule = quadpy.c1.gauss_legendre(nsols)
        params = {
            'load_center': affine(quadrule.points, -42.175, 152.175),
        }

        objs = list(enumerate(dictzip(**params)))
        objs = objs[:1]

        for i, params in tqdm(objs, 'Solving', total=len(objs)):
            self.run_single(self.directory('debug', i), geometry, maxstep=0, ngauss=10, dump_lhs=True, dump_rhs=True, nprocs=1, **params)

    def extract(self, nsols: int):
        # Check that no renumbering has taken place within IFEM
        fullpatches = self.load_solution(self.directory('merged', 'fullscale'))
        with open(self.directory('merged') / 'geometry.lr', 'rb') as f:
            rootpatches = load_lr(f)
        for root, (full, _) in zip(rootpatches, fullpatches):
            np.testing.assert_allclose(root.controlpoints, full.controlpoints, atol=1e-12)

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

    def load_numbering(self, path: Path = None):
        if path is None:
            path = self.directory('merged', 'fullscale')
        with h5py.File(path / 'bridge.hdf5', 'r') as f:
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

        assert len(patches) == len(numbering)
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
            'load_center': affine(quadrule.points, -42.175, 152.175),
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
@click.option('--ndim', '-n', default=2)
@click.option('--maxstep', default=10)
@click.option('--beta', default=5)
@click.option('--nsols', default=10)
@click.option('--nred', default=5)
@click.option('--nprocs', default=1)
@click.option('--nthreads', default=1)
@click.option('--nice', default=0)
@click.option('--ngauss', default=10)
def main(nsols: int, nred: int, **kwargs):
    case = BridgeCase(**kwargs, datadir=Path('data-linear'))
    case.setup()
    case.run(nsols)
    # case.merge(nsols)
    # case.run_fullscale(nsols)
    # case.fullscale()
    # case.extract(nsols)
    # case.verify_numbering()
    # case.project(nred)
    # case.run_rhs(nsols)
    # case.compare(nsols, nred)


if __name__ == '__main__':
    main()
