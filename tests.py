"""Numerical and scorer checks for the source-only reproduction package."""

# test_fast_feature_congestion.py
"""Verify alignment, edge behavior and restoration of the accelerated scorer."""
import unittest
import numpy as np
import visual_clutter.utils as u
from numerical_andrews.fast_feature_congestion import accelerated_fc

class FastConvolutionTest(unittest.TestCase):
    def test_asymmetric_odd_kernels_and_boundaries(self):
        rng=np.random.default_rng(73)
        original=u.conv2
        for shape in [(9,13),(16,18)]:
            array=rng.normal(size=shape)
            for kernel in [np.array([[1.,-2.,4.]]),np.array([[1.],[-2.],[4.]])]:
                expected=original(array,kernel,'same')
                with accelerated_fc(): actual=u.conv2(array,kernel,'same')
                np.testing.assert_allclose(actual,expected,atol=1e-14)
        self.assertIs(u.conv2,original)
    def test_general_kernel_fallback(self):
        original=u.conv2
        a=np.arange(30.).reshape(5,6); kernel=np.arange(6.).reshape(2,3)
        expected=original(a,kernel,'same')
        with accelerated_fc(): actual=u.conv2(a,kernel,'same')
        np.testing.assert_array_equal(actual,expected)


# test_fc_reference.py
"""Independent source-level oracles for the MIT June 2007 preprocessing.

These are scalar/explicit-padding Python translations, not MATLAB executions.
Source hashes and precise verification scope are in fc_reference_provenance.json.
"""
import itertools,math,unittest
import numpy as np
from scipy.signal import convolve2d
from numerical_andrews.congestion import official_rgb_to_lab,official_gaussian_pyramid


def scalar_reference(pixel):
    channels=[]
    for value in pixel:
        v=float(value)/255
        channels.append(((v+.055)/1.055)**2.4 if v>.04045 else v/12.92)
    r,g,b=channels
    xyz=[(.4124*r+.3576*g+.1805*b)/95.047,
         (.2126*r+.7152*g+.0722*b)/100.,
         (.0193*r+.1192*g+.9505*b)/108.883]
    f=[v**(1/3) if v>.008856 else 7.787*v+16/116 for v in xyz]
    return [116*f[1]-16,500*(f[0]-f[1]),200*(f[1]-f[2])]


def source_reduce(image):
    # MATLAB filt2 adds kernel-size-minus-one reflected border, then crops
    # conv2(...,'same'); reduce subsamples columns before filtering rows.
    kernel=np.array([[.05,.25,.4,.25,.05]])
    padded=np.pad(image,((0,0),(4,4)),mode='reflect')
    horizontal=convolve2d(padded,kernel,mode='same')[:,4:-4][:,::2]
    padded=np.pad(horizontal,((4,4),(0,0)),mode='reflect')
    return convolve2d(padded,kernel.T,mode='same')[4:-4,:][::2,:]

class ReferencePreprocessingTests(unittest.TestCase):
    def test_rgb_against_scalar_matlab_source_translation(self):
        # Both sides of the uint8 gamma threshold, dark/bright and saturated RGB.
        values=[0,1,10,11,64,128,240,255]
        pixels=np.array(list(itertools.product(values,repeat=3)),dtype=np.uint8)
        expected=np.array([scalar_reference(p) for p in pixels]).reshape(16,32,3)
        np.testing.assert_allclose(official_rgb_to_lab(pixels.reshape(16,32,3)),expected,rtol=0,atol=1e-12)
    def test_reference_white_scaling_is_preserved(self):
        white=official_rgb_to_lab(np.full((1,1,3),255,dtype=np.uint8))[0,0]
        self.assertAlmostEqual(white[0],116*math.pow(.01,1/3)-16,places=12)
        self.assertLess(white[0],10)  # Historical source scaling, not L*=100.
    def test_pyramid_edges_sampling_and_odd_even_sizes(self):
        rng=np.random.default_rng(4)
        for shape in [(17,20),(19,21)]:
            impulse=np.zeros(shape);impulse[0,0]=1;impulse[-1,-1]=2
            for array in [np.ones(shape),impulse,rng.normal(size=shape)]:
                actual=official_gaussian_pyramid(array,3);reference=array.copy()
                for level in range(3):
                    np.testing.assert_allclose(actual[level,0],reference,rtol=0,atol=1e-14)
                    if level<2:reference=source_reduce(reference)


# test_mmqv_family.py
import unittest
import numpy as np
from numerical_andrews.experiments.optimize_mmqv_feature_congestion import mmqv_transform, family_envelope
from numerical_andrews.core import evaluate_mmqv_modes, evaluate_limiting_aligned_mmqv_modes


class MMQVFamilyTests(unittest.TestCase):
    def test_isometry_and_unchanged_derivative_energy(self):
        rng = np.random.default_rng(18)
        for d in (4, 5, 10, 30):
            for _ in range(10):
                transform = mmqv_transform(d, rng.uniform(0,2*np.pi,d//2),
                                          rng.choice([-1,1],(d-1)//2),rng.choice([-1,1]))
                frequencies = np.array([(j+1)//2 for j in range(transform.shape[0])])
                np.testing.assert_allclose(transform.T@transform,np.eye(d),atol=1e-14)
                energy = transform.T@(frequencies[:,None]**2*transform)
                np.testing.assert_allclose(energy,np.diag(frequencies[:d]**2),atol=1e-12)

    def test_both_named_bases_are_in_search_family(self):
        points = np.linspace(-0.5,0.5,257)
        for d in (4,5,10,30):
            ambient = d+(d%2==0)
            standard = evaluate_mmqv_modes(ambient,points)
            c = mmqv_transform(d,np.zeros(d//2),np.ones((d-1)//2),1)
            s = mmqv_transform(d,np.full(d//2,np.pi/2),-np.ones((d-1)//2),1)
            np.testing.assert_allclose(standard@c,evaluate_mmqv_modes(d,points),atol=1e-14)
            np.testing.assert_allclose(standard@s,evaluate_limiting_aligned_mmqv_modes(d,points),atol=1e-14)

    def test_uniform_amplitude_bound_including_unpaired_mode(self):
        rng=np.random.default_rng(7)
        for d in (4,5,30):
            scores=rng.normal(size=(8,d))
            points=np.linspace(-.5,.5,4097)
            trig=evaluate_mmqv_modes(d+(d%2==0),points)
            for _ in range(10):
                transform=mmqv_transform(d,rng.uniform(0,2*np.pi,d//2),
                                        rng.choice([-1,1],(d-1)//2),rng.choice([-1,1]))
                self.assertLessEqual(np.max(np.abs(scores@transform.T@trig.T)),family_envelope(scores)+1e-12)



# test_spectral_ordering.py
"""Regression checks for the strict-interlacing selection rule."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import eigh

import numerical_andrews.core as plots


class SpectralOrderingTests(unittest.TestCase):
    def test_smallest_sections_against_dense_spectrum(self):
        # An independent full-matrix solve checks both the cutoff boundary and
        # the selected spectrum, including the unmatched final even mode.
        for cutoff in range(1, 6):
            even = np.diag(0.2 * np.arange(cutoff + 1)**2 + 2.0)
            for j in range(cutoff):
                even[j, j + 1] = even[j + 1, j] = -np.sqrt(2) if j == 0 else -1
            odd = even[1:, 1:]
            expected = np.sort(np.r_[eigh(even, eigvals_only=True),
                                      eigh(odd, eigvals_only=True)])
            for count in range(1, 2 * cutoff + 2):
                with self.subTest(cutoff=cutoff, count=count):
                    modes, _ = plots.spatial_spectral_modes(0.2, cutoff, count)
                    np.testing.assert_allclose(
                        [mode.eigenvalue for mode in modes], expected[:count], atol=1e-12
                    )

    def test_cross_parity_roundoff_does_not_swap_modes(self):
        # Simulate a rounding inversion of a nearly coincident O1/E2 pair.
        even = [plots.Mode('even', 0, 1.0, np.array([1.0]), 0, 0),
                plots.Mode('even', 1, 10.0, np.array([1.0]), 0, 0)]
        odd = [plots.Mode('odd', 0, np.nextafter(10.0, np.inf),
                          np.array([1.0]), 0, 0)]
        with patch.object(plots, '_sector_modes', side_effect=[even, odd]):
            modes, _ = plots.spatial_spectral_modes(100.0, 1, 3)
        self.assertIs(modes[0], even[0])
        self.assertIs(modes[1], odd[0])
        self.assertIs(modes[2], even[1])

    def test_large_alpha_keeps_odd_before_next_even(self):
        for alpha in (10.0, 100.0, 1e4):
            modes, _ = plots.spatial_spectral_modes(alpha, 32, 30)
            for j in range(1, len(modes) - 1, 2):
                self.assertEqual((modes[j].parity, modes[j + 1].parity), ('odd', 'even'))
                self.assertEqual(modes[j + 1].sector_rank, modes[j].sector_rank + 1)

    def test_insufficient_section_is_rejected(self):
        with self.assertRaises(ValueError):
            plots.spatial_spectral_modes(0.2, 1, 4)

    def test_minimal_requests_and_diagnostic_neighbors(self):
        for count in (1, 4, 5, 10, 30):
            modes, sectors = plots.spatial_spectral_modes(0.2, 32, count)
            self.assertEqual(len(sectors['even']), (count + 1) // 2)
            self.assertEqual(len(sectors['odd']), count // 2)
            expanded, neighbors = plots.spatial_spectral_modes(
                0.2, 32, count, extra_per_sector=3
            )
            self.assertEqual(len(neighbors['even']), (count + 1) // 2 + 3)
            self.assertEqual(len(neighbors['odd']), count // 2 + 3)
            for mode, reference in zip(modes, expanded):
                np.testing.assert_allclose(mode.coefficients, reference.coefficients,
                                           atol=1e-12, rtol=1e-12)



# test_standardized_geometry.py
"""Check the geometry preserved by the standardized companion experiment."""
import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer
from numerical_andrews.core import load_example_datasets

class StandardizedGeometryTest(unittest.TestCase):
    def test_standardized_pca_preserves_variance_weighted_distances(self):
        original=load_breast_cancer()
        datasets=load_example_datasets(include_standardized=True)
        raw=next(d for d in datasets if d.key=='bc')
        standardized=next(d for d in datasets if d.key=='bc_standardized')
        np.testing.assert_array_equal(raw.labels,standardized.labels)
        np.testing.assert_allclose(standardized.values.mean(axis=0),0,atol=1e-14)
        np.testing.assert_allclose(standardized.values.std(axis=0),1,atol=1e-14)
        scale=original.data.std(axis=0)
        for i,j in [(0,568),(1,200),(100,300),(43,71)]:
            expected=np.sum(((original.data[i]-original.data[j])/scale)**2)
            measured=np.sum((standardized.scores[i]-standardized.scores[j])**2)
            self.assertAlmostEqual(measured,expected,places=10)
        self.assertGreater(raw.singular_values[0]**2/np.sum(raw.singular_values**2),.98)
        self.assertLess(standardized.singular_values[0]**2/np.sum(standardized.singular_values**2),.45)


# test_truncation_certificate.py
import unittest
from fractions import Fraction as F
import numpy as np
from scipy.linalg import eigh_tridiagonal
from numerical_andrews.experiments.certify_truncation import (matrix_data, eigen_bracket, sturm_count,
                                certify, rayleigh_residual, rational_candidate)


class CertificationTests(unittest.TestCase):
    def test_sturm_zero_pivots_and_exact_roots(self):
        # Eigenvalues -sqrt(2), 0, sqrt(2); internal and final zero pivots.
        diagonal, off = [F(0)]*3, [F(1)]*2
        self.assertEqual(sturm_count(diagonal, off, F(0)), 1)
        for x, expected in [(-2, 0), (-1, 1), (1, 2), (2, 3)]:
            self.assertEqual(sturm_count(diagonal, off, F(x)), expected)
        lo, hi = eigen_bracket(diagonal, off, 2)
        self.assertLessEqual(lo, 0)
        self.assertGreaterEqual(hi, 0)
        lo, hi = eigen_bracket(diagonal, off, 3)
        self.assertLessEqual(lo*lo, 2)
        self.assertGreaterEqual(hi*hi, 2)

    def test_certificates_and_larger_sections(self):
        for alpha, size in [('0.001', 96), ('0.2', 24), ('10', 12)]:
            for sector in ('even', 'odd'):
                for index in (1, 3):
                    with self.subTest(alpha=alpha, sector=sector, index=index):
                        result = certify(alpha, sector, size, index)
                        lo, hi = result['eigenvalue_interval']
                        self.assertLessEqual(lo, hi)
                        self.assertLess(hi-lo, F(1, 10**20))
                        self.assertLess(result['eigenvector_error_squared_upper'], F(1, 10**20))
                        # Independent floating-point diagnostic, not certification.
                        d, q = matrix_data(alpha, sector, size*2)
                        eig = eigh_tridiagonal(np.array(d, dtype=float),
                              -np.sqrt(np.array(q, dtype=float)),
                              select='i', select_range=(index-1,index-1))[0][0]
                        self.assertAlmostEqual(float((lo+hi)/2), eig, delta=1e-10)

    def test_full_residual_matches_symmetric_coordinates(self):
        for sector in ('even', 'odd'):
            d, q = matrix_data('0.2', sector, 8)
            z = [F(i+1, 9) for i in range(8)]
            rho, r2, internal, boundary = rayleigh_residual(d, sector, z)
            w = np.array(z, dtype=float)
            if sector == 'even':
                w[1:] *= np.sqrt(2)
            a = np.diag(np.array(d, dtype=float))
            off = -np.sqrt(np.array(q, dtype=float))
            a += np.diag(off, 1) + np.diag(off, -1)
            rho_float = w@a@w/(w@w)
            r2_float = (np.linalg.norm(a@w-rho_float*w)**2+w[-1]**2)/(w@w)
            self.assertAlmostEqual(float(rho), rho_float)
            self.assertAlmostEqual(float(r2), r2_float)
            self.assertEqual(r2, internal+boundary)

    def test_failed_tests_and_perturbed_candidate(self):
        with self.assertRaisesRegex(ValueError, 'Tail bound'):
            certify('0.001', 'even', 3, 1)
        with self.assertRaisesRegex(ValueError, 'Residual/separation'):
            certify('0.2', 'odd', 24, 3, candidate=[1]+[0]*23)
        d, q = matrix_data('0.2', 'even', 24)
        z = rational_candidate(d, q, 'even', 3)
        baseline = certify('0.2', 'even', 24, 3, candidate=z)
        z[0] += F(1, 10**6)
        perturbed = certify('0.2', 'even', 24, 3, candidate=z)
        self.assertGreater(perturbed['internal_residual_squared'], baseline['internal_residual_squared'])
        self.assertGreater(perturbed['eigenvector_error_squared_upper'], baseline['eigenvector_error_squared_upper'])



# test_union_fc_acceleration.py
"""Boundary alignment and scoped restoration for union-search acceleration."""
import unittest
import numpy as np
import visual_clutter.utils as u
from numerical_andrews.union_fc_acceleration import union_accelerated_fc

class UnionAccelerationTests(unittest.TestCase):
    def test_fft_alignment_with_asymmetric_even_and_odd_filters(self):
        rng=np.random.default_rng(912)
        image=rng.normal(size=(73,79));original=u.conv2
        for shape in [(4,6),(7,5)]:
            kernel=rng.normal(size=shape)
            for mode in ['same',None]:
                expected=original(image,kernel,mode)
                with union_accelerated_fc(): actual=u.conv2(image,kernel,mode)
                np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
                self.assertIs(u.conv2,original)
    def test_overlap_cache_does_not_depend_on_input_values(self):
        rng=np.random.default_rng(38);kernel=np.array([[.1,.2,.4,.2,.1]])
        original=u.RRoverlapconv
        for image in [rng.normal(size=(73,79)),np.ones((73,79)),np.zeros((73,79))]:
            expected=original(kernel,image)
            with union_accelerated_fc():actual=u.RRoverlapconv(kernel,image)
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
            self.assertIs(u.RRoverlapconv,original)

class SkeletonWorkflowTests(unittest.TestCase):
    def test_named_archive_survives_later_profile_replacement(self):
        import tempfile
        from pathlib import Path
        from numerical_andrews.__main__ import archive_named
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'feature_congestion_alpha_profile.csv'
            source.write_text('fresh named-basis scores')
            archive_named(root)
            source.write_text('later optimized scores')
            self.assertEqual((root / 'feature_congestion_two_baseline_profile.csv').read_text(),
                             'fresh named-basis scores')

    def test_initial_archives_accumulate_without_losing_raw_searches(self):
        import tempfile
        from pathlib import Path
        from numerical_andrews.__main__ import archive_initial
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'optimized_mmqv'
            source.mkdir()
            for key in ('iris', 'bc_standardized'):
                for suffix in ('.json', '_best.png', '_history.csv'):
                    (source / (key + suffix)).write_text(key)
                archive_initial(root, [key])
            for key in ('iris', 'bc_standardized'):
                for suffix in ('.json', '_best.png', '_history.csv'):
                    self.assertEqual((root / 'optimized_mmqv_initial' / (key + suffix)).read_text(), key)

    def test_every_workflow_driver_is_in_the_skeleton(self):
        import importlib
        from numerical_andrews.__main__ import plan
        for _, _, module, _ in plan(1):
            if not module.startswith('@'):
                with self.subTest(module=module):
                    importlib.import_module('numerical_andrews.experiments.' + module)

    def test_plan_is_read_only_and_runs_outside_repository(self):
        import os
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        import numerical_andrews
        with tempfile.TemporaryDirectory() as folder:
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                PYTHONPATH=str(Path(numerical_andrews.__file__).resolve().parent.parent))
            result = subprocess.run([sys.executable, '-m', 'numerical_andrews', '--plan'],
                cwd=folder, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('archive_standardized', result.stdout)
            self.assertIn('fitted_searches', result.stdout)
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
