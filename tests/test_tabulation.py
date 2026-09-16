"""Tests for the tabulation and the file it is written to.

The layout is a contract with Galacticus's ``atlas_compendium.F90``, which
checks the shape of every array it reads against the axes. Those checks are
restated here in the Fortran's own terms, so that a change to the layout fails
against them rather than against a downstream run.
"""

import h5py
import numpy as np
import pytest

from dustcompendium.tabulation import Tabulation, axis_dataset_name, read_tabulation

WAVELENGTHS = np.array([0.3, 0.44, 0.55, 1.0])
INCLINATIONS = np.array([0.0, 45.0, 90.0])
DEPTHS = np.array([0.0, 0.1, 1.0, 10.0, 100.0])
RADII = np.array([0.1, 1.0])
SPHEROID_DEPTHS = np.array([0.0, 1.0, 10.0])


def build(spheroid_dust=False):
    """A tabulation with the published axes, or with a spheroid dust axis added."""
    axis_names = ["opticalDepth:disk"]
    axis_values = {"opticalDepth:disk": DEPTHS, "scaleRadial:spheroid": RADII}
    if spheroid_dust:
        axis_names.append("opticalDepth:spheroid")
        axis_values["opticalDepth:spheroid"] = SPHEROID_DEPTHS
    axis_names.append("scaleRadial:spheroid")
    emitter_axes = {
        "disk": tuple(name for name in axis_names if name != "scaleRadial:spheroid")
        if not spheroid_dust
        else tuple(axis_names),
        "spheroid": tuple(axis_names),
    }
    shapes = {
        emitter: (
            WAVELENGTHS.size,
            INCLINATIONS.size,
            *(axis_values[name].size for name in axes),
        )
        for emitter, axes in emitter_axes.items()
    }
    rng = np.random.default_rng(7)
    attenuation = {emitter: rng.uniform(0.1, 1.0, shape) for emitter, shape in shapes.items()}
    uncertainty = {emitter: value * 0.01 for emitter, value in attenuation.items()}
    extrapolation, residual = {}, {}
    for emitter, axes in emitter_axes.items():
        trailing = tuple(axis_values[name].size for name in axes if name != "opticalDepth:disk")
        extrapolation[emitter] = rng.normal(size=(2, WAVELENGTHS.size, INCLINATIONS.size, *trailing))
        residual[emitter] = rng.uniform(0.0, 0.1, (WAVELENGTHS.size, INCLINATIONS.size, *trailing))
    return Tabulation(
        label="test",
        wavelengths=WAVELENGTHS,
        inclinations=INCLINATIONS,
        axis_names=tuple(axis_names),
        axis_values=axis_values,
        emitter_axes=emitter_axes,
        attenuation=attenuation,
        uncertainty=uncertainty,
        extrapolation=extrapolation,
        residual=residual,
        opacity=250.0,
        metadata={"dustDescription": "test grains"},
    )


class TestAxisNames:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("opticalDepth:disk", "opticalDepth"),
            ("opticalDepth:spheroid", "spheroidOpticalDepth"),
            ("scaleRadial:spheroid", "spheroidScaleRadial"),
        ],
    )
    def test_the_published_names_are_kept(self, name, expected):
        """These are not a pattern, so they are a table."""
        assert axis_dataset_name(name) == expected

    def test_anything_else_is_derived(self):
        assert axis_dataset_name("scaleHeight:disk") == "diskScaleHeight"


class TestFormatVersion:
    def test_the_published_axes_are_version_one(self):
        assert build().format_version == 1

    def test_an_extra_axis_makes_it_version_two(self):
        assert build(spheroid_dust=True).format_version == 2


class TestGalacticusContract:
    """The shapes ``atlas_compendium.F90`` checks, in the order it checks them.

    Fortran indexes in the reverse of C, so an array stored as
    ``[wavelength, inclination, opticalDepth]`` is read as
    ``[opticalDepth, inclination, wavelength]``. Each assertion below is the
    Fortran check written out.
    """

    @pytest.fixture
    def written(self, tmp_path):
        path = tmp_path / "attenuations.hdf5"
        build().write(str(path))
        return path

    def test_the_file_is_recognised_by_its_opacity_attribute(self, written):
        with h5py.File(written, "r") as handle:
            assert "opacity" in handle.attrs

    def test_every_dataset_it_reads_is_present(self, written):
        required = {
            "wavelength",
            "inclination",
            "opticalDepth",
            "spheroidScaleRadial",
            "attenuationDisk",
            "attenuationSpheroid",
            "extrapolationCoefficientsDisk",
            "extrapolationCoefficientsSpheroid",
        }
        with h5py.File(written, "r") as handle:
            assert required <= set(handle)

    def test_attenuation_disk_has_the_shape_the_axes_imply(self, written):
        with h5py.File(written, "r") as handle:
            fortran = handle["attenuationDisk"].shape[::-1]
            assert fortran == (DEPTHS.size, INCLINATIONS.size, WAVELENGTHS.size)

    def test_attenuation_spheroid_has_the_shape_the_axes_imply(self, written):
        with h5py.File(written, "r") as handle:
            fortran = handle["attenuationSpheroid"].shape[::-1]
            assert fortran == (
                RADII.size,
                DEPTHS.size,
                INCLINATIONS.size,
                WAVELENGTHS.size,
            )

    def test_extrapolation_disk_has_the_shape_the_axes_imply(self, written):
        with h5py.File(written, "r") as handle:
            fortran = handle["extrapolationCoefficientsDisk"].shape[::-1]
            assert fortran == (INCLINATIONS.size, WAVELENGTHS.size, 2)

    def test_extrapolation_spheroid_has_the_shape_the_axes_imply(self, written):
        with h5py.File(written, "r") as handle:
            fortran = handle["extrapolationCoefficientsSpheroid"].shape[::-1]
            assert fortran == (RADII.size, INCLINATIONS.size, WAVELENGTHS.size, 2)

    def test_the_coefficient_axis_is_constant_then_logarithmic(self, written):
        """Galacticus splits it as (...,1) constant and (...,2) logarithmic."""
        with h5py.File(written, "r") as handle:
            dataset = handle["extrapolationCoefficientsDisk"]
            assert next(iter(dataset.attrs["axes"])) == "coefficient"
            assert dataset.shape[0] == 2

    def test_a_version_two_file_would_be_rejected_by_the_shape_checks(self, tmp_path):
        """Loudly wrong is the outcome wanted; quietly misread is not."""
        path = tmp_path / "v2.hdf5"
        build(spheroid_dust=True).write(str(path))
        with h5py.File(path, "r") as handle:
            fortran = handle["attenuationDisk"].shape[::-1]
            assert fortran != (DEPTHS.size, INCLINATIONS.size, WAVELENGTHS.size)
            assert handle.attrs["formatVersion"] == 2


class TestRoundTrip:
    @pytest.mark.parametrize("spheroid_dust", [False, True], ids=["v1", "v2"])
    def test_what_is_written_can_be_read_back(self, tmp_path, spheroid_dust):
        original = build(spheroid_dust=spheroid_dust)
        path = tmp_path / "attenuations.hdf5"
        original.write(str(path))
        restored = read_tabulation(str(path))
        assert restored.label == original.label
        assert restored.opacity == pytest.approx(original.opacity)
        assert restored.format_version == original.format_version
        assert set(restored.emitters) == set(original.emitters)
        np.testing.assert_allclose(restored.wavelengths, original.wavelengths)
        np.testing.assert_allclose(restored.inclinations, original.inclinations)
        for emitter in original.emitters:
            np.testing.assert_allclose(restored.attenuation[emitter], original.attenuation[emitter])
            np.testing.assert_allclose(restored.extrapolation[emitter], original.extrapolation[emitter])

    def test_metadata_survives(self, tmp_path):
        path = tmp_path / "attenuations.hdf5"
        build().write(str(path))
        assert read_tabulation(str(path)).metadata["dustDescription"] == "test grains"

    def test_axes_are_recovered_under_their_own_names(self, tmp_path):
        """Not the dataset names: those are the published ones and do not invert."""
        path = tmp_path / "attenuations.hdf5"
        build().write(str(path))
        restored = read_tabulation(str(path))
        assert restored.emitter_axes["spheroid"] == (
            "opticalDepth:disk",
            "scaleRadial:spheroid",
        )
        assert restored.axis_names == ("opticalDepth:disk", "scaleRadial:spheroid")

    def test_the_stored_dataset_names_remain_the_published_ones(self, tmp_path):
        path = tmp_path / "attenuations.hdf5"
        build().write(str(path))
        with h5py.File(path, "r") as handle:
            assert list(handle["attenuationSpheroid"].attrs["axes"]) == [
                "wavelength",
                "inclination",
                "opticalDepth",
                "spheroidScaleRadial",
            ]

    def test_a_file_without_an_opacity_is_refused(self, tmp_path):
        path = tmp_path / "not-a-tabulation.hdf5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("wavelength", data=WAVELENGTHS)
        with pytest.raises(ValueError, match="not a dust compendium tabulation"):
            read_tabulation(str(path))


class TestChecks:
    def test_a_mismatched_attenuation_is_caught_before_writing(self, tmp_path):
        tabulation = build()
        tabulation.attenuation["disk"] = np.zeros((2, 2, 2))
        with pytest.raises(ValueError, match="attenuation for 'disk' has shape"):
            tabulation.write(str(tmp_path / "bad.hdf5"))

    def test_a_mismatched_extrapolation_is_caught(self, tmp_path):
        tabulation = build()
        tabulation.extrapolation["spheroid"] = np.zeros((2, 2))
        with pytest.raises(ValueError, match="extrapolation for 'spheroid' has shape"):
            tabulation.write(str(tmp_path / "bad.hdf5"))

    def test_the_extrapolation_axis_is_the_disk_optical_depth(self):
        assert build().extrapolation_axis == "opticalDepth:disk"
