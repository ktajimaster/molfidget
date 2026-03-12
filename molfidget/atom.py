import numpy as np
import trimesh

from molfidget.config import AtomConfig, DefaultAtomConfig
from molfidget.constants import atom_radius_table, atom_color_table

class Atom:
    def __init__(self, config: AtomConfig, default: DefaultAtomConfig):
        self.name = config.name
        self.elem, self.id = self.name.split('_')
        if self.elem not in atom_radius_table:
            raise ValueError(f"Unknown atom {self.elem}: name: {self.name}")
        self.radius = atom_radius_table[self.elem]
        self.vdw_scale = config.vdw_scale if config.vdw_scale is not None else default.vdw_scale
        self.shape_radius = self.vdw_scale * self.radius
        self.x, self.y, self.z = config.position
        self.position = config.position
        self.color = config.color if config.color is not None else atom_color_table[self.elem]
        self.in_axis = bool(config.in_axis) if config.in_axis is not None else False
        self.in_axis_gap_mm = config.in_axis_gap_mm if config.in_axis_gap_mm is not None else 0.1
        self.in_axis_shaft_radius = config.in_axis_shaft_radius if config.in_axis_shaft_radius is not None else 0.3
        self.in_axis_shaft_length = config.in_axis_shaft_length if config.in_axis_shaft_length is not None else 0.3
        self.in_axis_stopper_radius = config.in_axis_stopper_radius if config.in_axis_stopper_radius is not None else 0.4
        self.in_axis_stopper_length = config.in_axis_stopper_length if config.in_axis_stopper_length is not None else 0.2
        self.in_axis_chamfer_length = config.in_axis_chamfer_length if config.in_axis_chamfer_length is not None else 0.1
        self.in_axis_wall_thickness = config.in_axis_wall_thickness if config.in_axis_wall_thickness is not None else 0.1
        self.in_axis_shaft_gap = config.in_axis_shaft_gap if config.in_axis_shaft_gap is not None else 0.03
        self.pairs = {}
        self.mesh_parts = None

    def update_bonds(self, bonds: dict):
        self.pairs = {}
        for bond in bonds.values():
            if bond.atom1_name == self.name or bond.atom2_name == self.name:
                self.pairs[(bond.atom1_name, bond.atom2_name)] = bond

    def __repr__(self):
        return f"{self.name}: ({self.x}, {self.y}, {self.z})"

    def create_trimesh_model(self):
        # Sphere mesh for the atom
        self.mesh = trimesh.primitives.Sphere(
            radius = self.shape_radius, center=[0, 0, 0]
        )

    def _resolve_length(self, value, value_mm, scale: float, fallback: float) -> float:
        if value_mm is not None:
            return value_mm / scale
        if value is not None:
            return value
        return fallback

    def _resolve_spin_params(self, spin_default, bond_gap_mm: float, scale: float):
        shaft_radius = self.in_axis_shaft_radius
        shaft_length = self.in_axis_shaft_length
        stopper_radius = self.in_axis_stopper_radius
        stopper_length = self.in_axis_stopper_length
        chamfer_length = self.in_axis_chamfer_length
        wall_thickness = self.in_axis_wall_thickness
        shaft_gap = self.in_axis_shaft_gap
        bond_gap = (bond_gap_mm / scale) if bond_gap_mm is not None else 0.1
        return {
            "shaft_radius": shaft_radius,
            "shaft_length": shaft_length,
            "stopper_radius": stopper_radius,
            "stopper_length": stopper_length,
            "chamfer_length": chamfer_length,
            "wall_thickness": wall_thickness,
            "shaft_gap": shaft_gap,
            "bond_gap": bond_gap,
        }

    def _axis_direction_from_bonds(self, atoms: dict) -> np.ndarray:
        vectors = []
        for bond in self.pairs.values():
            if bond.bond_type in ("none", "plane"):
                continue
            if bond.atom1_name == self.name:
                other_name = bond.atom2_name
            else:
                other_name = bond.atom1_name
            other = atoms.get(other_name)
            if other is None:
                continue
            vec = np.array([other.x - self.x, other.y - self.y, other.z - self.z], dtype=float)
            vec_norm = np.linalg.norm(vec)
            if vec_norm <= 0:
                continue
            vec /= vec_norm
            if vectors:
                current = np.sum(vectors, axis=0)
                if np.dot(vec, current) < 0:
                    continue
            vectors.append(vec)

        if vectors:
            axis = np.sum(vectors, axis=0)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-8:
                return axis / axis_norm
        return np.array([1.0, 0.0, 0.0])

    def _create_spin_shaft(self, p):
        d1 = p["shaft_length"] + p["wall_thickness"] + p["shaft_gap"] - p["chamfer_length"] + p["bond_gap"] / 2
        cylinder1 = trimesh.creation.cylinder(radius=p["shaft_radius"], height=d1)
        cylinder1.apply_translation([0, 0, -d1 / 2])

        cylinder3 = trimesh.creation.cylinder(radius=p["shaft_radius"], height=p["chamfer_length"])
        cylinder3.apply_translation([0, 0, p["chamfer_length"] / 2])
        cone1 = trimesh.creation.cone(radius=p["shaft_radius"], height=2 * p["shaft_radius"], sections=32)
        cone1 = trimesh.boolean.intersection([cone1, cylinder3], check_volume=False)
        cylinder1 = trimesh.boolean.union([cylinder1, cone1], check_volume=False)
        cylinder1.apply_translation([0, 0, p["shaft_length"] - p["chamfer_length"]])

        stopper = trimesh.creation.cylinder(radius=p["stopper_radius"], height=p["stopper_length"])
        stopper.apply_translation([0, 0, -p["stopper_length"] / 2 - p["wall_thickness"] - p["shaft_gap"]])
        return trimesh.boolean.union([cylinder1, stopper], check_volume=False)

    def _create_spin_cavity(self, p):
        eps = 0.01
        d1 = p["wall_thickness"] + eps
        cavity1 = trimesh.creation.cylinder(radius=p["shaft_radius"] + p["shaft_gap"], height=d1)
        cavity1.apply_translation([0, 0, -d1 / 2 + eps])
        d2 = p["stopper_length"] + 2 * p["shaft_gap"]
        cavity2 = trimesh.creation.cylinder(radius=p["stopper_radius"] + p["shaft_gap"], height=d2)
        cavity2.apply_translation([0, 0, -d2 / 2 - d1 + eps])
        return trimesh.boolean.union([cavity1, cavity2], check_volume=False)

    def split_with_internal_axis(self, atoms: dict, spin_default, bond_gap_mm: float, scale: float):
        axis = self._axis_direction_from_bonds(atoms)

        r = self.shape_radius
        sphere = self.mesh.copy()
        half_box = trimesh.creation.box(extents=[4 * r, 4 * r, 2 * r])
        rot = trimesh.geometry.align_vectors([0, 0, 1], axis)

        half_pos_box = half_box.copy()
        half_pos_box.apply_translation([0, 0, r])
        half_pos_box.apply_transform(rot)
        half_neg_box = half_box.copy()
        half_neg_box.apply_translation([0, 0, -r])
        half_neg_box.apply_transform(rot)
        half_pos = trimesh.boolean.intersection([sphere, half_pos_box], check_volume=False)
        half_neg = trimesh.boolean.intersection([sphere, half_neg_box], check_volume=False)
        if half_pos is None or half_neg is None:
            return [self.mesh]
        in_axis_gap = self.in_axis_gap_mm / scale
        if in_axis_gap > 0:
            gap_cut = trimesh.creation.box(extents=[4 * r, 4 * r, in_axis_gap])
            gap_cut.apply_transform(rot)
            half_pos = trimesh.boolean.difference([half_pos, gap_cut], check_volume=False)
            half_neg = trimesh.boolean.difference([half_neg, gap_cut], check_volume=False)

        spin_params = self._resolve_spin_params(spin_default, bond_gap_mm, scale)
        shaft = self._create_spin_shaft(spin_params)
        cavity = self._create_spin_cavity(spin_params)
        if in_axis_gap > 0:
            shaft.apply_translation([0, 0, in_axis_gap / 2])
            cavity.apply_translation([0, 0, -in_axis_gap / 2])
        shaft.apply_transform(rot)
        cavity.apply_transform(rot)

        half_pos = trimesh.boolean.union([half_pos, shaft], check_volume=False)
        half_neg = trimesh.boolean.difference([half_neg, cavity], check_volume=False)
        self.mesh_parts = [half_pos, half_neg]
        return self.mesh_parts
