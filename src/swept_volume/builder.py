from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..motion.pose import ToolPose
from ..tool.tool_geometry import ToolGeometry
from .envelope import select_translation_envelope_between_poses


@dataclass(frozen=True)
class SweptVolume:
    """Pre-stacked triangle data for zero-overhead numpy sampling.

    All triangle geometry is stored in contiguous (n_tri, …) arrays so the
    sampler can use them directly without a per-triangle Python loop.
    """

    vertices: np.ndarray       # (n_tri, 3, 3) — [tri, vertex, xyz]
    normals: np.ndarray        # (n_tri, 3) — unit normals
    component_ids: np.ndarray  # (n_tri,) int — -1 means "no component"

    sources: np.ndarray

    @property
    def n_tri(self) -> int:
        return len(self.vertices)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n_tri == 0:
            nan = np.full(3, np.nan)
            return nan, nan
        flat = self.vertices.reshape(-1, 3)
        return flat.min(axis=0), flat.max(axis=0)

    @property
    def triangles(self):
        """Lazy backward-compatible Triangle tuple — not used in the hot path."""
        from .triangle import Triangle
        return tuple(
            Triangle(
                self.vertices[k, 0],
                self.vertices[k, 1],
                self.vertices[k, 2],
                normal=self.normals[k],
                component_id=(
                    None if int(self.component_ids[k]) < 0
                    else int(self.component_ids[k])
                ),
                source=str(self.sources[k]),
            )
            for k in range(self.n_tri)
        )

    @classmethod
    def empty(cls) -> SweptVolume:
        return cls(
            np.empty((0, 3, 3), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty(0, dtype=int),
            np.empty(0, dtype=object),
        )


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Normalise each row of (N, 3) to unit length; zero rows stay zero."""
    lens = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.where(lens > 1e-12, lens, 1.0)
    return np.where(lens > 1e-12, arr / safe, 0.0)


class SweptVolumeBuilder:
    """Build a triangle swept surface between sampled tool poses."""

    def __init__(
        self,
        tool: ToolGeometry,
        radial_segments: int = 24,
        axial_segments: int = 8,
        use_envelope: bool = False,
        envelope_eps: float = 0.05,
        active_cutting_only: bool = True,
    ) -> None:
        self.tool = tool
        self.radial_segments = radial_segments
        self.axial_segments = axial_segments
        self.use_envelope = use_envelope
        self.envelope_eps = envelope_eps
        self.active_cutting_only = active_cutting_only

    def build_between(
        self,
        start: ToolPose,
        end: ToolPose,
        samples: int = 2,
    ) -> SweptVolume:
        if self.use_envelope:
            return self.build_translation_envelope_between(start, end)
        poses = [
            start.interpolate(end, t)
            for t in np.linspace(0.0, 1.0, max(2, int(samples)))
        ]
        return self.build_from_poses(poses)

    def build_translation_envelope_between(
        self,
        start: ToolPose,
        end: ToolPose,
    ) -> SweptVolume:
        """Build a 3-axis translational swept surface from envelope candidates."""
        envelope = select_translation_envelope_between_poses(
            self.tool,
            start,
            end,
            eps=self.envelope_eps,
            n_u=self.radial_segments,
            n_v=self.axial_segments,
            include_shank=True,
            include_non_cutting=not self.active_cutting_only,
        )
        if envelope.count < 2:
            return SweptVolume.empty()

        p0 = envelope.points                               # (N, 3)
        p1 = p0 + (end.position - start.position)
        n0 = envelope.normals                              # (N, 3)
        comp_ids_env = np.asarray(envelope.component_ids, dtype=int)

        groups: dict[int, list[int]] = {}
        for local_idx, comp in enumerate(comp_ids_env):
            groups.setdefault(int(comp), []).append(local_idx)

        all_verts: list[np.ndarray] = []
        all_norms: list[np.ndarray] = []
        all_comps: list[np.ndarray] = []

        for group_indices in groups.values():
            if len(group_indices) < 2:
                continue
            ia = np.array(group_indices[:-1], dtype=int)
            ib = np.array(group_indices[1:], dtype=int)
            comp_id = int(comp_ids_env[ia[0]])

            # T1: (p0[a], p1[a], p1[b])
            T1v = np.stack([p0[ia], p1[ia], p1[ib]], axis=1)  # (M, 3, 3)
            # T2: (p0[a], p1[b], p0[b])
            T2v = np.stack([p0[ia], p1[ib], p0[ib]], axis=1)
            Tn = n0[ia] + n0[ib]                               # (M, 3)

            all_verts += [T1v, T2v]
            all_norms += [Tn, Tn]
            m = len(ia)
            all_comps += [np.full(m, comp_id, dtype=int), np.full(m, comp_id, dtype=int)]

        if not all_verts:
            return SweptVolume.empty()

        verts_arr = np.concatenate(all_verts, axis=0)
        norms_arr = _normalize_rows(np.concatenate(all_norms, axis=0))
        comp_arr = np.concatenate(all_comps, axis=0)
        src_arr = np.full(len(comp_arr), "translation_envelope", dtype=object)
        return SweptVolume(verts_arr, norms_arr, comp_arr, src_arr)

    def build_from_poses(self, poses: Iterable[ToolPose]) -> SweptVolume:
        pose_list = list(poses)
        if len(pose_list) < 2:
            raise ValueError("at least two poses are required")

        rings: list[np.ndarray] = []
        ring_normals: list[np.ndarray] = []
        ring_comp_ids: list[np.ndarray] = []

        # For pure-translation moves (same rotation throughout), reuse the local
        # surface sample from the first pose and just translate it for subsequent
        # poses.  This avoids repeated transform_surface calls for the common 3-axis case.
        cached_local_pts: np.ndarray | None = None
        cached_local_norms: np.ndarray | None = None
        cached_ids: np.ndarray | None = None
        ref_rotation: np.ndarray | None = None

        for pose in pose_list:
            same_rotation = (
                ref_rotation is not None
                and np.allclose(pose.rotation, ref_rotation, atol=1e-9)
            )
            if same_rotation and cached_local_pts is not None:
                # Translate the already-transformed first ring to this pose's position.
                offset = pose.position - pose_list[0].position
                pts = cached_local_pts + offset
                nrm = cached_local_norms
                ids = cached_ids
            else:
                surface = self.tool.transform_surface(
                    pose,
                    radial_segments=self.radial_segments,
                    axial_segments=self.axial_segments,
                    include_non_cutting=not self.active_cutting_only,
                )
                pts = surface["points"]
                nrm = surface["normals"]
                ids_raw = surface.get("component_ids")
                ids = (
                    np.full(pts.shape[:2], -1, dtype=int)
                    if ids_raw is None
                    else np.asarray(ids_raw, dtype=int)
                )
                if ref_rotation is None:
                    cached_local_pts = pts
                    cached_local_norms = nrm
                    cached_ids = ids
                    ref_rotation = pose.rotation

            rings.append(pts)
            ring_normals.append(nrm)
            ring_comp_ids.append(ids)

        n_ring, n_col, _ = rings[0].shape
        n_pairs = len(rings) - 1

        all_verts: list[np.ndarray] = []
        all_norms: list[np.ndarray] = []
        all_comps: list[np.ndarray] = []

        i_idx = np.arange(n_ring)
        ni_idx = (i_idx + 1) % n_ring
        j_face = np.arange(n_col - 1)   # 0 .. n_col-2
        j_all = np.arange(n_col)         # 0 .. n_col-1
        M_face = n_ring * (n_col - 1)

        for a in range(n_pairs):
            p0, p1 = rings[a], rings[a + 1]                 # (R, C, 3)
            n0, n1 = ring_normals[a], ring_normals[a + 1]
            c0, c1 = ring_comp_ids[a], ring_comp_ids[a + 1]

            # ---- side surface triangles ----------------------------------------
            # Index helpers: (R, C-1, 3)
            P0ij   = p0[i_idx[:, None],  j_face[None, :], :]
            P0nij  = p0[ni_idx[:, None], j_face[None, :], :]
            P1ij   = p1[i_idx[:, None],  j_face[None, :], :]
            P1nij  = p1[ni_idx[:, None], j_face[None, :], :]
            N0ij   = n0[i_idx[:, None],  j_face[None, :], :]
            N0nij  = n0[ni_idx[:, None], j_face[None, :], :]
            N1ij   = n1[i_idx[:, None],  j_face[None, :], :]
            N1nij  = n1[ni_idx[:, None], j_face[None, :], :]
            C0ij   = c0[i_idx[:, None],  j_face[None, :]]   # (R, C-1)

            # np.stack(..., axis=2) inserts vertex axis: (R, C-1, 3_verts, 3_coords)
            T1v = np.stack([P0ij, P1ij, P1nij], axis=2).reshape(M_face, 3, 3)
            T2v = np.stack([P0ij, P1nij, P0nij], axis=2).reshape(M_face, 3, 3)
            T1n = (N0ij + N1ij + N1nij).reshape(M_face, 3)
            T2n = (N0ij + N1nij + N0nij).reshape(M_face, 3)
            C_face_flat = C0ij.reshape(M_face)

            all_verts += [T1v, T2v]
            all_norms += [T1n, T2n]
            all_comps += [C_face_flat, C_face_flat]

            # ---- side cap triangles (at j=0 and j=n_col-1) --------------------
            for j_cap in (0, n_col - 1):
                c0_ctr = p0[:, j_cap].mean(axis=0)   # (3,)
                c1_ctr = p1[:, j_cap].mean(axis=0)
                P0cap  = p0[i_idx, j_cap, :]          # (R, 3)
                P1cap  = p1[i_idx, j_cap, :]
                Ccap   = c0[i_idx, j_cap]             # (R,)

                # Triangle (c0_ctr, c1_ctr, p1[i, j_cap])
                Tv1 = np.stack([
                    np.broadcast_to(c0_ctr, (n_ring, 3)).copy(),
                    np.broadcast_to(c1_ctr, (n_ring, 3)).copy(),
                    P1cap,
                ], axis=1)
                # Triangle (c0_ctr, p1[i, j_cap], p0[i, j_cap])
                Tv2 = np.stack([
                    np.broadcast_to(c0_ctr, (n_ring, 3)).copy(),
                    P1cap,
                    P0cap,
                ], axis=1)
                zeros_R3 = np.zeros((n_ring, 3))
                all_verts += [Tv1, Tv2]
                all_norms += [zeros_R3, zeros_R3]
                all_comps += [Ccap, Ccap]

            # ---- start cap and end cap -----------------------------------------
            for p, n_arr, c_ids in ((p0, n0, c0), (p1, n1, c1)):
                # Cap face triangles: (R, C-1) pairs of triangles
                Pij   = p[i_idx[:, None],  j_face[None, :], :]
                Pnij  = p[ni_idx[:, None], j_face[None, :], :]
                Pij1  = p[i_idx[:, None],  (j_face + 1)[None, :], :]
                Pnij1 = p[ni_idx[:, None], (j_face + 1)[None, :], :]
                Nij   = n_arr[i_idx[:, None],  j_face[None, :], :]
                Nnij  = n_arr[ni_idx[:, None], j_face[None, :], :]
                Nij1  = n_arr[i_idx[:, None],  (j_face + 1)[None, :], :]
                Nnij1 = n_arr[ni_idx[:, None], (j_face + 1)[None, :], :]
                Cface = c_ids[i_idx[:, None], j_face[None, :]]   # (R, C-1)

                Tcv1 = np.stack([Pij, Pnij, Pnij1], axis=2).reshape(M_face, 3, 3)
                Tcv2 = np.stack([Pij, Pnij1, Pij1], axis=2).reshape(M_face, 3, 3)
                Tcn1 = (Nij + Nnij + Nnij1).reshape(M_face, 3)
                Tcn2 = (Nij + Nnij1 + Nij1).reshape(M_face, 3)
                Cface_flat = Cface.reshape(M_face)

                all_verts += [Tcv1, Tcv2]
                all_norms += [Tcn1, Tcn2]
                all_comps += [Cface_flat, Cface_flat]

                # Cap fan triangles: centre of each column → all ring pairs
                # centers[j] = p[:, j].mean(axis=0) for all j at once
                centers = p.mean(axis=0)    # (C, 3)
                # Broadcast to (R, C, 3)
                centers_bc = np.broadcast_to(centers[None, :, :], (n_ring, n_col, 3)).copy()
                Prj  = p[i_idx[:, None],   j_all[None, :], :]   # (R, C, 3)
                Prnj = p[ni_idx[:, None],  j_all[None, :], :]
                Nrj  = n_arr[i_idx[:, None],  j_all[None, :], :]
                Nrnj = n_arr[ni_idx[:, None], j_all[None, :], :]
                # component_id: from first ring element per column
                Cfan = np.broadcast_to(
                    c_ids[0, j_all][None, :], (n_ring, n_col)
                ).copy()

                Tfan_v = np.stack([centers_bc, Prj, Prnj], axis=2)  # (R, C, 3, 3)
                Tfan_n = Nrj + Nrnj                                   # (R, C, 3)
                M_fan = n_ring * n_col

                all_verts.append(Tfan_v.reshape(M_fan, 3, 3))
                all_norms.append(Tfan_n.reshape(M_fan, 3))
                all_comps.append(Cfan.reshape(M_fan))

        if not all_verts:
            return SweptVolume.empty()

        verts_arr = np.concatenate(all_verts, axis=0)    # (n_tri, 3, 3)
        norms_arr = _normalize_rows(np.concatenate(all_norms, axis=0))
        comp_arr  = np.concatenate(all_comps, axis=0).astype(int)
        src_arr = np.full(len(comp_arr), "swept", dtype=object)
        return SweptVolume(verts_arr, norms_arr, comp_arr, src_arr)
