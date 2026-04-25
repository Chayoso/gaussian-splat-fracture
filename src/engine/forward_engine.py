"""
ForwardEngine — callable simulation wrapper for inverse optimization.

Caches expensive mesh/particle setup, re-creates parameter-dependent
components (elasticity, simulator) per simulate() call.
"""

import sys
import copy
import time
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gaussian-splatting"))

from src.config.loader import load_config, apply_overrides_dict
from src.mpm_core.mpm_pipeline import create_mpm_model, configure_loading
from src.constitutive_models.model_factory import create_elasticity_model
from src.engine.loading_transforms import apply_loading_transforms
from src.rendering.postprocess import depth_to_normal, create_video


class ForwardEngine:
    """
    Callable forward simulation engine.

    Usage:
        engine = ForwardEngine("configs/gravity_drop_test.yaml")
        frames = engine.simulate(E=1.5e7, Gc=60000, nu=0.25, num_frames=100)
    """

    def __init__(self, config_path: str, fast_mode: bool = False):
        """
        Initialize engine and cache expensive mesh/particle setup.

        Args:
            config_path: Path to YAML config file
            fast_mode: If True, reduce particles/resolution for faster iteration
        """
        self.config_path = config_path
        self.base_config = load_config(config_path)

        if fast_mode:
            self._apply_fast_mode(self.base_config)

        # Seed before mesh setup (must match run.py order)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Resolve device
        device_type = self.base_config.device.type
        if device_type == "cuda" and not torch.cuda.is_available():
            device_type = "cpu"
        self.device = torch.device(device_type)
        if device_type == "cuda" and hasattr(self.base_config.device, 'gpu_id'):
            torch.cuda.set_device(self.base_config.device.gpu_id)

        # Apply material preset
        from src.core.material_presets import resolve_material_preset, validate_l0
        self.base_config = resolve_material_preset(self.base_config)
        self.base_config = validate_l0(self.base_config)

        # Cache: mesh → point clouds (expensive, ~10s)
        self._setup_mesh_cache()

    def _setup_mesh_cache(self):
        """One-time mesh processing and particle sampling."""
        from src.preprocessing.mesh_converter import MeshToPointCloudConverter

        config = self.base_config
        converter = MeshToPointCloudConverter(
            mesh_path=config.mesh.path,
            target_particle_count=config.particles.target_count,
            surface_sample_ratio=config.particles.get('surface_ratio', 0.5),
            use_poisson=config.particles.get('use_poisson_sampling', False),
            poisson_depth=config.particles.get('poisson_depth', 8),
            normalize_to_unit_cube=config.particles.get('normalize_to_unit_cube', True),
        )

        self._volume_pcd, self._surface_pcd, self._surface_mask = converter.convert()
        self._mesh_meta = getattr(converter, 'mesh_meta', None)

        # Store numpy copies for cloning
        self._volume_points_np = np.asarray(self._volume_pcd.points).copy()
        self._volume_normals_np = np.asarray(self._volume_pcd.normals).copy()
        self._surface_mask_np = self._surface_mask.copy()

    def simulate(self, E: float = None, Gc: float = None, nu: float = None,
                 num_frames: int = None, seed: int = 42,
                 save_frames: bool = False, return_frames: bool = True,
                 material_tint: tuple = None,
                 material_texture: str = None,
                 **kwargs) -> List[torch.Tensor]:
        """
        Run a forward simulation with given material parameters.

        Args:
            E: Young's modulus (overrides config)
            Gc: Fracture toughness (overrides config)
            nu: Poisson's ratio (overrides config)
            num_frames: Total frames (overrides config)
            seed: Random seed for reproducibility
            save_frames: If True, save PNG frames to disk
            return_frames: If True, return list of (3,H,W) tensors

        Returns:
            List of rendered frame tensors if return_frames=True, else empty list
        """
        # Deterministic seeding
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Deep-copy config and apply overrides
        config = OmegaConf.create(OmegaConf.to_container(self.base_config, resolve=True))
        overrides = {"E": E, "Gc": Gc, "nu": nu, "num_frames": num_frames}
        overrides.update(kwargs)
        config = apply_overrides_dict(config, overrides)

        if not save_frames:
            OmegaConf.update(config, "simulation.save_frames", False)

        # Parameter-dependent setup
        mpm_model = create_mpm_model(config, self._volume_pcd, self.device)
        loading_params = configure_loading(config, mpm_model, self.device)
        elasticity = create_elasticity_model(config, self.device)
        gaussians = self._create_gaussians(config)

        simulator = self._create_simulator(config, mpm_model, gaussians,
                                            elasticity, loading_params)

        # Set initial normals
        if self._volume_normals_np is not None:
            all_normals = torch.from_numpy(self._volume_normals_np).float().to(self.device)
            simulator.visualizer.set_initial_normals(all_normals)

        # Apply procedural material texture BEFORE initialize
        # (initialize triggers step_rendering which caches _original_dc)
        self._material_props = None
        if material_texture is not None:
            from src.ml.material_texture import apply_material_texture, get_material_properties
            gauss_pos = gaussians._xyz.data.detach().cpu().numpy()
            apply_material_texture(gaussians, gauss_pos, material_texture)
            self._material_props = get_material_properties(material_texture)

        # Initialize and apply transforms
        simulator.initialize(
            torch.from_numpy(self._volume_points_np).float().to(self.device))

        # Pass surface normals for manifold-aware graph (ManifoldSimulator only)
        if self._volume_normals_np is not None and hasattr(simulator, 'set_surface_normals'):
            all_normals = torch.from_numpy(self._volume_normals_np).float().to(self.device)
            simulator.set_surface_normals(all_normals)

        apply_loading_transforms(config, simulator, loading_params, self.device)

        # Setup camera
        camera = self._setup_camera(config)

        # Pass camera position for back-face normal flipping
        simulator._camera_pos = camera.camera_center

        # Run simulation and collect frames
        frames = self._run_loop(config, simulator, camera,
                                save_frames=save_frames,
                                return_frames=return_frames,
                                material_tint=material_tint,
                                material_props=self._material_props)
        return frames

    def render_from_checkpoint(self, checkpoint_path: str, num_frames: int = 50,
                               save_frames: bool = True, **config_overrides) -> List[torch.Tensor]:
        """
        Load checkpoint and continue simulation + rendering from that point.

        Skips all physics before the checkpoint frame — only runs from checkpoint onward.
        Useful for re-rendering with different camera/shading without re-running physics.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            num_frames: Frames to simulate AFTER the checkpoint
            save_frames: Save PNG frames to disk
            **config_overrides: Override config values (e.g. camera settings)
        """
        config = OmegaConf.create(OmegaConf.to_container(self.base_config, resolve=True))
        config = apply_overrides_dict(config, config_overrides)

        # Build full simulator
        mpm_model = create_mpm_model(config, self._volume_pcd, self.device)
        loading_params = configure_loading(config, mpm_model, self.device)
        elasticity = create_elasticity_model(config, self.device)
        gaussians = self._create_gaussians(config)

        simulator = self._create_simulator(config, mpm_model, gaussians,
                                            elasticity, loading_params)

        if self._volume_normals_np is not None:
            all_normals = torch.from_numpy(self._volume_normals_np).float().to(self.device)
            simulator.visualizer.set_initial_normals(all_normals)

        simulator.initialize(
            torch.from_numpy(self._volume_points_np).float().to(self.device))
        apply_loading_transforms(config, simulator, loading_params, self.device)

        # Load checkpoint (overwrites positions, velocities, damage, etc.)
        simulator.load_state(checkpoint_path)

        camera = self._setup_camera(config)
        simulator._camera_pos = camera.camera_center

        # Override total frames to run from checkpoint
        OmegaConf.update(config, "rendering.total_frames",
                         simulator.frame_count + num_frames)

        frames = self._run_loop(config, simulator, camera,
                                save_frames=save_frames,
                                return_frames=True,
                                start_frame=simulator.frame_count)
        return frames

    def _create_gaussians(self, config):
        """Create GaussianModel from cached surface point cloud."""
        # Import here to avoid top-level dependency on gaussian-splatting
        from scene.gaussian_model import GaussianModel

        gaussians = GaussianModel(sh_degree=config.gaussian_splatting.sh_degree)

        pretrained_ply = config.gaussian_splatting.get("pretrained_ply", None)
        if pretrained_ply is not None:
            from src.preprocessing.ply_loader import PretrainedPlyLoader
            scale_mult = config.gaussian_splatting.get("pretrained_scale_multiplier", 1.0)
            use_direct = config.gaussian_splatting.get("ply_direct", False)

            loader = PretrainedPlyLoader(
                ply_path=pretrained_ply,
                sh_degree=config.gaussian_splatting.sh_degree,
            )
            ply_data = loader.load_raw_ply()
            ply_xyz_norm, scale_factor = loader.normalize_positions(ply_data['xyz'])

            if use_direct:
                ply_to_surface = loader.create_direct_gaussians(
                    gaussians, ply_data, ply_xyz_norm,
                    np.asarray(self._surface_pcd.points),
                    scale_factor, scale_multiplier=scale_mult,
                )
                gaussians._ply_to_surface = torch.tensor(
                    ply_to_surface, dtype=torch.long, device="cuda")
                gaussians._ply_original_xyz = gaussians._xyz.data.clone()
            else:
                match_indices, distances = loader.match_to_surface_particles(
                    ply_xyz_norm, np.asarray(self._surface_pcd.points))
                loader.create_matched_gaussians(
                    gaussians, self._surface_pcd, ply_data, match_indices,
                    scale_factor, scale_multiplier=scale_mult)
        else:
            # Procedural Gaussians from mesh
            cam_distance = config.rendering.camera.distance
            cam_fov = np.radians(config.rendering.camera.fov)
            img_width = config.rendering.image_width
            gaussians.create_from_pcd(
                self._surface_pcd,
                cam_infos=[],
                spatial_lr_scale=1.0,
                camera_distance=cam_distance,
                image_width=img_width,
                fov_x=cam_fov,
            )

        return gaussians

    def _create_simulator(self, config, mpm_model, gaussians, elasticity, loading_params):
        """Create simulator (ManifoldSimulator or legacy HybridCrackSimulator)."""
        from src.core.coordinate_mapper import CoordinateMapper
        from src.visualization.gaussian_updater import GaussianCrackVisualizer

        coord_mapper = CoordinateMapper(
            mpm_bounds=(0.0, 1.0),
            world_center=np.array(list(config.coordinate_mapping.world_center)),
            world_scale=float(config.coordinate_mapping.world_scale),
            device=str(self.device)
        )

        visualizer = GaussianCrackVisualizer(
            damage_threshold=config.gaussian_splatting.damage_threshold,
            device=str(self.device),
            crack_color=tuple(config.gaussian_splatting.get('crack_color', [0.6, 0.08, 0.08])),
            crack_opacity_reduction=float(config.gaussian_splatting.get('crack_opacity_reduction', 0.70)),
            crack_max_opening=float(config.gaussian_splatting.get('crack_max_opening', 0.010)),
            crack_gap_fraction=float(config.gaussian_splatting.get('crack_gap_fraction', 0.35)),
            crack_edge_darken=float(config.gaussian_splatting.get('crack_edge_darken', 0.75)),
            crack_red_accent=float(config.gaussian_splatting.get('crack_red_accent', 0.10)),
            crack_tip_scale_boost=float(config.gaussian_splatting.get('crack_tip_scale_boost', 0.20)),
            crack_tip_opacity_boost=float(config.gaussian_splatting.get('crack_tip_opacity_boost', 0.10)),
            material_family=str(config.gaussian_splatting.get('material_family', 'neutral_reference')),
            crack_band_weight=float(config.gaussian_splatting.get('crack_band_weight', 0.80)),
            crack_visited_weight=float(config.gaussian_splatting.get('crack_visited_weight', 0.45)),
            crack_tip_weight=float(config.gaussian_splatting.get('crack_tip_weight', 0.95)),
            crack_core_weight=float(config.gaussian_splatting.get('crack_core_weight', 1.00)),
            split_gap_gain=float(config.gaussian_splatting.get('split_gap_gain', 1.0)),
            fragment_shell_gain=float(config.gaussian_splatting.get('fragment_shell_gain', 1.0)),
            fragment_contrast_gain=float(config.gaussian_splatting.get('fragment_contrast_gain', 1.0)),
            debris_darkening=float(config.gaussian_splatting.get('debris_darkening', 0.20)),
            shard_scale_gain=float(config.gaussian_splatting.get('shard_scale_gain', 1.0)),
            shard_opacity_gain=float(config.gaussian_splatting.get('shard_opacity_gain', 1.0)),
            damage_scale_shrink=float(config.gaussian_splatting.get('damage_scale_shrink', 0.50)),
            damage_center_opacity_reduction=float(config.gaussian_splatting.get('damage_center_opacity_reduction', 0.70)),
            diffuse_damage_strength=float(config.gaussian_splatting.get('diffuse_damage_strength', 0.12)),
            interior_surface_enable=bool(config.gaussian_splatting.get('interior_surface_enable', True)),
            interior_surface_threshold=float(config.gaussian_splatting.get('interior_surface_threshold', 0.34)),
            interior_surface_max_fraction=float(config.gaussian_splatting.get('interior_surface_max_fraction', 0.012)),
            interior_surface_scale=float(config.gaussian_splatting.get('interior_surface_scale', 0.82)),
            interior_surface_opacity=float(config.gaussian_splatting.get('interior_surface_opacity', 0.72)),
            interior_surface_darken=float(config.gaussian_splatting.get('interior_surface_darken', 0.38)),
            interior_surface_gap_gain=float(config.gaussian_splatting.get('interior_surface_gap_gain', 0.72)),
        )

        # Seismic params
        seismic_params = {}
        if hasattr(config, 'seismic'):
            seismic_enabled = config.seismic.get('enabled', False)
            if loading_params.get("seismic_override") is not None:
                seismic_enabled = loading_params["seismic_override"]
            seismic_params = {
                'enabled': seismic_enabled,
                'amplitude': float(config.seismic.get('amplitude', 0)),
                'frequency': float(config.seismic.get('frequency', 50)),
                'direction': list(config.seismic.get('direction', [1, 0, 0])),
                'ramp_time': float(config.seismic.get('ramp_time', 0.01)),
            }

        surface_mask = torch.from_numpy(self._surface_mask_np).bool().to(self.device)

        # Choose simulator based on config
        use_manifold = config.simulation.get('use_manifold', False)

        if use_manifold:
            from src.core.manifold_simulator import ManifoldSimulator

            # Fracture params from phase_field + manifold config sections
            pf_params = OmegaConf.to_container(config.phase_field, resolve=True)
            manifold_cfg = OmegaConf.to_container(
                config.get('manifold', {}), resolve=True) if hasattr(config, 'manifold') else {}
            fracture_params = {**pf_params, **manifold_cfg}

            return ManifoldSimulator(
                mpm_model=mpm_model,
                gaussians=gaussians,
                elasticity_module=elasticity,
                coord_mapper=coord_mapper,
                visualizer=visualizer,
                surface_mask=surface_mask,
                physics_substeps=config.rendering.physics_substeps,
                fracture_params=fracture_params,
                simulation_mode=config.simulation.mode,
                seismic_params=seismic_params,
            )
        else:
            # Legacy path: import from legacy/ directory
            import sys as _sys
            _legacy_root = str(PROJECT_ROOT / "legacy")
            if _legacy_root not in _sys.path:
                _sys.path.insert(0, _legacy_root)
            from legacy.src.core.hybrid_simulator import HybridCrackSimulator
            from legacy.src.constitutive_models.damage_mapper import VolumetricToSurfaceDamageMapper

            damage_mapper = VolumetricToSurfaceDamageMapper(
                projection_method=config.damage_projection.method,
                k_neighbors=config.damage_projection.k_neighbors,
                influence_radius=config.damage_projection.influence_radius,
                damage_threshold=config.damage_projection.get('damage_threshold', 0.01),
                use_faiss=config.damage_projection.get('use_faiss', True),
                device=str(self.device)
            )

            pf_params = OmegaConf.to_container(config.phase_field, resolve=True)

            return HybridCrackSimulator(
                mpm_model=mpm_model,
                gaussians=gaussians,
                elasticity_module=elasticity,
                coord_mapper=coord_mapper,
                damage_mapper=damage_mapper,
                visualizer=visualizer,
                surface_mask=surface_mask,
                physics_substeps=config.rendering.physics_substeps,
                phase_field_params=pf_params,
                simulation_mode=config.simulation.mode,
                seismic_params=seismic_params,
            )

    def _setup_camera(self, config):
        """Build rendering camera from config."""
        from src.renderer.camera.config import make_matrices_from_yaml
        from scene.cameras import MiniCam

        cam_config = config.rendering.camera
        width = config.rendering.image_width
        height = config.rendering.image_height

        elev_rad = np.radians(cam_config.elevation)
        azim_rad = np.radians(cam_config.azimuth)
        distance = cam_config.distance

        target_cfg = cam_config.get('target', [0.5, 0.5, 0.5])
        target = np.array(list(target_cfg))

        x = target[0] + distance * np.cos(elev_rad) * np.cos(azim_rad)
        y = target[1] + distance * np.cos(elev_rad) * np.sin(azim_rad)
        z = target[2] + distance * np.sin(elev_rad)
        eye = [float(x), float(y), float(z)]

        fov_deg = cam_config.fov
        fov_rad = np.radians(fov_deg)
        fx = width / (2.0 * np.tan(fov_rad / 2.0))
        fy = fx
        cx, cy = width / 2.0, height / 2.0

        camera_yaml = {
            "width": width, "height": height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "znear": 0.01, "zfar": 100.0,
            "lookat": {"eye": eye, "target": target.tolist(), "up": [0, 0, 1]}
        }

        w, h, tanfovx, tanfovy, view_matrix, proj_matrix, _ = make_matrices_from_yaml(camera_yaml)
        wvt = torch.from_numpy(view_matrix).cuda()
        fpt = torch.from_numpy(proj_matrix).cuda()

        fov_x = 2.0 * np.arctan(tanfovx)
        fov_y = 2.0 * np.arctan(tanfovy)

        return MiniCam(w, h, fov_y, fov_x, 0.01, 100.0, wvt, fpt)

    def _run_loop(self, config, simulator, camera,
                  save_frames=False, return_frames=True,
                  start_frame=0, material_tint=None,
                  material_props=None) -> List[torch.Tensor]:
        """Execute simulation loop and optionally collect rendered frames."""
        from gaussian_renderer import render

        device = self.device
        bg_color = torch.tensor(config.rendering.background_color, device=device)
        pipe = type('obj', (object,), {
            'convert_SHs_python': False,
            'compute_cov3D_python': False,
            'debug': True,
            'antialiasing': False
        })()

        total_frames = config.rendering.total_frames
        frames_out = []
        render_last_only = bool(config.rendering.get('render_last_only', False))
        render_every = max(int(config.rendering.get('render_every', 1) or 1), 1)
        render_frames_cfg = config.rendering.get('render_frames', None)
        render_frame_set = None
        if render_frames_cfg is not None:
            render_frame_set = {int(f) for f in list(render_frames_cfg)}

        def should_render_frame(frame_idx: int) -> bool:
            if render_frame_set is not None:
                return frame_idx in render_frame_set
            if render_last_only:
                return frame_idx == total_frames - 1
            return (frame_idx % render_every == 0) or (frame_idx == total_frames - 1)

        # Apply impact/notch if configured
        if config.external_force.enabled:
            import torch as _torch
            impact_center = _torch.tensor(
                list(config.external_force.center), device=self.device)
            impact_dir = None
            if hasattr(config.external_force, 'direction') and config.external_force.direction is not None:
                impact_dir = _torch.tensor(
                    list(config.external_force.direction), device=self.device)
            simulator.initialize_deformation_impact(
                impact_center_mpm=impact_center,
                impact_energy=float(config.external_force.magnitude),
                impact_radius=float(config.external_force.radius),
                impact_direction=impact_dir,
            )

        if hasattr(config, 'pre_notch') and config.pre_notch.get('enabled', False):
            notch_list = []
            for notch in config.pre_notch.notches:
                notch_list.append({
                    'start': list(notch.start),
                    'end': list(notch.end),
                    'damage': float(notch.damage),
                })
            simulator.apply_pre_notch(notch_list)

        # Output dirs
        frame_dir = Path(config.output.frame_dir)
        if save_frames:
            frame_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint dir
        ckpt_dir = Path(config.output.get('checkpoint_dir', 'output/checkpoints'))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_ckpts = bool(config.simulation.get('save_checkpoint', False))
        ckpt_interval = int(config.simulation.get('checkpoint_interval', 0) or 0)

        # Auto-checkpoint: pre-impact baseline
        ckpt_frames = {40}
        _impact_ckpts_added = False
        # Match run.py diagnostic settings
        simulator._save_diagnostics = False
        simulator._output_dir = str(Path(config.output.get('frame_dir', 'output/frames')).parent)
        simulator._diag_frames = set()

        start_time = time.time()

        for frame in range(start_frame, total_frames):
            simulator._render_frame = frame
            # Physics step (including frame 0 to sync Gaussian positions)
            simulator.step_rendering()

            # Auto-detect impact frame for checkpointing
            if (hasattr(simulator, '_gravity_drop_contacted')
                    and simulator._gravity_drop_contacted
                    and not _impact_ckpts_added):
                _impact_ckpts_added = True
                impact_frame = frame
                ckpt_frames.update({impact_frame, impact_frame + 1,
                                    impact_frame + 5, impact_frame + 10,
                                    impact_frame + 20, impact_frame + 50})

            # Save checkpoint at key frames
            if frame in ckpt_frames:
                ckpt_path = str(ckpt_dir / f"checkpoint_{frame:04d}.pt")
                simulator.save_state(ckpt_path)
            if save_ckpts and ckpt_interval > 0:
                if frame > 0 and frame % ckpt_interval == 0:
                    ckpt_path = str(ckpt_dir / f"checkpoint_{frame:04d}.pt")
                    simulator.save_state(ckpt_path)

            do_render = should_render_frame(frame)

            if not do_render:
                if frame % 20 == 0 or frame == total_frames - 1:
                    elapsed = time.time() - start_time
                    print(f"Frame {frame:04d}/{total_frames}: "
                          f"elapsed={elapsed:.1f}s render=skipped", flush=True)
                continue

            # Render only on selected output frames.
            rendering = render(camera, simulator.gaussians, pipe, bg_color)
            image = rendering["render"]
            depth_map = rendering["depth"]

            # Depth-to-normal post-process shading
            object_mask = (depth_map > 0).float()
            if object_mask.sum() > 0:
                normal_from_depth = depth_to_normal(depth_map, camera)
                cam_pos = camera.camera_center
                scene_center = torch.tensor([0.5, 0.5, 0.5], device='cuda')
                light_dir = scene_center - cam_pos
                light_dir = light_dir / (light_dir.norm() + 1e-8)
                light_dir_reshaped = light_dir.view(3, 1, 1)
                diffuse = (normal_from_depth * light_dir_reshaped).sum(
                    dim=0, keepdim=True).clamp(0.0, 1.0)
                ambient = 0.55
                lit_intensity = ambient + (1.0 - ambient) * diffuse

                # Specular highlights (Blinn-Phong)
                spec_term = torch.zeros_like(diffuse)
                if material_props and material_props.get("specular", 0) > 0.01:
                    view_dir = torch.nn.functional.normalize(
                        cam_pos.view(3, 1, 1) - torch.tensor([0.5, 0.5, 0.5],
                        device='cuda').view(3, 1, 1), dim=0)
                    half_vec = torch.nn.functional.normalize(
                        light_dir_reshaped + view_dir, dim=0)
                    ndoth = (normal_from_depth * half_vec).sum(
                        dim=0, keepdim=True).clamp(0.0, 1.0)
                    shininess = material_props.get("shininess", 32)
                    spec_strength = material_props["specular"]
                    spec_term = spec_strength * (ndoth ** shininess)

                image_lit = image * lit_intensity + spec_term
                bg_intensity = 0.98
                image_bg = image * bg_intensity
                # Gradient background for translucent materials
                if material_props and material_props.get("opacity_scale", 1.0) < 0.8:
                    H, W = depth_map.shape[1], depth_map.shape[2]
                    # Soft vertical gradient: light blue top → warm gray bottom
                    t = torch.linspace(0, 1, H, device='cuda').unsqueeze(1).expand(H, W)
                    bg_r = 0.85 + (1.0 - t) * 0.10   # top lighter
                    bg_g = 0.88 + (1.0 - t) * 0.08
                    bg_b = 0.92 + (1.0 - t) * 0.06
                    image_bg = torch.stack([bg_r, bg_g, bg_b], dim=0)
                else:
                    image_bg = image * bg_intensity

                image = image_bg * (1.0 - object_mask) + image_lit * object_mask
                image = torch.clamp(image, 0.0, 1.0)

            # Apply material color tint (object only, keep background white)
            if material_tint is not None:
                tint = torch.tensor(material_tint, device=image.device).view(3, 1, 1)
                image_tinted = image * tint
                image = image_tinted * object_mask + image * (1.0 - object_mask)
                image = torch.clamp(image, 0.0, 1.0)

            if return_frames:
                frames_out.append(image.detach().cpu())

            if save_frames:
                from torchvision.utils import save_image
                save_image(image, str(frame_dir / f"frame_{frame:04d}.png"))

            # Progress
            if frame % 20 == 0 or frame == total_frames - 1:
                elapsed = time.time() - start_time
                print(f"Frame {frame:04d}/{total_frames}: "
                      f"elapsed={elapsed:.1f}s render=done", flush=True)

        # Video
        if save_frames and bool(config.output.get('make_video', True)):
            create_video(frame_dir, config.output.video_path,
                         config.rendering.fps)

        return frames_out

    @staticmethod
    def _apply_fast_mode(config):
        """Reduce particle count and resolution for faster iteration."""
        OmegaConf.update(config, "particles.target_count",
                         min(50000, config.particles.target_count))
        OmegaConf.update(config, "mpm.num_grids", 64)
        OmegaConf.update(config, "rendering.image_width", 512)
        OmegaConf.update(config, "rendering.image_height", 512)
        OmegaConf.update(config, "rendering.physics_substeps",
                         min(8, config.rendering.physics_substeps))
