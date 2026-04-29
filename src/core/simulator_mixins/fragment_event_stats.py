"""FragmentEventStatsMixin for ManifoldSimulator.

This module is behavior-preserving extraction from manifold_simulator.py.
"""



class FragmentEventStatsMixin:
    def _reset_frame_fragment_event_stats(self) -> None:
        """Reset per-visible-frame fragment birth/release counters."""
        self._frame_fragment_stats = {
            "detect_calls": 0,
            "new_detached_nodes": 0,
            "phase_candidate_patches": 0,
            "phase_approved_patches": 0,
            "phase_rejected_patches": 0,
            "phase_approved_nodes": 0,
            "phase_approval_score_max": 0.0,
            "phase_approval_score_weighted_sum": 0.0,
            "phase_approval_cvol_max": 0.0,
            "phase_approval_gate_max": 0.0,
            "pseudo_thickness_mass": 0.0,
            "birth_phase_score": 0.0,
            "birth_cvol_max": 0.0,
            "birth_phase_gate_max": 0.0,
            "birth_phase_approved": False,
            "open_release_patches": 0,
            "open_release_nodes": 0,
            "open_release_score_max": 0.0,
            "impact_closure_patches": 0,
            "impact_closure_nodes": 0,
            "impact_closure_score_max": 0.0,
        }

    def _accumulate_frame_fragment_event_stats(self) -> None:
        """Accumulate stats across multiple burst detections in one frame."""
        manager = self.fragment_manager
        if manager is None:
            return
        if not hasattr(self, "_frame_fragment_stats"):
            self._reset_frame_fragment_event_stats()
        stats = self._frame_fragment_stats
        stats["detect_calls"] += 1

        sum_fields = (
            "new_detached_nodes",
            "phase_candidate_patches",
            "phase_approved_patches",
            "phase_rejected_patches",
            "phase_approved_nodes",
            "open_release_patches",
            "open_release_nodes",
            "impact_closure_patches",
            "impact_closure_nodes",
        )
        for field in sum_fields:
            stats[field] += int(getattr(manager, f"last_{field}", 0))

        max_fields = (
            "phase_approval_score_max",
            "phase_approval_cvol_max",
            "phase_approval_gate_max",
            "birth_phase_score",
            "birth_cvol_max",
            "birth_phase_gate_max",
            "open_release_score_max",
            "impact_closure_score_max",
        )
        for field in max_fields:
            stats[field] = max(
                float(stats[field]),
                float(getattr(manager, f"last_{field}", 0.0)),
            )

        approved = int(getattr(manager, "last_phase_approved_patches", 0))
        if approved > 0:
            stats["phase_approval_score_weighted_sum"] += (
                float(getattr(manager, "last_phase_approval_score_mean", 0.0))
                * approved
            )
        stats["pseudo_thickness_mass"] += float(
            getattr(manager, "last_pseudo_thickness_mass", 0.0)
        )
        stats["birth_phase_approved"] = bool(
            stats["birth_phase_approved"]
            or getattr(manager, "last_birth_phase_approved", False)
        )

    def _frame_fragment_stat(self, name: str, fallback=0):
        stats = getattr(self, "_frame_fragment_stats", None)
        if stats is None:
            return fallback
        return stats.get(name, fallback)

    # ================================================================
    # Main frame update
    # ================================================================
