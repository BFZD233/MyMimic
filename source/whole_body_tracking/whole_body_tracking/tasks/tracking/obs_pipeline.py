from __future__ import annotations

from abc import ABC, abstractmethod

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp

# Legacy terms currently defined in `ObservationsCfg.PolicyCfg`.
LEGACY_POLICY_TERMS = (
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
)

# Legacy terms currently defined in `ObservationsCfg.PrivilegedCfg`.
LEGACY_CRITIC_TERMS = (
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "body_pos",
    "body_ori",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
)

# Extra policy terms reserved for non-legacy builders.
TWIST2_POLICY_TERMS = ("twist2_motion", "twist2_proprio", "twist2_future")
TWIST2_1432_POLICY_TERMS = ("twist2_1432_motion", "twist2_1432_proprio", "twist2_1432_future")
NON_LEGACY_POLICY_TERMS = TWIST2_POLICY_TERMS + TWIST2_1432_POLICY_TERMS


@configclass
class ObsPipelineCfg:
    """Single-source config for actor/critic observation pipeline selection."""

    mode: str = "legacy"
    actor_mode: str | None = None
    critic_mode: str | None = None

    include_history: bool = False
    history_len: int = 10

    include_future: bool = False
    future_steps: tuple[int, ...] = (0,)
    future_include_joint_pos: bool = True
    future_include_joint_vel: bool = True

    def __post_init__(self):
        self.mode = self.mode.lower()
        self.actor_mode = self.actor_mode.lower() if self.actor_mode is not None else None
        self.critic_mode = self.critic_mode.lower() if self.critic_mode is not None else None

        if self.history_len < 0:
            raise ValueError(f"`history_len` must be >= 0, got: {self.history_len}")

        self.future_steps = tuple(int(step) for step in self.future_steps)
        if len(self.future_steps) == 0:
            self.future_steps = (0,)
        if any(step < 0 for step in self.future_steps):
            raise ValueError(f"`future_steps` must be non-negative, got: {self.future_steps}")

        if self.include_future and not (self.future_include_joint_pos or self.future_include_joint_vel):
            raise ValueError("`include_future=True` requires at least one of joint_pos/joint_vel features enabled.")

    def resolved_actor_mode(self) -> str:
        return self.actor_mode if self.actor_mode is not None else self.mode

    def resolved_critic_mode(self) -> str:
        return self.critic_mode if self.critic_mode is not None else self.mode


def _set_group_term_history(group: ObsGroup, term_names: tuple[str, ...], history_len: int) -> None:
    for term_name in term_names:
        term_cfg = getattr(group, term_name, None)
        if isinstance(term_cfg, ObsTerm):
            term_cfg.history_length = history_len
            term_cfg.flatten_history_dim = True


def _disable_terms(group: ObsGroup, term_names: tuple[str, ...]) -> None:
    for term_name in term_names:
        if hasattr(group, term_name):
            setattr(group, term_name, None)


class ObservationBuilder(ABC):
    name: str

    @abstractmethod
    def configure_actor(self, group: ObsGroup, cfg: ObsPipelineCfg) -> None:
        """Configure actor (policy) observation group."""

    @abstractmethod
    def configure_critic(self, group: ObsGroup, cfg: ObsPipelineCfg) -> None:
        """Configure critic observation group."""


class LegacyObservationBuilder(ObservationBuilder):
    name = "legacy"

    def configure_actor(self, group: ObsGroup, cfg: ObsPipelineCfg) -> None:
        _disable_terms(group, NON_LEGACY_POLICY_TERMS)

        # Optional, config-driven history over legacy actor terms.
        history_len = cfg.history_len if cfg.include_history else 0
        group.history_length = None
        group.flatten_history_dim = True
        _set_group_term_history(group, LEGACY_POLICY_TERMS, history_len)

        # Optional future branch appended to legacy observations.
        if cfg.include_future:
            group.twist2_future = ObsTerm(
                func=mdp.motion_future_joint_obs,
                params={
                    "command_name": "motion",
                    "future_steps": cfg.future_steps,
                    "include_joint_pos": cfg.future_include_joint_pos,
                    "include_joint_vel": cfg.future_include_joint_vel,
                },
            )
            group.twist2_future.history_length = 0
        else:
            group.twist2_future = None

    def configure_critic(self, group: ObsGroup, cfg: ObsPipelineCfg) -> None:
        # Keep critic default unchanged in legacy mode.
        group.history_length = None
        group.flatten_history_dim = True
        _set_group_term_history(group, LEGACY_CRITIC_TERMS, 0)


class Twist2LikeObservationBuilder(ObservationBuilder):
    name = "twist2_like"

    def configure_actor(self, group: ObsGroup, cfg: ObsPipelineCfg) -> None:
        # Disable legacy terms for a clean semantic split.
        _disable_terms(group, LEGACY_POLICY_TERMS + TWIST2_1432_POLICY_TERMS)

        # Build actor terms with explicit semantic blocks.
        group.twist2_motion = ObsTerm(func=mdp.twist2_like_motion_obs, params={"command_name": "motion"})
        group.twist2_proprio = ObsTerm(func=mdp.twist2_like_proprio_obs)

        if cfg.include_future:
            group.twist2_future = ObsTerm(
                func=mdp.motion_future_joint_obs,
                params={
                    "command_name": "motion",
                    "future_steps": cfg.future_steps,
                    "include_joint_pos": cfg.future_include_joint_pos,
                    "include_joint_vel": cfg.future_include_joint_vel,
                },
            )
            group.twist2_future.history_length = 0
        else:
            group.twist2_future = None

        group.history_length = None
        group.flatten_history_dim = True
        history_len = cfg.history_len if cfg.include_history else 0
        _set_group_term_history(group, ("twist2_motion", "twist2_proprio"), history_len)

    def configure_critic(self, group: ObsGroup, cfg: ObsPipelineCfg) -> None:
        # Phase-1 choice: keep critic in legacy structure to minimize intrusiveness.
        group.history_length = None
        group.flatten_history_dim = True
        _set_group_term_history(group, LEGACY_CRITIC_TERMS, 0)


class Twist21432ObservationBuilder(ObservationBuilder):
    name = "twist2_1432"

    def configure_actor(self, group: ObsGroup, cfg: ObsPipelineCfg) -> None:
        # Disable other actor term families first.
        _disable_terms(group, LEGACY_POLICY_TERMS + TWIST2_POLICY_TERMS)

        # TWIST2-compatible actor composition:
        #   obs = current(127) + history(10 * 127) + future(35) = 1432
        # In IsaacLab term-history semantics, "current + 10 history" maps to history_length=11.
        group.twist2_1432_motion = ObsTerm(func=mdp.twist2_1432_motion_obs, params={"command_name": "motion"})
        group.twist2_1432_proprio = ObsTerm(func=mdp.twist2_1432_proprio_obs)

        if cfg.include_future:
            group.twist2_1432_future = ObsTerm(
                func=mdp.twist2_1432_future_motion_obs,
                params={
                    "command_name": "motion",
                    "future_steps": cfg.future_steps,
                },
            )
            group.twist2_1432_future.history_length = 0
            group.twist2_1432_future.flatten_history_dim = True
        else:
            group.twist2_1432_future = None

        group.history_length = None
        group.flatten_history_dim = True
        history_len = (cfg.history_len + 1) if cfg.include_history else 0
        _set_group_term_history(group, ("twist2_1432_motion", "twist2_1432_proprio"), history_len)

    def configure_critic(self, group: ObsGroup, cfg: ObsPipelineCfg) -> None:
        # Keep critic in legacy structure for now.
        group.history_length = None
        group.flatten_history_dim = True
        _set_group_term_history(group, LEGACY_CRITIC_TERMS, 0)


OBSERVATION_BUILDERS: dict[str, ObservationBuilder] = {
    LegacyObservationBuilder.name: LegacyObservationBuilder(),
    Twist2LikeObservationBuilder.name: Twist2LikeObservationBuilder(),
    Twist21432ObservationBuilder.name: Twist21432ObservationBuilder(),
}


def _get_builder(mode: str) -> ObservationBuilder:
    if mode not in OBSERVATION_BUILDERS:
        supported = ", ".join(sorted(OBSERVATION_BUILDERS))
        raise ValueError(f"Unsupported observation mode: '{mode}'. Supported modes: {supported}")
    return OBSERVATION_BUILDERS[mode]


def apply_observation_pipeline(env_cfg) -> None:
    """Apply actor/critic observation builders based on env_cfg.obs_pipeline."""
    pipeline_cfg: ObsPipelineCfg = env_cfg.obs_pipeline
    actor_builder = _get_builder(pipeline_cfg.resolved_actor_mode())
    critic_builder = _get_builder(pipeline_cfg.resolved_critic_mode())

    actor_builder.configure_actor(env_cfg.observations.policy, pipeline_cfg)
    critic_builder.configure_critic(env_cfg.observations.critic, pipeline_cfg)
