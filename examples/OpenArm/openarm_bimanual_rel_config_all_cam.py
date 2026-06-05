from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

_ACTION_CHUNK_LEN = 32  # typical GR00T chunk horizon

openarm_config = {
    # IMPORTANT: these keys must match the dataset's modality.json "video" keys,
    # NOT the real robot camera nicknames.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "wrist_right",
            "wrist_left",
            "ceiling",
            "head_left",
            "head_right",
        ],
    ),

    # Right-only state keys must match modality.json "state"
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "right_arm",
            "right_gripper",
            "left_arm",
            "left_gripper",
        ],
    ),

    # Right-only action keys must match modality.json "action"
    "action": ModalityConfig(
        delta_indices=list(range(_ACTION_CHUNK_LEN)),
        modality_keys=[
            "right_arm",
            "right_gripper",
            "left_arm",
            "left_gripper",
        ],
        # MUST be same length as modality_keys (2)
        # Choose ABSOLUTE or RELATIVE to match your dataset semantics.
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),

    # Keep consistent with SO100 pattern: "annotation.<key>"
    # Your dataset modality.json has annotation key: "human.action.task_description"
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.human.action.task_description",
        ],
    ),
}

register_modality_config(
    openarm_config,
    embodiment_tag=EmbodimentTag.OPENARM_BIMANUAL_REL_ALL_CAM,
)
