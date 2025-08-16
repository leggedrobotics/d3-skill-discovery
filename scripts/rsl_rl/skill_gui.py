"""skill_gui_modular.py - fully modular GUI for skill control."""

from __future__ import annotations

import threading
import torch
from typing import Dict, List, Literal, Tuple

import dearpygui.dearpygui as dpg

SkillType = Literal["diayn", "metra"]


def _default_values(dim: int, kind: SkillType) -> List[float]:
    if kind == "diayn":
        return [1.0 / dim] * dim
    return [0.0] * dim  # metra


class SkillControlGUI:
    """GUI to control skill factors.
    For this GUI to work, the skill setup in the environment must be set up"""

    def __init__(self, gui_setup: Dict[str, Tuple[int, SkillType]]):
        if not gui_setup:
            raise ValueError("gui_setup cannot be empty")

        self.blocks: List[dict] = []
        value_size = 0
        for name, (dim, kind) in gui_setup.items():
            blk = dict(
                name=name,
                dim=dim,
                kind=kind,
                values=_default_values(dim, kind),
                axis_tags=[],  # slider/joystick widget IDs
                weight_tag=None,
                weight=0.0,
                diayn2_tag=None,  # ID of special single slider (dim‑2 diayn)
            )
            self.blocks.append(blk)
            value_size += dim

        self.extrinsic_weight = 1.0
        self._skill_size = value_size + 1 + len(self.blocks)
        self.skill = torch.zeros(self._skill_size, dtype=torch.float32)

        self._updating_weights = False

        self._rebuild_skill()

    def _assert_dims(self, blk):
        if len(blk["values"]) != blk["dim"]:
            raise RuntimeError(
                f"Block '{blk['name']}' value list length {len(blk['values'])} "
                f"does not equal declared dim {blk['dim']}!"
            )

    def _normalise_if_diayn(self, blk):
        if blk["kind"] != "diayn" or blk["dim"] == 2:
            # dim‑2 handled analytically by single‑slider logic
            return
        vals = torch.tensor(blk["values"], dtype=torch.float32)
        total = vals.sum().item()
        if total > 0:
            vals /= total
        blk["values"] = vals.tolist()

    def _renormalise_weights(self):
        if self._updating_weights:
            return
        self._updating_weights = True

        w = [self.extrinsic_weight] + [blk["weight"] for blk in self.blocks]
        norm = torch.linalg.vector_norm(torch.tensor(w)).item()
        if norm == 0.0:
            self.extrinsic_weight = 1.0
            for blk in self.blocks:
                blk["weight"] = 0.0
        else:
            scale = 1.0 / norm
            self.extrinsic_weight *= scale
            for blk in self.blocks:
                blk["weight"] *= scale

        dpg.set_value("extrinsic_weight", self.extrinsic_weight)
        for blk in self.blocks:
            dpg.set_value(blk["weight_tag"], blk["weight"])

        self._rebuild_skill()
        self._updating_weights = False

    def _rebuild_skill(self):
        offset = 0
        for blk in self.blocks:
            self._assert_dims(blk)
            dim = blk["dim"]
            self.skill[offset : offset + dim] = torch.tensor(blk["values"], dtype=torch.float32)
            offset += dim
        self.skill[offset] = self.extrinsic_weight
        offset += 1
        for blk in self.blocks:
            self.skill[offset] = blk["weight"]
            offset += 1

    def _axis_cb(self, sender, app_data, user_data):
        blk, start_idx = user_data
        blk["values"][start_idx] = float(app_data[0])
        blk["values"][start_idx + 1] = float(app_data[1])
        self._normalise_if_diayn(blk)
        self._rebuild_skill()

    def _single_cb(self, sender, app_data, user_data):
        blk, idx = user_data
        blk["values"][idx] = float(app_data)
        self._normalise_if_diayn(blk)
        self._rebuild_skill()

    def _diayn2_cb(self, sender, app_data, user_data):
        """Special callback for dim‑2 diayn (single slider)."""
        blk = user_data
        v = float(app_data)
        blk["values"][0] = v
        blk["values"][1] = 1.0 - v
        self._rebuild_skill()

    def _weight_cb(self, sender, app_data, user_data):
        blk = user_data
        blk["weight"] = float(app_data)
        self._renormalise_weights()

    def _extr_weight_cb(self, sender, app_data, user_data):
        self.extrinsic_weight = float(app_data)
        self._renormalise_weights()

    def _make_block_widgets(self, blk):
        name, dim, kind = blk["name"], blk["dim"], blk["kind"]
        lo, hi = (-1.0, 1.0) if kind == "metra" else (0.0, 1.0)

        dpg.add_separator()
        dpg.add_text(name.replace("_", " ").title())

        if kind == "diayn" and dim == 2:
            tag = f"{name}_mix"
            blk["diayn2_tag"] = tag
            blk["axis_tags"].append(tag)
            dpg.add_slider_float(
                tag=tag,
                min_value=0.0,
                max_value=1.0,
                default_value=blk["values"][0],
                label=f"{name}[0] / (1‑x)",
                user_data=blk,
                callback=self._diayn2_cb,
            )
        else:
            for i in range(0, dim, 2):
                size = 2 if i + 1 < dim else 1
                tag = f"{name}_axis{i//2}"
                blk["axis_tags"].append(tag)
                if size == 2:
                    dpg.add_slider_floatx(
                        tag=tag,
                        size=2,
                        min_value=lo,
                        max_value=hi,
                        default_value=blk["values"][i : i + 2],
                        label="" if dim == 2 else f"{name}[{i}:{i+2}]",
                        user_data=(blk, i),
                        callback=self._axis_cb,
                    )
                else:
                    dpg.add_slider_float(
                        tag=tag,
                        min_value=lo,
                        max_value=hi,
                        default_value=blk["values"][i],
                        label=f"{name}[{i}]",
                        user_data=(blk, i),
                        callback=self._single_cb,
                    )

        blk["weight_tag"] = f"{name}_weight"
        dpg.add_slider_float(
            tag=blk["weight_tag"],
            label=f"{name} weight",
            min_value=0.0,
            max_value=1.0,
            default_value=0.0,
            user_data=blk,
            callback=self._weight_cb,
        )

    def _reset(self):
        for blk in self.blocks:
            blk["values"] = _default_values(blk["dim"], blk["kind"])
            blk["weight"] = 0.0

            if blk["kind"] == "diayn" and blk["dim"] == 2:
                dpg.set_value(blk["diayn2_tag"], blk["values"][0])
            else:
                for axis_i, tag in enumerate(blk["axis_tags"]):
                    slice_start = axis_i * 2
                    vals = blk["values"][slice_start : slice_start + 2]
                    dpg.set_value(tag, vals if len(vals) == 2 else vals[0])

            dpg.set_value(blk["weight_tag"], 0.0)

        self.extrinsic_weight = 1.0
        self._renormalise_weights()

    def launch(self):
        dpg.create_context()
        dpg.create_viewport(title="Skill Control GUI", width=450, height=720)
        with dpg.window(label="Skill Control", width=430, height=700):
            for blk in self.blocks:
                self._make_block_widgets(blk)
            dpg.add_separator()
            dpg.add_text("Weights")
            dpg.add_slider_float(
                tag="extrinsic_weight",
                label="Extrinsic weight",
                min_value=0.0,
                max_value=1.0,
                default_value=self.extrinsic_weight,
                callback=self._extr_weight_cb,
            )
            for blk in self.blocks:
                dpg.move_item(blk["weight_tag"], parent=dpg.last_container())
            dpg.add_separator()
            dpg.add_button(label="Reset", callback=lambda s, a, u: self._reset())
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()


if __name__ == "__main__":
    # Example usage
    setup = {
        "metra_pos": (2, "metra"),
        "heading": (2, "diayn"),
        "base_height": (2, "diayn"),
        "roll_pitch": (4, "diayn"),
        "base_vel": (4, "diayn"),
    }
    gui = SkillControlGUI(setup)
    threading.Thread(target=gui.launch, daemon=True).start()

    import time

    counter = 0
    while True:
        # do something else
        # e.g. read skill tensor
        if counter % 1 == 0:
            print("Skill tensor:")
            print(gui.skill)
        time.sleep(0.1)
        counter += 1
