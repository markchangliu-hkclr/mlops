import os
import shutil
from pathlib import Path
from typing import Literal


def tag_from_dirname(
    dataset_dir: str,
    raw_root: str,
    restore_raw_root: bool,
    mode: Literal["w", "a"]
) -> None:
    assert mode in ["w", "a"]

    dataset_raw_label_dir = os.path.join(dataset_dir, "raw_labels")
    batchnames = os.listdir(dataset_raw_label_dir)
    batchnames.sort()

    for bn in batchnames:
        raw_batch_root = os.path.join(raw_root, bn)
        casenames = os.listdir(raw_batch_root)
        casenames.sort()

        ds_tag_dir = os.path.join(dataset_raw_label_dir, "tags")

        if not os.path.exists(ds_tag_dir):
            os.makedirs(ds_tag_dir)

        for cn in casenames:
            tags = cn.split("_")
            tags = [t.strip() for t in tags]

            raw_case_dir = os.path.join(raw_batch_root, cn)
            filenames = os.listdir(raw_case_dir)
            filenames.sort()

            for fn in filenames:
                if not fn.endswith((".png", ".jpg", ".jpeg")):
                    continue
                
                src_img_p = os.path.join(raw_case_dir, fn)
                img_stem = Path(fn).stem
                tag_name = f"{img_stem}.txt"
                ds_tag_p = os.path.join(ds_tag_dir, tag_name)

                with open(ds_tag_p, mode) as f:
                    for t in tags:
                        f.write(f"{t}\n")
        
                if restore_raw_root:
                    dst_img_p = os.path.join(raw_batch_root, fn)
                    shutil.move(src_img_p, dst_img_p)
            
            if restore_raw_root:
                shutil.rmtree(raw_case_dir)