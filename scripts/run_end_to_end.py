import os
import shutil

base_src = "data"
base_dst = "data/dataset_nn"

if os.path.exists(base_dst):
    shutil.rmtree(base_dst) and print(f"Removed existing directory: {base_dst}")

splits = ["train", "validation", "test"]

angles = ["angles_45", "angles_60", "angles_90", "angles_180"]

def angle_fmt(a):
    num = a.split("_")[1]
    return f"angle_{int(num):03d}"

for split in splits:
    dst_split = "val" if split == "validation" else split

    for ang in angles:
        ang_dst = angle_fmt(ang)

        # SOURCE PATHS
        sin_src = os.path.join(base_src, "sinogram_corrupted", split, ang)
        tv_src = os.path.join(base_src, "reco", ang, split)

        # DEST PATHS
        sin_dst = os.path.join(base_dst, dst_split, "sinograms", ang_dst)
        tv_dst = os.path.join(base_dst, dst_split, "tv", ang_dst)

        os.makedirs(sin_dst, exist_ok=True)
        os.makedirs(tv_dst, exist_ok=True)

        # COPY sinograms
        if os.path.exists(sin_src):
            for f in os.listdir(sin_src):
                shutil.copy2(
                    os.path.join(sin_src, f),
                    os.path.join(sin_dst, f)
                )

        # COPY tv
        if os.path.exists(tv_src):
            for f in os.listdir(tv_src):
                shutil.copy2(
                    os.path.join(tv_src, f),
                    os.path.join(tv_dst, f)
                )

print("Done")