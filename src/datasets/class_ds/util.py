import os
import shutil
from tqdm.auto import tqdm
import h5py
from PIL import Image
import concurrent.futures
from uuid import uuid4
import json
import glob
import time
import gdown
import gzip
from ...util import download_s3_directory


def make_s3_ds(s3_uri, dir):
    download_s3_directory(s3_uri, dir, True)
    archives = ["annotations.tar.gz", "images.tar.gz"]
    for archive in archives:
        archive_path = os.path.join(dir, archive)
        shutil.unpack_archive(archive_path, dir)
        os.remove(archive_path)


def make_pcam_ds(g_url, data_dir):
    def gunzip_file(gzipped_file, output_file):
        with gzip.open(gzipped_file, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    def download_and_extract(data_dir):
        print("downloading...")
        gdown.download_folder(
            url=g_url,
            output=data_dir,
        )
        print("extracting from gz...")
        archives = glob.glob(f"{data_dir}/*.h5.gz")
        for src in archives:
            dst = src.rstrip(".gz")
            gunzip_file(src, dst)
            os.remove(src)

    def brake_dirs(root_dir):
        class_map = {0: "normal", 1: "abnormal"}

        def h52png(x_file_path: str, y_file_path, output_dir: str):
            assert x_file_path.endswith("_x.h5")
            assert y_file_path.endswith("_y.h5")

            # determine paths/directories
            split = x_file_path.split("_")[-2]
            if split == "valid":
                split = "val"
            img_dir = os.path.join(output_dir, "images")
            ant_dir = os.path.join(
                output_dir, "annotations", "classification", "splits"
            )
            ant_path = os.path.join(ant_dir, f"{split}.txt")
            for cls_nm in class_map.values():
                assert os.path.exists(os.path.join(img_dir, cls_nm))
            assert os.path.exists(ant_dir)

            x_file = h5py.File(x_file_path)
            y_file = h5py.File(y_file_path)

            img_paths = []
            for x, y in tqdm(
                zip(x_file["x"], y_file["y"]),
                total=len(x_file["x"]),
                desc=os.path.split(x_file_path)[1],
            ):
                img_rel_path = os.path.join(
                    class_map[int(y.flatten()[0])], f"{uuid4()}.png"
                )
                dst_path = os.path.join(img_dir, img_rel_path)
                x = Image.fromarray(x)
                x.save(dst_path)
                img_paths.append(os.path.join("images", img_rel_path))

            with open(ant_path, "w") as handler:
                handler.write("\n".join(img_paths))

            x_file.close()
            y_file.close()

            return 0

        def h52png_train_mask(
            mask_file_path: str, names_file_path: str, output_dir: str
        ):
            assert mask_file_path.endswith("train_mask.h5")
            assert os.path.exists(output_dir)

            with open(names_file_path) as handler:
                names = [
                    os.path.split(p)[1] for p in handler.read().strip().split("\n")
                ]
            h5_file = h5py.File(mask_file_path)
            h5_data = h5_file["mask"]

            assert len(h5_data) == len(names)

            for mask, nm in tqdm(
                zip(h5_data, names),
                total=len(h5_data),
                desc=os.path.split(mask_file_path)[1],
            ):
                dst_path = os.path.join(output_dir, nm)
                mask = Image.fromarray(mask[:, :, 0])
                mask.save(dst_path)

            h5_file.close()

        h5_files = glob.glob(f"{root_dir}/*.h5")
        pairs = []
        for split in ["train", "valid", "test"]:
            x = [f for f in h5_files if f.endswith(f"{split}_x.h5")][0]
            y = [f for f in h5_files if f.endswith(f"{split}_y.h5")][0]
            pairs.append((x, y))

        # make dirs
        img_dir = os.path.join(root_dir, "images")
        ant_dir = os.path.join(root_dir, "annotations", "classification", "splits")
        for cls_nm in class_map.values():
            os.makedirs(os.path.join(img_dir, cls_nm), exist_ok=True)
        os.makedirs(ant_dir, exist_ok=True)
        # export classes
        classes_file_path = os.path.join(
            root_dir, "annotations", "classification", "classes.json"
        )
        with open(classes_file_path, "w") as handler:
            json.dump(class_map, handler, indent=4)
        # move  metadata files
        other_dir = os.path.join(root_dir, "annotations", "other")
        os.makedirs(other_dir, exist_ok=True)
        meta_files = [p for p in glob.glob(f"{root_dir}/*.csv")]
        for src in meta_files:
            dst = src.replace(root_dir, other_dir)
            os.rename(src, dst)

        # parallel invocation of h52png
        print("extraing from h5...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for pair in pairs:
                futures.append(executor.submit(h52png, *pair, root_dir))
                time.sleep(0.5)  # make sure train starts logging first
            for future in concurrent.futures.as_completed(futures):
                future.result()

        train_mask_file_path = [f for f in h5_files if f.endswith("train_mask.h5")][0]
        train_mask_names_file_path = os.path.join(
            root_dir, "annotations", "classification", "splits", "train.txt"
        )
        train_masks_output_dir = os.path.join(
            root_dir, "annotations", "segmentation", "train"
        )
        os.makedirs(train_masks_output_dir, exist_ok=True)
        h52png_train_mask(
            train_mask_file_path, train_mask_names_file_path, train_masks_output_dir
        )

        # clean up old files
        print("cleaning up...")
        residuals = [p for p in glob.glob(f"{root_dir}/*") if os.path.isfile(p)]
        for r in residuals:
            os.remove(r)

    download_and_extract(data_dir)
    brake_dirs(data_dir)
