#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
(c) 2023 Abhisek Maiti
This code is licensed under MIT license
For more details about license visit https://mit-license.org/license.txt
"""

import json
import torch
import argparse
import traceback
import numpy as np
import rasterio as rio
from einops import pack
from pathlib import Path
from einops import rearrange
from threading import Thread
from rich.progress import Progress
from rasterio.windows import Window
from multiprocessing.queues import Empty
from onnx2torch.converter import convert
from torch.multiprocessing import Manager
import torch.multiprocessing as multiprocessing


def predict(
    key: str,
    img1: Path,
    img2: Path,
    dst_dir: Path,
    block_height: int,
    block_width: int,
    lambda_wave: float,
    theta: float,
    model_path: str,
    device_id: str,
    task_id: int,
    progress_queue
):
    try:
        with rio.open(img1, "r") as src1, rio.open(img2, "r") as src2:
            canvas = Window(
                col_off=0, row_off=0, width=src1.width, height=src1.height
            )
            markers = np.meshgrid(
                np.arange(0, src1.height, block_height),
                np.arange(0, src1.width, block_width)
            )
            markers = np.stack(markers, -1).reshape(-1, len(markers)).tolist()
            meta = src1.meta.copy()
            meta["count"] = 2
            meta["nodata"] = np.nan
            meta["dtype"] = np.float32

            device = torch.device(device_id)
            model = convert(model_path).to(device=device, dtype=torch.float32)
            progress_queue.put(
                item={
                    "description": f"{key}:",
                    "task_id": task_id,
                    "total": len(markers),
                    "completed": 0
                },
                block=True,
                timeout=1
            )
            progress_queue.put(item=(task_id, True), block=True, timeout=1)
            with rio.open((dst_dir / f"{key}{img1.suffix}"), "w", **meta) as dst:
                for (roff, coff) in markers:
                    win = Window(
                        col_off=coff,
                        row_off=roff,
                        width=block_width,
                        height=block_height
                    ).intersection(canvas)
                    tile1 = rearrange(
                        src1.read(
                            window=win, masked=True, boundless=False
                        ), "c h w -> (h w) c"
                    )
                    if isinstance(tile1.mask, (bool, np.bool_)):
                        tile1.mask = np.full(
                            shape=tile1.shape,
                            fill_value=False,
                            dtype=bool
                        )

                    tile2 = rearrange(
                        src2.read(
                            window=win,
                            masked=True,
                            boundless=False),
                        "c h w -> (h w) c"
                    )
                    if isinstance(tile1.mask, (bool, np.bool_)):
                        tile2.mask = np.full(
                            shape=tile2.shape,
                            fill_value=False,
                            dtype=bool
                        )

                    valid_mask = np.logical_not(
                        np.any(
                            np.logical_or(tile1.mask, tile2.mask),
                            axis=-1,
                            keepdims=False
                        )
                    )

                    pred_tile = np.full(
                        shape=((win.height*win.width),meta["count"]),
                        fill_value=meta["nodata"], dtype=meta["dtype"]
                    )
                    if valid_mask.sum() > 0:
                        tile1 = tile1[valid_mask]
                        tile2 = tile2[valid_mask]
                        tile, _ = pack(
                            (
                                ((tile1 + tile2) / 2)**0.5,
                                ((tile1 - tile2) / 2)**0.5
                            ), "n *"
                        )
                        positive = np.all((tile > 0), axis=-1, keepdims=False)
                        valid_mask[valid_mask] = positive
                        tile = tile[positive]
                        tile = 10 * np.log10(tile)
                        tile = np.concatenate(
                            (
                                np.full(
                                    shape=(tile.shape[0], 1),
                                    fill_value=lambda_wave,
                                    dtype=tile.dtype
                                ),
                                np.full(
                                    shape=(tile.shape[0], 1),
                                    fill_value=theta,
                                    dtype=tile.dtype
                                ),
                                tile
                            ),
                            axis=-1
                        )
                        if positive.sum() > 0:
                            tile = torch.tensor(
                                tile, dtype=torch.float32, device=device
                            )
                            with torch.no_grad():
                                pred = model(
                                    tile
                                ).detach().clone().cpu().numpy().astype(
                                    meta["dtype"]
                                )
                                pred_tile[valid_mask] = pred
                            pred_tile = rearrange(
                                pred_tile,
                                "(h w) c -> c h w",
                                h=win.height,
                                w=win.width
                            )
                            dst.write(arr=pred_tile, window=win)
                    progress_queue.put(
                        item={
                            "task_id": task_id,
                            "advance": 1,
                            "visible": True,
                            "refresh": True
                        },
                        block=True,
                        timeout=1
                    )
        progress_queue.put(item=(task_id, False), block=True, timeout=1)
        return "Success!"
    except Exception as exception:
        return ''.join(traceback.format_exception(exception))


def manage_progress(progress, queue, main_id):
    main_task = progress.tasks[main_id]
    while main_task.completed < main_task.total:
        try:
            token = queue.get(block=True, timeout=10)
            if isinstance(token, tuple):
                task_id, flag = token
                if flag:
                    progress.start_task(task_id)
                else:
                    progress.update(task_id=main_id, advance=1, refresh=True)
            else:
                progress.update(**token)
            queue.task_done()
        except Empty:
            return False
    return True

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(
        description=(
            "Predict permitivity map from radar backscatter images.\n" +
            "Takes a JSON confoguartion file as commandline argument"
        )
    )
    parser.add_argument(
        '-c', '--conf',
        metavar='Configuration File',
        action='store',
        type=str,
        required=True,
        dest='conf_path',
        help='Specify the path of the configuration file.'
    )
    parser.add_argument(
        '-d', '--debug',
        metavar='Debug Log File',
        action='store',
        type=str,
        default=None,
        required=False,
        dest='log_path',
        help='Specify the path of the debug log file.'
    )
    args = parser.parse_args()
    conf_path = Path(args.conf_path)
    with open(conf_path, "r") as conf_src:
        conf = json.load(conf_src)
    block_height = conf["block_height"]
    block_width = conf["block_width"]
    lambda_wave = conf["lambda_wave"]
    theta = np.deg2rad(conf["theta"])
    input_images = conf["image_sets"]

    dst_dir = Path(conf["dst_dir"])
    dst_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    model_path = conf["model_path"]

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_ids = [
            f"cuda:{i}"
            for i in range(device_count)
        ]
    else:
        device_count = 1
        models = ['cpu']

    with Progress() as progress:
        pool = multiprocessing.Pool(processes=device_count)
        progress_queue = Manager().Queue()
        status = dict()
        total_count = len(input_images)

        main_id = progress.add_task(
            description="Total Progress:",
            start=True,
            completed=0,
            total=total_count,
            visible=True
        )

        progress_thread = Thread(
            target=manage_progress,
            kwargs={
                "progress": progress,
                "queue": progress_queue,
                "main_id": main_id
            }
        )
        progress_thread.daemon = True
        progress_thread.start()

        for i, (key, img_dict) in enumerate(input_images.items()):
            current_task = progress.add_task(
                description="",
                start=True,
                total=0,
                completed=0,
                visible=False
            )
            status[key] = pool.apply_async(
                func=predict,
                kwds={
                    "key": key,
                    "img1": Path(img_dict["S1"]),
                    "img2": Path(img_dict["S2"]),
                    "dst_dir": dst_dir,
                    "block_height": block_height,
                    "block_width": block_width,
                    "lambda_wave": lambda_wave,
                    "theta": theta,
                    "model_path": model_path,
                    "device_id": device_ids[i%device_count],
                    "task_id": current_task,
                    "progress_queue": progress_queue
                }
            )
        status = {k: p.get() for k, p in status.items()}

        pool.close()
        pool.join()

        progress_queue.join()
        progress_thread.join()

        if args.log_path:
            with open(Path(args.log_path), "w") as log_file:
                json.dump(status, log_file, indent=4)
