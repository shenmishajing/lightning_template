import json
import os
import time

import numpy as np
import torch
from pandas import DataFrame

from lightning_template.utils.visualization import draw_line_chart


def cuda_sync():
    torch.cuda.synchronize()


def visualize_speed_benchmark_res(result, output_path, main_arg_name, save_table):
    index = sorted(result)
    data = {}
    std_data = {}
    for i, ind in enumerate(index):
        for name, d in result[ind].items():
            if name not in data:
                data[name] = [0 for _ in range(i)]
            elif len(data[name]) < i:
                data[name].extend([0 for _ in range(i - len(data[name]))])
            data[name].append(np.mean(d))

            if name not in std_data:
                std_data[name] = [0 for _ in range(i)]
            elif len(std_data[name]) < i:
                std_data[name].extend([0 for _ in range(i - len(std_data[name]))])
            std_data[name].append(np.std(d))

    data = DataFrame(data, index=index)
    std_data = DataFrame(std_data, index=index)

    if save_table:
        data.to_csv(os.path.join(output_path, "mean.csv"))
        std_data.to_csv(os.path.join(output_path, "std.csv"))

    draw_line_chart(
        data,
        std_data,
        x_label=main_arg_name,
        y_label="duration (s)",
        save_path=os.path.join(output_path, "result.pdf"),
    )


def speed_benchmark(
    funcs,
    args,
    pre_func=None,
    post_func=None,
    repeat=3,
    check_res_func=None,
    root_path="work_dir/speed_benchmark",
    experiment_name=None,
    save_json=True,
    save_table=True,
    draw=True,
):
    if not isinstance(funcs, list):
        funcs = [funcs]

    if not isinstance(args["data"], list):
        args["data"] = [args["data"]]

    if pre_func is None:
        pre_func = cuda_sync
    if post_func is None:
        post_func = cuda_sync

    if experiment_name is not None:
        root_path = os.path.join(root_path, experiment_name)
    os.makedirs(root_path, exist_ok=True)

    result = {}
    for data in args["data"]:
        result[data[args["main_arg_name"]]] = {}
        except_res = None
        for func in funcs:
            result[data[args["main_arg_name"]]][func.__name__] = []
            cur_res = result[data[args["main_arg_name"]]][func.__name__]
            for _ in range(repeat):
                pre_func()
                duration = time.perf_counter()
                func_res = func(**data)
                post_func()
                duration = time.perf_counter() - duration

                cur_res.append(duration)

                if check_res_func is not None:
                    if except_res is None:
                        except_res = func_res
                    else:
                        if not check_res_func(except_res, func_res):
                            print(
                                f'result from func {func.__name__} is not correct on data with {data[args["main_arg_name"]]} = {data[args["main_arg_name"]]}'
                            )

    if save_json:
        json.dump(result, open(os.path.join(root_path, "result.json"), "w"))

    if draw:
        visualize_speed_benchmark_res(
            result, root_path, args["main_arg_name"], save_table
        )
