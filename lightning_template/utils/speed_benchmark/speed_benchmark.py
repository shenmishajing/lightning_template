import json
import os
import time

import numpy as np
import torch
from pandas import DataFrame

from lightning_template.utils.visualization import draw_line_chart


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


def check_results(x, y):
    if type(x) != type(y):
        return False

    if isinstance(x, tuple | list):
        return all([check_results(x[i], y[i]) for i in range(len(x))])
    if isinstance(x, dict):
        if set(x.keys()) != set(y.keys()):
            return False
        return all([check_results(x[k], y[k]) for k in x])

    if isinstance(x, np.ndarray):
        return np.allclose(x, y)
    if isinstance(x, torch.Tensor):
        return torch.allclose(x, y)
    if isinstance(x, float):
        return np.isclose(x, y)

    return x == y or x is y


def speed_benchmark(
    funcs,
    args,
    pre_func=None,
    post_func=None,
    repeat=3,
    num=1,
    warmup=1,
    check_result=True,
    check_result_func=None,
    root_path="work_dirs/speed_benchmark",
    experiment_name=None,
    save_json=True,
    save_table=True,
    draw=True,
):
    if not isinstance(funcs, list):
        funcs = [funcs]

    if not isinstance(args["data"], dict):
        args["data"] = {d[args["main_arg_name"]]: d for d in args["data"]}

    if pre_func is None:
        pre_func = torch.cuda.synchronize
    if post_func is None:
        post_func = torch.cuda.synchronize

    if check_result and check_result_func is None:
        check_result_func = check_results

    if experiment_name is not None:
        root_path = os.path.join(root_path, experiment_name)
    os.makedirs(root_path, exist_ok=True)

    result = {}
    for main_arg, data in args["data"].items():
        result[main_arg] = {}
        except_res = None
        for func in funcs:
            result[main_arg][func.__name__] = []
            cur_res = result[main_arg][func.__name__]

            for _ in range(warmup):
                func(**data)

            for _ in range(repeat):
                pre_func()
                duration = time.perf_counter()
                for _ in range(num):
                    func_res = func(**data)
                post_func()
                duration = time.perf_counter() - duration
                duration /= num

                cur_res.append(duration)

            if check_result_func is not None:
                if except_res is None:
                    except_res = func_res
                else:
                    if not check_result_func(except_res, func_res):
                        print(
                            f'result from func {func.__name__} is not correct on data with {args["main_arg_name"]} = {main_arg}'
                        )

    if save_json:
        json.dump(result, open(os.path.join(root_path, "result.json"), "w"))

    if draw:
        visualize_speed_benchmark_res(
            result, root_path, args["main_arg_name"], save_table
        )
