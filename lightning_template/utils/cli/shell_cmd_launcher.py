import os
import subprocess
import time
from collections import deque
from string import Template


def iter_arg_dict(arg_dict, keys=None, cur_arg_dict=None):
    if keys is None:
        keys = list(arg_dict.keys())

    if cur_arg_dict is None:
        cur_arg_dict = {}

    if len(keys) == 0:
        yield cur_arg_dict
    else:
        key = keys[0]
        for arg_value in arg_dict[key]:
            cur_arg_dict[key] = arg_value
            yield from iter_arg_dict(arg_dict, keys[1:], cur_arg_dict)
            cur_arg_dict.pop(key)


def single_cmd_launcher(
    cmd: str,
    log_dir: str = None,
    name: str = "",
    num: int = 1,
    parallel_num: int = 1,
    sleep_time: float = 0,
    **kwargs,
):
    parallel_num = max(1, parallel_num)
    tasks = deque(parallel_num)
    for num_ind in range(num):
        print(f"running cmd: {cmd}, num: {num_ind}")

        if num > 1:
            if name == "":
                cur_name = f"{num_ind}"
            else:
                cur_name = f"{name}_{num_ind}"
        else:
            cur_name = name

        if log_dir is not None:
            stdout = open(os.path.join(log_dir, f"{cur_name}.log"), "w")
        elif parallel_num == 1:
            stdout = subprocess.PIPE
        else:
            stdout = None

        t = subprocess.Popen(
            cmd, **kwargs, stdout=stdout, stderr=subprocess.STDOUT, shell=True
        )

        tasks.append(t)

        if tasks.full():
            tasks.popleft().wait()

        if sleep_time:
            time.sleep(sleep_time)

    return list(tasks)


def shell_cmd_launcher(
    cmd: str,
    arg_dict: dict[str, list] = None,
    log_dir: str = None,
    num: int = 1,
    **kwargs,
):
    if log_dir is not None:
        if num == 1:
            name = os.path.basename(log_dir)
            log_dir = os.path.dirname(log_dir)
        else:
            name = ""

        os.makedirs(log_dir, exist_ok=True)
    else:
        name = ""

    tasks = []
    if arg_dict is not None:
        for args in iter_arg_dict(arg_dict):
            cur_cmd = Template(cmd).substitute(args)
            keys = sorted(args.keys())
            name = "__".join([f"{key}_{args[key]}" for key in keys])
            tasks.extend(
                single_cmd_launcher(
                    cur_cmd, log_dir=log_dir, name=name, num=num, **kwargs
                )
            )
    else:
        tasks.extend(
            single_cmd_launcher(cmd, log_dir=log_dir, name=name, num=num, **kwargs)
        )

    for t in tasks:
        t.wait()
