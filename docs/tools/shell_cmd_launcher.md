## Shell Command Launcher

A simple tool to launch shell command many times.

### Args

#### cmd

The command to run, which is a string. You can use `$<var_name>` or `${<var_name>}` to represent a var defined in arg_dict. For details, see the [template doc](https://docs.python.org/3/library/string.html#template-strings).

#### arg_dict

A dict of args to replace the var in cmd. The key is the var name, and the value is a list of the var value.

The command will run with all combinations of the var values.

#### log_dir

The dir to save the log file. If set, the output of each command will be saved in `log_dir/<var_name1>_<var_value1>__<var_name2>_<var_value2>...__<var_namen>_<var_valuen>.log`.

#### num

The number of times to run the same command. If set, each command will run `num` times. If `log_dir` is also set, the output of each command will be saved in `log_dir/<var_name1>_<var_value1>__<var_name2>_<var_value2>...__<var_namen>_<var_valuen>_<num>.log`.

#### sleep_time

The sleep time between each command can be `None` or `int`, `None` by default. If set to `None`, all commands will be executed serially. If set to `int`, we will sleep `sleep_time` seconds between each command, and all commands will be executed in parallel.

#### other args

Other args will be passed to `subprocess.Popen` as kwargs. For details, see the [doc](https://docs.python.org/3/library/subprocess.html#popen-constructor).
