# Recommend Config File Structure

Config files are suggested to be put under `configs` folder with the following structure.

```
configs
|-- datasets
|   `-- <dataset_name>
|       |-- <dataset_name>_<task_type1>.yaml
|       |-- <dataset_name>_<task_type2>.yaml
|       `-- <dataset_name>_<task_type3>.yaml
|-- models
|   `-- <model_name>
|       |-- <model_name>_<other>_<model>_<info1>.yaml
|       |-- <model_name>_<other>_<model>_<info2>.yaml
|       `-- <model_name>_<other>_<model>_<info3>.yaml
|-- runs
|   `-- <model_name>
|       |-- <model_info>_<batch_size>_<scedule_info>_<dataset_info1>.yaml
|       |-- <model_info>_<batch_size>_<scedule_info>_<dataset_info2>.yaml
|       `-- <model_info>_<batch_size>_<scedule_info>_<dataset_info3>.yaml
|-- schedules
|   |-- <scedule_info1>.yaml
|   |-- <scedule_info2>.yaml
|   `-- <scedule_info3>.yaml
`-- default_runtime.yaml
```

## Usage of config files

Every config file under `configs/runs` is a complete config file, so you can run an experiment with just a config file from `configs/runs` following `--config` flag of lightning CLI. For example, to fit an example model on the example dataset using 8 gpus with batch size 2 on every gpu and 1x schedule, you may run

```bash
cli fit --config configs/runs/example_model/example_model_8xb2_1x_example_dataset.yaml
```

But every other config file is just a part of a complete config file, you should use them by combining them with each other or write a complete config file under `configs/runs` using `__base__` to inherit from them. For details on inheriting config files, see [deep_update](deep_update.md).
