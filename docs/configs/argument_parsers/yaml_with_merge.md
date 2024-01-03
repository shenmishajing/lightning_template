# Yaml with merge Argument Parser

## Introduction ##

The default parser for lightning CLI and a yaml parser with the [deep update](../deep_update.md) feature.

## Usage ##

Use the `--config` flag of lightning CLI to load a yaml config file with this parser. Support `__base__` and all [deep update](../deep_update.md) keywords, for more details of [deep update](../deep_update.md), see [deep update doc](../deep_update.md).

## Config inherit ##

Use the `__base__` keyword to inherit config files. The value of it must be `str` or `List[str]`, which is a relative path of the config file to inherit and can appear at any dict level.

For example (using yaml to present python dict).

```yaml
# configs/model/example.yaml
config:
    A:
        abc: 1
    B:
        a: d
    C:
        A: a
        C: c

# configs/model/test.yaml
config:
    A:
        abc: 1
    B:
        b: e
    C:
        B: b
        C: d

# configs/runs/example.yaml
__base__: ../model/example.yaml
config:
    B:
        c: b
    C:
        __base__: [ [../model/test.yaml, config.C ] ]
        D: f
    D:
        __base__: [ [../model/test.yaml, config.C ] ]
        C: f

# result
config:
    A:
        abc: 1
    B:
        a: d
        c: b
    C:
        A: a
        B: b
        C: d
    D:
        B: b
        C: f
```
## Yaml import ##

Yaml supports anchor and alias features, but it's not convenient sometimes, especially when you want to alias a part of key-value pairs in a dict. Therefore, you can use the `__import__` keyword to create the anchor first and use it in the config file, while the context under the `__import__` will be ignored during the parsing. Carefully, the `__import__` keyword is only supported at the top level of the config file.

For example (using yaml to present python dict).

```yaml
__import__:
    # all config under __import__ will be ignored
    # this part is only used for creating anchors
    import_train_img_scale_kwargs: &import_train_img_scale_kwargs
        img_scale: !!python/tuple [ 512, 512 ]
        keep_ratio: false
    import_test_img_scale_kwargs: &import_test_img_scale_kwargs
        img_scale: !!python/tuple [ 128, 128 ]
        keep_ratio: true

dataset_cfg:
    pipeline:
        train:
            -   type: Resize
                <<: *import_train_img_scale_kwargs
        test:
            -   type: Resize
                <<: *import_test_img_scale_kwargs
```

```yaml

dataset_cfg:
    __import__:
        # unsupported __import__ keyword not at top level!
        # will not be ignored!
        import_train_img_scale_kwargs: &import_train_img_scale_kwargs
            img_scale: !!python/tuple [ 512, 512 ]
            keep_ratio: false
        import_test_img_scale_kwargs: &import_test_img_scale_kwargs
            img_scale: !!python/tuple [ 128, 128 ]
            keep_ratio: true
    pipeline:
        train:
            -   type: Resize
                <<: *import_train_img_scale_kwargs
        test:
            -   type: Resize
                <<: *import_test_img_scale_kwargs
```

## All keywords ##

| keyword       | value                                                                                                                                                                                             | effect                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `__base__`    | `str` or `list[str]` or `list[[str, str]]`,(each `str` should be a relative path from current config file, when there are two str, the second one will be the key (using `.` to split) to import) | Merge every config one by one, current last.                                                          |
| `__import__`  | Any                                                                                                                                                                                               | Just delete this, for convenience of reference in yaml                                                |
| `__delete__`  | `True` or `str,int` or `list[str,int]`,`True` for delete all keys from other config, `str,int` only delete the specific key (for dict) or index (for list)                                        | Delete some part of config from other.                                                                |
| `change_item` | `list[[index, item]]`,used only when merge list                                                                                                                                                   | Add ability of merg list, change the `list[index]` from other to `item`                               |
| `insert_item` | `list[[index, item, (extend)]]`,used only when merge list                                                                                                                                         | Add ability of merg list, insert item to the `list` at `index`, extend=True if insert a list of items |
| `pre_item`    | `Any`or `list[Any]`,used only when merge list                                                                                                                                                     | Add ability of merg list, add the value in the start of the list from other to item                   |
| `post_item`   | `Any`or `list[Any]`,used only when merge list                                                                                                                                                     | Add ability of merg list, add the value in the end of the list from other to item                     |
