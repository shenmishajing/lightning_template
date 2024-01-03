# Deep Update

## Introduction ##

A method to merge two objects, named `source` and `override`, use `override` to modify `source` data.

## Usage ##

The `source` should be a dict or a list. The `override` must be a `dict`, or we will return the `override` directly.

### When the source is a dict ###

#### delete key ####

Use the `__delete__` keyword to delete keys in `source`, `True` for delete all, `str` or `List[str]` for specific keys.

For example (using yaml to present python dict).

```yaml
# source
config:
    A:
        abc: 1
    B:
        a: d
        b: e
    C:
        A: a
        B: b
        C: c

# override
config:
    A:
        __delete__: true
    B:
        __delete__: b
    C:
        __delete__: [ A, B ]

# result
config:
    A: {}
    B:
        a: d
    C:
        C: c
```

#### add key and modify key ####

Just write the key-value pair to add or modify context in the `source`.

For example (using yaml to present python dict).

```yaml
# source
config:
    A:
        abc: 1
    B:
        a: d
        b: e

# override
config:
    A:
        abc: 2
    B:
        c: c
    C:
        a: A

# result
config:
    A:
        abc: 2
    B:
        a: d
        b: e
        c: c
    C:
        a: A
```

### When the source is a list ###

#### delete item ####

Use the `__delete__` keyword to delete items in the `source`, `True` for delete all, `int` or `List[int]` for specific items, and negative int for counting from the end.

For example (using yaml to present python dict).

```yaml
# source
config:
    A:
        - abc
        - efg
    B:
        - 123
        - 234
    C: [ a, b, c ]

# override
config:
    A:
        __delete__: true
    B:
        __delete__: 0
    C:
        __delete__: [ 0, -1 ]

# result
config:
    A: []
    B:
        - 234
    C:
        - b
```

#### modify item ####

Use the `change_item` keyword to modify items in the `source`, the value of it must be `List[List[index, item]]`. It will change every item at the index to the corresponding item.

For example (using yaml to present python dict).

```yaml
# source
config:
    A:
        - abc
        - efg
    B: [ a, b, c ]

# override
config:
    A:
        change_item:
            - [ 0, A ]
    B:
        change_item:
            - [ -1, B ]
            - [ 0, C ]

# result
config:
    A:
        - A
        - efg
    B: [ C, b, B ]
```

#### add items ####

##### pre items #####

Use the `pre_item` keyword to add items to the beginning of the `source`, the value of it must be an item or a list. If the value is a list, every item in it will be added to the beginning of the `source`, otherwise, add the value to the beginning of the `source`.

For example (using yaml to present python dict).

```yaml
# source
config:
    A:
        - abc
        - efg
    B: [ a, b, c ]

# override
config:
    A:
        pre_item: A
    B:
        pre_item: [ B, C ]

# result
config:
    A:
        - A
        - abc
        - efg
    B:
        - B
        - C
        - a
        - b
        - c
```

##### post items #####

Use the `post_item` keyword to add items to the end of the `source`. The value of it must be an item or a list. If the value is a list, every item in it will be added to the end of the `source`, otherwise, we will add the value to the `source`.

For example (using yaml to present python dict).

```yaml
# source
config:
    A:
        - abc
        - efg
    B: [ a, b, c ]

# override
config:
    A:
        post_item: A
    B:
        post_item: [ B, C ]

# result
config:
    A:
        - abc
        - efg
        - A
    B:
        - a
        - b
        - c
        - B
        - C
```

##### insert items #####

Use the `insert_item` keyword to insert items to the `source`. The value of it must be `List[List[index, item]` or `List[index, item, extend]]`, we will insert the item or the list of items to the position specified by the index. If the index is bigger or equal to the `len` of the `source` or smaller than the `-len` of the `source`, this will work as `post_item` and `pre_item`. The `extend` is a bool to indicate whether the item is a list of items or not. If it is set to `True`, the elements of the item will be inserted one by one and if it is set to `False`, the item will be treated as an element and will be inserted to the `source`. You can omit it when it is False. This works well with `__delete_`_` and multiple inserts, the insert order and delete index will be calculated automatically.

For example (using yaml to present python dict).

```yaml
# source
config:
    A:
        - abc
        - efg
    B: [ a, b, c ]
    C: [ 1, 2, 3, 4 ]
    D: [ 1, 2, 3, 4 ]
    E: [ 1, 2, 3, 4 ]

# override
config:
    A:
        insert_item:
            - [ 0, A ]
            - [ 1, B ]
    B:
        insert_item:
            - [ -1, B ]
            - [ 1, [ 1, 2, 3 ], true ]
    C:
        insert_item:
            - [ -5, A ]
            - [ 4, B ]
            - [ 5, C ]
    D:
        __delete__: [ 1, 2 ]
        insert_item:
            - [ 0, A ]
            - [ 3, B ]
            - [ 1, [ C, D ], true ]
    E:
        __delete__: true
        insert_item:
            - [ 0, A ]
            - [ 3, B ]
            - [ 1, [ C, D ], true ]

# result
config:
    A:
        - A
        - abc
        - B
        - efg
    B:
        - a
        - 1
        - 2
        - 3
        - b
        - B
        - c
    C: [ A, 1, 2, 3, 4, B, C]
    D: [ A, 1, C, D, B, 4 ]
    E: [ A, C, D, B ]
```

## All Keywords ##

| keyword       | value                                                                                                                                                      | effect                                                                                                |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `__delete__`  | `True` or `str,int` or `list[str,int]`,`True` for delete all keys from other config, `str,int` only delete the specific key (for dict) or index (for list) | Delete some part of config from other.                                                                |
| `change_item` | `list[[index, item]]`,used only when merge list                                                                                                            | Add ability of merg list, change the `list[index]` from other to `item`                               |
| `insert_item` | `list[[index, item, (extend)]]`,used only when merge list                                                                                                  | Add ability of merg list, insert item to the `list` at `index`, extend=True if insert a list of items |
| `pre_item`    | `Any`or `list[Any]`,used only when merge list                                                                                                              | Add ability of merg list, add the value in the start of the list from other to item                   |
| `post_item`   | `Any`or `list[Any]`,used only when merge list                                                                                                              | Add ability of merg list, add the value in the end of the list from other to item                     |
