from typing import Dict, List


def deep_update(source, override):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    if isinstance(source, Dict) and isinstance(override, Dict):
        if "__delete__" in override:
            delete_keys = override.pop("__delete__")
            if isinstance(delete_keys, str):
                delete_keys = [delete_keys]

            if isinstance(delete_keys, list):
                for k in delete_keys:
                    if k in source:
                        source.pop(k)
            elif delete_keys:
                return override
        for key, value in override.items():
            if isinstance(value, Dict) and key in source:
                source[key] = deep_update(source[key], value)
            else:
                source[key] = override[key]
        return source
    elif isinstance(source, List) and isinstance(override, Dict):
        if "__delete__" in override and override["__delete__"] is True:
            override.pop("__delete__")
            source = []

        if "change_item" in override:
            change_item = override.pop("change_item")
            for index, v in change_item:
                source[index] = deep_update(source[index], v)

        if "insert_item" in override:
            insert_item = override.pop("insert_item")
            for i in range(len(insert_item)):
                if insert_item[i][0] < 0:
                    insert_item[i][0] = len(source) + insert_item[i][0]

            if "__delete__" in override:
                if isinstance(override["__delete__"], int):
                    override["__delete__"] = [override["__delete__"]]
                for i in range(len(override["__delete__"])):
                    if override["__delete__"][i] < 0:
                        if override["__delete__"][i] < -len(source):
                            raise ValueError(
                                f'Cannot delete item at index {override["__delete__"][i]} from {source}'
                            )
                        else:
                            override["__delete__"][i] += len(source)

            pre_items = []
            post_items = []
            insert_list = []
            insert_item.sort(key=lambda x: x[0])
            for item in insert_item:
                if len(item) == 3:
                    index, value, extend = item
                else:
                    index, value = item
                    extend = False

                if index < 0:
                    if extend:
                        assert isinstance(value, list), "Cannot extend a non-list"
                        pre_items.extend(item)
                    else:
                        pre_items.append(item)
                elif index >= len(source):
                    if extend:
                        assert isinstance(value, list), "Cannot extend a non-list"
                        post_items.extend(item)
                    else:
                        post_items.append(item)
                else:
                    insert_list.append([index, value, extend])
            insert_list.reverse()

            for item in insert_list:
                index, value, extend = item
                if extend:
                    assert isinstance(value, list), "Cannot extend a non-list"
                    if index >= len(source):
                        source = source + value
                    else:
                        value.reverse()
                        for v in value:
                            source.insert(index, v)
                else:
                    source.insert(index, value)

                if "__delete__" in override:
                    for i in range(len(override["__delete__"])):
                        if override["__delete__"][i] >= index:
                            if extend:
                                override["__delete__"][i] += len(value)
                            else:
                                override["__delete__"][i] += 1

            source = pre_items + source + post_items
            if pre_items:
                for i in range(len(override["__delete__"])):
                    override["__delete__"][i] += len(pre_items)

        if "__delete__" in override:
            delete_keys = override.pop("__delete__")
            if isinstance(delete_keys, int):
                delete_keys = [delete_keys]

            for i in range(len(delete_keys)):
                delete_keys[i] = int(delete_keys[i])
                if delete_keys[i] < 0:
                    if delete_keys[i] < -len(source):
                        raise ValueError(
                            f"Cannot delete item at index {delete_keys[i]} from {source}"
                        )
                    else:
                        delete_keys[i] += len(source)
            delete_keys = list(set(delete_keys))
            delete_keys.sort(reverse=True)
            for k in delete_keys:
                source.pop(k)
        if "pre_item" in override:
            source = (
                override["pre_item"]
                if isinstance(override["pre_item"], list)
                else [override["pre_item"]]
            ) + source
        if "post_item" in override:
            source = source + (
                override["post_item"]
                if isinstance(override["post_item"], list)
                else [override["post_item"]]
            )
        return source
    return override
