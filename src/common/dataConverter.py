import dataclasses

def rowToObject(row: dict, cls: type, column_mapping: dict) -> object:
    mapped = {
        dc_field: _safe_cast(row.get(pd_col))  # pd_col이 없으면 None 자동 반환
        for pd_col, dc_field in column_mapping.items()
    }

    # 누락된 필드를 None으로 강제로 채움 (row에 없어도)
    all_fields = {f.name for f in dataclasses.fields(cls)}
    for f in all_fields:
        mapped.setdefault(f, None)

    return cls(**mapped)

def _safe_cast(value):
    # None은 그대로
    if value is None:
        return None
    # float이지만 정수로 표현 가능하면 int로
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value