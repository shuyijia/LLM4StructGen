def parse_attributes(value):
    if value.lower in {"true", "false"}:
        return value.lower() == "true"
    elif value.lower() == "random":
        return value.lower()
    elif value.lower() == "none" or not value:
        return None
    else:
        attributes = value.split(",")
        return attributes