"""
Parse property file, return snapshot with values
"""

# constants
BANNED_GPU = "banned_gpu"
MAX_THREAD_ON_GPU = "max_thread_on_gpu"
MAX_ALIVE_THREAD = "max_alive_thread"


class PropertyContext:
    """
    Contains structured property file
    """

    def __init__(self, banned_gpu: int, max_thread_on_gpu: int, max_alive_threads: int,
                 add_list: list, remove_list: list, stop_list: list):
        self.max_thread_on_gpu = max_thread_on_gpu
        self.banned_gpu = banned_gpu
        self.max_alive_threads = max_alive_threads
        self.add_list = add_list
        self.remove_list = remove_list
        self.stop_list = stop_list

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, PropertyContext):
            return False
        return self.banned_gpu == o.banned_gpu and self.max_thread_on_gpu == o.max_thread_on_gpu and \
               self.add_list == o.add_list and self.remove_list == o.remove_list and self.stop_list == o.stop_list and \
               self.max_alive_threads == o.max_alive_threads

    def __str__(self) -> str:
        return "banned_gpu = {},\nmax_thread_on_gpu = {},\nmax_alive_threads = {},\nadd_list = {},\n" \
               "remove_list = {}\n,stop_list = {}" \
            .format(self.banned_gpu, self.max_thread_on_gpu, self.max_alive_threads, self.add_list, self.remove_list,
                    self.stop_list)


def __read_property(property_file) -> dict:
    """
    Read property file by line and return dict with key value
    :param property_file: path to property file
    :return: dict with parsed property file
    """
    return dict(line.strip().split('=') for line in open(property_file) if
                not line.strip().startswith('#') and len(line.strip()) > 0)


def __parse_add_command(add_command: str) -> tuple:
    key_value = list(filter(lambda x: len(x) > 0, add_command.split(";")))
    dct = {}
    for kv in key_value:
        kv_s = kv.split("|")
        if len(kv_s) != 2:
            continue
        dct[kv_s[0]] = kv_s[1]
    # validate correct command
    if "--script_name" not in dct:
        return False, None
    if "--script_memory" not in dct:
        return False, None
    if "--pre_train" not in dct:
        return False, None
    if "--run_name" not in dct:
        return False, None
    if "--algorithm_name" not in dct:
        return False, None
    if "--left_class_number" not in dct:
        return False, None
    if "--right_class_number" not in dct:
        return False, None
    if "--classifier_learning_rate" not in dct:
        return False, None
    if "--model_identifier" not in dct:
        return False, None

    return True, dct


def __parse_remove_command(remove_command: str) -> tuple:
    try:
        return True, int(remove_command)
    except Exception:
        return False, 0


def __parse_stop_command(stop_command: str) -> tuple:
    try:
        return True, int(stop_command)
    except Exception:
        return False, 0


def __register_commands(dct: dict, latest_index: int) -> tuple:
    actual_index = latest_index
    add_commands = []
    remove_commands = []
    stop_commands = []
    for offset in range(len(dct.keys())):
        add_key = "{}.add".format(actual_index)
        remove_key = "{}.remove".format(actual_index)
        stop_key = "{}.stop".format(actual_index)

        if add_key in dct:
            is_valid, dct = __parse_add_command(dct[add_key])
            actual_index += 1
            if is_valid:
                add_commands.append(dct)
        elif remove_key in dct:
            is_valid, value = __parse_remove_command(dct[remove_key])
            actual_index += 1
            if is_valid:
                remove_commands.append(value)
        elif stop_key in dct:
            is_valid, value = __parse_stop_command(dct[stop_key])
            actual_index += 1
            if is_valid:
                stop_commands.append(value)
    return actual_index, add_commands, remove_commands, stop_commands


def process_property_file(property_file, latest_index: int) -> tuple:
    dct = __read_property(property_file)
    banned_gpu = 228 if BANNED_GPU not in dct else int(dct[BANNED_GPU])
    max_thread_on_gpu = 228 if MAX_THREAD_ON_GPU not in dct else int(dct[MAX_THREAD_ON_GPU])
    max_alive_thread = 10 if MAX_ALIVE_THREAD not in dct else int(dct[MAX_ALIVE_THREAD])
    latest_index, add_list, remove_list, stop_list = __register_commands(dct, latest_index)

    return PropertyContext(banned_gpu, max_thread_on_gpu, max_alive_thread, add_list, remove_list,
                           stop_list), latest_index
