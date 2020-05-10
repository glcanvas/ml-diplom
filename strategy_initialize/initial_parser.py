import random


def parse_incoming_args(args: list):
    commands = []
    for arg in args:
        arg = str(arg).lower().strip()
        values = arg.split(';')
        command = {}
        for idx, v in enumerate(values):
            splitted = v.split("=")

            if len(splitted) != 2:
                continue

            command[splitted[0]] = splitted[1]
        commands.append(command)
    return commands


RANDOM = random.Random(0)
SEED_LIST = [RANDOM.randint(1, 500) for _ in range(3)]
ATTENTION_MODULE_LEARNING_RATES = [1e-3]
CLASS_BORDER = [(0, 5)]
RUN_NAME_RANGE_FROM = 1010
TRAIN_SIZE = 1800
EPOCHS_COUNT = 150

if __name__ == "__main__":
    print(SEED_LIST)