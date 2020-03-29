import sys
from multiprocessing import Pool, Value
import subprocess
from datetime import datetime
import os
import random
import executor_nvsmi as nsmi
import property as p
import time
from threading import Thread

alive_threads = Value('i', 0)


def execute():
    print(alive_threads.value)
    time.sleep(random.randrange(0, 5))
    alive_threads.value -= 1


if __name__ == "__main__":
    l = []
    for i in range(100):
        alive_threads.value += 1
        t = Thread(target=execute)
        t.start()
        l.append(t)

    for i in l:
        i.join()
