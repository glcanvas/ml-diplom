import sys
import traceback

print(sys.path)
sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")
sys.path.insert(0, "/home/ubuntu/ml3/ml-diplom")

def a():
    raise Exception("DDS")
if __name__ == "__main__":

    try:
        a()
    except BaseException as e:
        print(e)
        print("FF")
        s = traceback.extract_tb(e.__traceback__)
        print(s)