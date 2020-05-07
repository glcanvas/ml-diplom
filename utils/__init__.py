import sys

print(sys.path)
sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")

if __name__ == "__main__":

    try:
        exit(0)
    except BaseException as e:
        print(e)
        print("FF")
