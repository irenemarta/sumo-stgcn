import os, errno
from pathlib import Path

TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
INPUTS_DIR = os.path.join(ROOT_DIR, "inputs")
TLS_ON_SET  = {
    "joinedS_5726802070_#5more",
    "joinedS_7044704202_#8more",
    "joinedS_12090799868_12136674195_12136674196_2602200967_#25more",
    "joinedS_442883032_442883033_442883058_5726788762_#10more",
    "joinedS_1274135270_480164728_5743628119_5743628120_#3more",
    "joinedS_5743605972_#11more",
    "joinedS_429516080_5743628105_5743628106_5743628107_#10more",
    "joinedS_1491322243_246625689_2602093460_2602093464_#17more",
    "joinedS_10176227249_10176227251_442883041_442883057_#17more",
    "joinedS_12101709375_#14more",
    "joinedS_28807263_441653642_441653644_5758251607_#22more",
}

print(ROOT_DIR)
SRC = os.path.join(ROOT_DIR, "src")

def _ensure_folder_existence():
    __structure = {
        "data": ["built_dataset"],
        "inputs": ["DetOut_Day"],
        "src": ["blocks", "checkpoints", "data", "models"],
    }

    try:
        for parent, children in __structure.items():
            if not children:
                os.makedirs(os.path.join(ROOT_DIR, parent), exist_ok=True)
            else:
                for child in children:
                    os.makedirs(os.path.join(ROOT_DIR, parent, child), exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
