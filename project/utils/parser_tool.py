import argparse
import time
class ParserTool:
    def __init__(self):
        self.parser=argparse.ArgumentParser(description="My Architecture")
        self.parser.add_argument(
            "--cfg",
            type=str,
            default="./confs/coco.yaml"
        )
        self.parser.add_argument(
            "--output",
            type=str,
            default="./output/{}".format(time.strftime("%Y%m%d%H%M%S"))
        )
        self.parser.add_argument(
            "--opt",
            type=str,
            default=list(),
            nargs="+"
        )
    def get_args(self):
        return self.parser.parse_args()