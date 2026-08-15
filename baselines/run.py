r"""Run any registered baseline over the sequences a data config selects.

    python baselines/run.py --baseline da3 \
        --data baselines/configs/data/old_house.yaml \
        --model baselines/configs/da3.yaml

Each baseline needs its own environment, so this is normally run from inside that baseline's
container (`baselines/<name>/run_container.sh`) — where it is equivalent to that baseline's own
`run_sequence.py`, which takes the same arguments and is what the container's shell starts in.
Use whichever reads better; this one exists so a script that runs several baselines doesn't have
to know where each one's runner lives.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from baselines import registry
from baselines.common.config import add_config_args
from baselines.common.runner import run_from_args


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True, metavar="NAME",
                   help=f"which baseline to run; registered: {', '.join(registry.available())}")
    add_config_args(p)
    args = p.parse_args(argv)
    run_from_args(registry.get(args.baseline), args)


if __name__ == "__main__":
    main()
