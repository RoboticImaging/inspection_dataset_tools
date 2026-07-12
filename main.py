import argparse

from inspection_viz import AssetDataset, Viewer


def main():
    p = argparse.ArgumentParser(description="Inspection dataset visualizer")
    p.add_argument("asset", help="asset folder, e.g. /path/to/glass_house")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    Viewer(AssetDataset(args.asset), port=args.port).run()


if __name__ == "__main__":
    main()
