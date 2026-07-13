import argparse

from inspection_viz import AssetDataset, Viewer


def main():
    p = argparse.ArgumentParser(description="Inspection dataset visualizer")
    p.add_argument("asset", help="asset folder, e.g. /path/to/glass_house")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--mesh-faces", type=int, default=1_000_000, help="face budget for gt mesh vis")
    args = p.parse_args()
    Viewer(AssetDataset(args.asset), port=args.port, mesh_max_faces=args.mesh_faces).run()


if __name__ == "__main__":
    main()
