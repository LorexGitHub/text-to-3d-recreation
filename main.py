import argparse
import sys
from pathlib import Path
from utils.io_utils import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Text-to-3D: Generate explorable 3D worlds from text descriptions"
    )
    parser.add_argument("description", nargs="?", help="Text description of the space")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--input-file", "-f", help="Read description from a text file"
    )
    args = parser.parse_args()

    if args.input_file:
        description = Path(args.input_file).read_text(encoding="utf-8-sig").strip()
    elif args.description:
        description = args.description
    else:
        description = sys.stdin.read().strip()

    if not description:
        print("Error: No description provided. Pass it as an argument, via --input-file, or pipe it.")
        sys.exit(1)

    config = load_config(args.config)

    from pipeline.graph import create_pipeline
    app = create_pipeline(config)

    initial_state = {"raw_description": description}
    final_state = app.invoke(initial_state)

    print(f"\nDone! Output mesh: {final_state['output_mesh_path']}")


if __name__ == "__main__":
    main()
