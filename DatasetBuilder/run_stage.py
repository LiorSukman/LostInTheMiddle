"""Run individual pipeline stages.

Usage:
    python -m DatasetBuilder.run_stage 1    # Harvest
    python -m DatasetBuilder.run_stage 2    # Chunk
    python -m DatasetBuilder.run_stage 5    # Build distractors
    python -m DatasetBuilder.run_stage 6    # Validate
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m DatasetBuilder.run_stage <stage_number>")
        print("Stages: 1 (harvest), 2 (chunk), 5 (distractors), 6 (validate)")
        sys.exit(1)

    stage = sys.argv[1]

    if stage == "1":
        from DatasetBuilder.pipeline.stage1_harvest import run
        run()
    elif stage == "2":
        from DatasetBuilder.pipeline.stage2_chunk import run
        run()
    elif stage == "4":
        from DatasetBuilder.pipeline.stage4_full_filter import run
        run()
    elif stage == "5":
        from DatasetBuilder.pipeline.stage5_distractors import run
        run()
    elif stage == "6":
        from DatasetBuilder.pipeline.stage6_validate import run
        run()
    else:
        print(f"Unknown stage: {stage}")
        print("Stage 3 (QA extraction) is handled by Claude directly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
