import sys

from godseye.cli import main as cli_main
from godseye.dashboard import launch_dashboard
from godseye.guided_run import run_guided_workflow


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selected_video = launch_dashboard()
        if selected_video is not None:
            run_guided_workflow(selected_video)
    else:
        cli_main()
