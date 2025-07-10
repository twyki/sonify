import os
import sys


def main():
    # Reinvoke via Streamlit CLI
    entry = os.path.join(
        os.path.dirname(__file__), '1_single_file_demo.py'
    )
    sys.argv = ['streamlit', 'run', entry]
    from streamlit.web.cli import main as st_main
    sys.exit(st_main())
