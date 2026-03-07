"""Allow running as a module: python -m lightcurve_agent"""

import sys
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_agent import main

if __name__ == "__main__":
    main()
