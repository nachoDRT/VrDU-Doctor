import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "ensurepip"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
#subprocess.check_call([sys.executable, "-m", "pip", "list"])
