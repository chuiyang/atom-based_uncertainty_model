import os
import sys
print(sys.argv[1])
os.makedirs(f'{sys.argv[1]}', exist_ok=True)
