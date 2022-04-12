import sys
import re

for line in sys.stdin:
    # delete ▁
    # add @@ to morphemes which are not the first in a word
    line = line.lstrip()
    line = re.sub(r'▁ ', '▁', line)
    line = re.sub(r' (?!▁)', ' @@', line)
    line = re.sub(r'▁', '', line)
    sys.stdout.write(line)
