import determined.common.context
import pathlib
blah = determined.common.context.read_v1_context(pathlib.Path("."))
print()
for _file in sorted([_b.path for _b in blah]):
    print(_file)

