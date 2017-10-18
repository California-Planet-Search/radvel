
import commands

import radvel


def _run_cmd(cmd):
    status, output = commands.getstatusoutput(cmd)

    return (status, output)

def test_help():
    stat, out = _run_cmd('radvel --help')
    print(out)
    assert stat == 0, "{} failed with exit code {}".format(cmd,stat)


if __name__ == '__main__':
    test_help()
