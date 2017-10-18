
import subprocess

import radvel


def _run_cmd(cmd):
    p = subprocess.Popen(cmd.split())
    p.wait()
    status = p.poll()
    out,stderr = p.communicate()

    return (status, out)

def test_help():
    cmd = 'radvel --help'
    stat,out = _run_cmd(cmd)
    #print(out)
    assert stat == 0, "{} failed with exit code {}".format(cmd,stat)


if __name__ == '__main__':
    test_help()
