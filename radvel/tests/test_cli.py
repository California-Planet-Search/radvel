import warnings
import subprocess
import sys

import radvel
import radvel.cli

warnings.simplefilter('ignore')

def _run_cmd(cmd):
    p = subprocess.Popen(cmd.split())
    p.wait()
    status = p.poll()
    out,stderr = p.communicate()

    return (status, out)

def test_help():
    """
    Test command-line install
    """
    
    cmd = 'radvel --help'
    stat,out = _run_cmd(cmd)
    #print(out)
    assert stat == 0, "{} failed with exit code {}".format(cmd,stat)

def test_clipy():
    """
    Test basic argument parsing
    """
    
    sys.argv = ["radvel", "--version"]
    try:
        radvel.cli.main()
    except SystemExit:
        pass
    

if __name__ == '__main__':
    test_clipy()
    test_help()
