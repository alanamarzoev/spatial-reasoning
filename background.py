#!/usr/bin/env python3

import os
import subprocess
import sys

_DIRNAME = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
_RESULTS_DIR = os.path.join(_DIRNAME, 'results')

os.makedirs(_RESULTS_DIR, exist_ok=True)

def run():
    identifier = sys.argv[1]
    com = sys.argv[2:]

    def _results(name):
        return os.path.join(_RESULTS_DIR, '{}.{}'.format(identifier, name))

    def _rm(name):
        try:
            os.unlink(_results(name))
        except:
            pass

    _rm('stdout'); _rm('stderr'); _rm('pid'); _rm('result'); _rm('com')
    print('h4')
    with open(_results('com'), 'w') as f:
        f.write(' '.join(com) + '\n')

    with open(_results('stdout'), 'w') as stdout, \
         open(_results('stderr'), 'w') as stderr:
        proc = subprocess.Popen(
            com,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            universal_newlines=True,
            start_new_session=True,
        )
        with open(_results('pid'), 'w') as f:
            f.write('{}\n'.format(proc.pid))
        with open(_results('pgid'), 'w') as f:
            f.write('{}\n'.format(os.getpgid(proc.pid)))
        res = proc.wait()
        with open(_results('result'), 'w') as f:
            f.write('{}\n'.format(res))


def main():
    print('hi')
    if os.fork() > 0:
        print('help2')
        sys.exit(0)

    # os.chdir("/")
    os.setsid()
    os.umask(0)
    print('hi2')
    if os.fork() > 0:
        print('help')
        sys.exit(0)

    run()
    print('h3')

if __name__ == '__main__':
    main()