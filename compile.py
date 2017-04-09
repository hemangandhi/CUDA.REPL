import subprocess as proc

def run_make(path):
    ran = proc.run(['make', '-f', path + '/makefile'], stdout=proc.PIPE, stderr=proc.PIPe)
    try:
        rc = ran.check_returncode()
        print(str(ran.stdout))
        return "Succes"
    except proc.CalledProcessError as e:
        print(e)
        print(str(ran.stderr))
        return "Issues exist"

def run_progn():
    ran = proc.run('./test', stdout=proc.PIPE, stderr=proc.PIPE)
    try:
        rc = ran.check_returncode()
        print(str(ran.stdout))
        return "Succes"
    except proc.CalledProcessError as e:
        print(e)
        print(str(ran.stderr))
        return "Issues exist"

