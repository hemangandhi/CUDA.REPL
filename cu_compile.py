import subprocess as proc

def run_make(path):
    try:
        ran = proc.check_output(['make', '-f', path + '/makefile'], stderr=proc.STDOUT)
        print(ran.decode('utf-8'))
        return "Success"
    except proc.CalledProcessError as e:
        print(e.output.decode('utf-8'))
        return "Issues exist"

def run_progn():
    try:
        ran = proc.check_output(['./test'], stderr=proc.STDOUT)
        print(ran.decode('utf-8'))
        return "Succes"
    except proc.CalledProcessError as e:
        print(e.output.decode('utf-8'))
        return "Issues exist"

