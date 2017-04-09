import kernel
from parameters import generate_call
from cu_compile import *
from functools import reduce
from load_file import intern_file
import sys

def prompt(kernel,end=" > "):
    return input(kernel["name"] + end)

def tokenize_cmd(cmd):
    l_ind = 0
    r_ind = 1
    in_q = False
    while r_ind < len(cmd):
        if not in_q and cmd[r_ind] == " ":
            yield cmd[l_ind:r_ind].replace('"', '')
            l_ind = r_ind + 1
        elif cmd[r_ind] == '"':
            in_q = not in_q
        r_ind = r_ind + 1
    if l_ind < r_ind:
        yield cmd[l_ind:r_ind]


def parse_args_and_apply(to, what, val, state):
    def int_where_possible(val):
        try:
            return int(val)
        except:
            return val

    if hasattr(to, "__call__"):
        try:
            intified = list(map(int_where_possible, what))
            if intified[0] in state:
                intified[0] = state[intified[0]]
            return to(*intified)
        except Exception as e:
            print(e)
            return "Enter " + val + "._ to get help on this function - this invocation was erroneous!"
    else:
        return to

def eval_cmd(state, cmd):

    def subscript_dict(d, idx):
        if type(d) == str or type(d) == list:
            return d
        if type(d) != dict:
            if hasattr(d, '__call__') and hasattr(d, '__doc__'):
                return d.__doc__
            d = d.__dict__
        return d.get(idx, d.get("__doc__", "Can't understand you"))

    toks = list(tokenize_cmd(cmd))
    where = reduce(subscript_dict, toks[0].split("."), state)
    if len(toks) > 1:
        return parse_args_and_apply(where, toks[1:], toks[0], state)
    else:
        if hasattr(where, '__call__'):
            try:
                return where()
            except Exception as e:
                print(e)
                return "Looks like you need parameters! %" + toks[0] + "._ for details!"
        else:
            return where

def parse_line(state, inp):
    if inp.startswith("%"):
        next = eval_cmd(state, inp[1:])
        if type(next) != dict or "curnel" not in next:
            if next is not None:
                print(str(next))
        else:
            state = next
    else:
        kernel.add_line(state["curnel"], inp)
    return state

def driver():
    default = kernel.mk_kernel("origin", "", "__global__")
    state = {i : getattr(kernel, i) for i in dir(kernel) if not i.startswith('__')}
    state["curnel"] = default
    state["origin"] = default

    def help():
        return """
            Welcome to a CUDA REPL!
            The hope is to help with small-scale testing of kernels for
            correctness.
            There is much functionality in a hopefully simple system.

            '%' is the prefix for commands. Otherwise, all input is added
            to the current kernel (with minimal syntax-checking).
            '%ls' lists the functionality.

            '%curnel' is the kernel currently being worked on.
            '%thing._' will get the documentation for thing.
        """

    def mk_kernel(name, types, access):
        """Makes a kernel and sets the current to it."""
        if name in state:
            return "Cannot overwrite!"
        k = kernel.mk_kernel(name, types, access)
        state[name] = k
        state["curnel"] = k
        return state

    def exit():
        """Exit the app."""
        print("Bye!")
        sys.exit()

    def ls():
        """List all the functions that can be called."""
        fns = ('\t' + i for i in state if hasattr(state[i], '__call__'))
        ker = ('\t' + i for i in state if type(state[i]) == dict and i != 'curnel')
        return "Functions:\n" + '\n'.join(fns) + '\nKernels:\n' + '\n'.join(ker)

    def rm_kernel(kernel):
        """Get rid of a kernel."""
        nonlocal state
        if kernel not in state:
            return "No such kernel."
        del state[kernel]
        return state

    def alter_kernel(kernel, what, new):
        """Alters the name, access, and types of a kernel."""
        nonlocal state
        if what not in ["name", "access", "types"]:
            return "Cannot alter "+ what + " in this way."
        if what == 'name':
            old = kernel['name']
            kernel['name'] = new
            state[new] = kernel
            del state[old]
            if state['curnel']['name'] == old:
                state['curnel'] = kernel
            return state
        elif what == 'access' and new not in ['__global__', '__host__', '__device__']:
            return 'Cannot set to this type'
        else:
            kernel[what] = new.replace('"', '')
            return state

    def save(path):
        """Saves to path. Overwrites."""
        try:
            with open(path, 'w') as f:
                f.write('#include <cuda_runtime.h>\n')
                f.write('#include <stdio.h>\n')
                for i in state:
                    if type(state[i]) == dict and i != 'curnel':
                        f.write(kernel.gen_code(state[i]))
                f.flush()
                return "Saved to " + path
        except Exception as e:
            print(e)
            return "Error in writing"

    def load(path):
        """Loads kernels from path. Will ignore non-kernel code."""
        for i in intern_file(path):
            state[i['name']] = i
            state['curnel'] = i
        print("Warning: loaded code will not reflect the entire file, only the kernels")
        print("Saving will overwrite potentially important code.")
        return state

    def ch_kernel(name):
        """Alter the kernel being modified by 'curnel'"""
        if name not in state or type(state[name]) != dict:
            return "Invalid transfer"
        state['curnel'] = state[name]
        return state

    def gen_test_file(path):
        """Generates a complete test file, as ready for compilation as your code."""
        ls, rs, cs = [], [], []
        try:
            with open(path, 'w') as f:
                f.write('#include <cuda_runtime.h>\n')
                f.write('#include <stdio.h>\n')
                for i in state:
                    if type(state[i]) == dict and i != 'curnel':
                        gen = kernel.gen_code(state[i])
                        f.write(gen + ('\n' if not gen.endswith('\n') else ''))
                        l, r, c = generate_call(state[i])
                        ls.append(l)
                        rs.append(r)
                        cs.append(c)
                lj = '\n'.join(ls)
                rj = '\n'.join(rs)
                cj = '\n'.join(cs)
                f.write('\n'.join(['int main(){', lj, cj, rj, '}']))
                f.flush()
                return "Saved to " + path
        except Exception as e:
            print(e)
            return "Error in writing"

    def make(path):
        """Makes "makefile" for file at path."""
        try:
            with open('/'.join(path.split('/')[:-1]) + '/makefile', 'w') as m:
                m.write("all: {0}\n\tnvcc {0} -o test".format(path))
            return "Success"
        except Exception as e:
            print(e)
            return "Error in IO."

    def dump_file(path):
        """Print the contents of a file."""
        try:
            with open(path) as m:
                return m.read() + '\n'
        except Exception as e:
            print(e)
            return "Error in IO."

    def compile(path = None):
        """Makes the program at path. All output is from GNU/make"""
        if path is None:
            path = '/tmp/test.cu'
        if gen_test_file(path).startswith("Saved to") and make(path) == "Success":
            return run_make('/'.join(path.split('/')[:-1]))
        return "Files not found"

    def run_from(path=None):
        """Saves, compiles and runs a particular file."""
        if compile(path) == "Success":
            return run_progn()
        else:
            return "Compilation of compilation issues."


    state["mk_kernel"] = mk_kernel
    state["alter_kernel"] = alter_kernel
    state["exit"] = exit
    state["ls"] = ls
    state["help"] = help
    state["rm_kernel"] = rm_kernel
    state["save"] = save
    state["load"] = load
    state["gen_test_file"] = gen_test_file
    state["compile"] = compile
    state["run_from"] = run_from
    state["dump_file"] = dump_file
    state["make"] = make

    print(help())
    while state != None:
        state = parse_line(state, prompt(state["curnel"]))

if __name__ == "__main__":
    driver()
