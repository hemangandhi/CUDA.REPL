import kernel
from functools import reduce
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
            return where()
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
    default = kernel.mk_kernel("default", "", "__global__")
    state = {i : getattr(kernel, i) for i in dir(kernel) if not i.startswith('__')}
    state["curnel"] = default
    state["default"] = default

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
        if what not in ["name", "access", "type"]:
            return "Cannot alter "+ what + " in this way."
        if what == 'name':
            old = kernel['name']
            kernel['name'] = new
            state[what] = kernel
            del state[old]
            return state
        elif what == 'access' and new not in ['__global__', '__host__', '__device__']:
            return 'Cannot set to this type'
        else:
            kernel[what] = new
            return state

    state["mk_kernel"] = mk_kernel
    state["alter_kernel"] = alter_kernel
    state["exit"] = exit
    state["ls"] = ls
    state["help"] = help
    state["rm_kernel"] = rm_kernel

    while state != None:
        state = parse_line(state, prompt(state["curnel"]))

if __name__ == "__main__":
    driver()
