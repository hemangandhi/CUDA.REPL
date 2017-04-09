import kernel

def guess_if_kernel(lines, ind):
    if lines[ind].split(' ')[0] in ['__global__', '__device__', '__host__']:
        if ind > 0 and lines[ind - 1].strip().startswith('template'):
            return ind - 1
        else:
            return ind
    else:
        return -1

def partition_into_functions(f):
    l_ind = -1
    ctr = 0
    for i, l in enumerate(f):
        ctr = ctr + l.count('{') - l.count('}')
        if l_ind == -1 and ctr > 0:
            l_ind = i
        elif ctr == 0 and l_ind >= 0:
            yield (l_ind, i)
            l_ind = -1

def to_kernels(f):
    for l, r in partition_into_functions(f):
        l = guess_if_kernel(f, l)
        if l == -1:
            continue
        yield kernel.mk_kernel(f[l].split(' ')[1],
                f[l][f[l].find('(') + 1:f[l].find(')')],
                f[l].split(' ')[0],
                f[l+1:r])

def intern_file(path):
    with open(path) as f:
        yield from to_kernels(f.readlines())
