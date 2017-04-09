def get_print_spec():
    return input("Enter the printf spec > ")

def gen_array(param, sz = None):
    typ = ' '.join(param.split(' ')[:-1])
    raw_t = typ.split(' ')[0]
    name = param.split(' ')[-1]
    if sz is None:
        values = input("Enter the values you'd like to test separated by commas > ")
        a_size = len(values.split(','))
        host_ln = raw_t + " " + name + "[] = {" + values + "};"
    else:
        values = input("Enter the value > ")
        host_ln = raw_t + " " + name + " = " + values + ";"
        a_size = sz;

    dev_alloc = typ + ' gpu_' + name + ';\ncudaMalloc( &gpu_' + name +\
            ', sizeof(' + raw_t + ') * ' + str(a_size) + ');'
    h2d_cpy = 'cudaMemcpy(gpu_' + name + ', ' + name + ', sizeof(' + raw_t + ') * ' + str(a_size) +\
            ', cudaMemcpyHostToDevice);'
    d2h_cpy = 'cudaMemcpy(' + name + ', gpu_' + name + ', sizeof(' + raw_t + ') * ' + str(a_size) +\
            ', cudaMemcpyDeviceToHost);'
    free = 'cudaFree( gpu_' + name + ');'
    if sz is None:
        print_l = 'for(int i = 0; i < ' + str(a_size) + '; i++)\nprintf("' + name + '[%d] : ' +\
                get_print_spec() + '\\n", i, ' + name + '[i]);'
    else:
        print_l = 'printf("' + name + ' = ' + get_print_spec() + '\\n", '+ name + ');'

    return ('\n'.join([host_ln, dev_alloc, h2d_cpy]), '\n'.join([d2h_cpy, free, print_l]), a_size)

def is_arr_spec(spc):
    return len(spc[3]) == 3

def gen_const(param, speced):
    names = {i[1]: i[3][2] for i in list(filter(is_arr_spec, speced))}
    value = input("Enter the constant value. A choice of " + str(list(names.keys())) + " is possible. > ")
    if value in names:
        value = str(names[value])
    return value

def specify_params(kernel):
    params = list(map(lambda x: [x[1].strip(), x[1].split(' ')[-1], x[0], tuple()], enumerate(kernel['types'].split(','))))
    speced = []
    while len(params) > 0:
        names = {i[1]: i for i in params}
        who = input("Choose a parameter from " + str(list(names.keys())) + " > ")
        while who not in names:
            who = input("Choose a parameter from " + str(list(names.keys())) + " > ")
        speced = list(filter(lambda x: x[1] != who, speced))
        params = list(filter(lambda x: x[1] != who, params))

        typ = input("Choose the kind of variable this should be...\n(ie. array, constant, pointer) > ")
        while typ not in ['array', 'constant', 'pointer']:
            typ = input("Choose the kind of variable this should be...\n(ie. array, constant, pointer) > ")
        if typ == 'array':
            names[who][3] = gen_array(names[who][0])
            speced.append(names[who])
        elif typ == 'pointer':
            names[who][3] = gen_array(names[who][0], 1)
            speced.append(names[who])
        else:
            names[who][3] = gen_const(names[who][0], speced)
            speced.append(names[who])
    return speced

def generate_call(kernel):
    par = sorted(specify_params(kernel), key = lambda x: x[2])

    block_dim = (input("Enter the block dimension > "))
    block_num = (input("Enter the block count > "))

    l_cd = []
    r_cd = ['printf("Last error: %s \\n", cudaGetErrorString(cudaDeviceSynchronize()));']
    call = []
    fst = True
    for spec in par:
        if is_arr_spec(spec):
            l_cd.append(spec[3][0])
            r_cd.append(spec[3][1])
            if fst:
                call.append(kernel['name'] + '<<<' + block_dim + ',' + block_num + '>>>( gpu_' + spec[1])
                fst = False
            else:
                call.append('gpu_' + spec[1])
        else:
            call.append(spec[3])
    return '\n'.join(l_cd), '\n'.join(r_cd), ','.join(call) + ');'

