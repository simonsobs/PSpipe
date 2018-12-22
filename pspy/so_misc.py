"""
@brief: misc python routines 
"""
from __future__ import absolute_import, print_function

def is_empty(s):
    '''
        if s is either '' or None, this function returns True. Otherwise, false.
    '''
    return True if (s is None) or (len(s) == 0) else False

def get_from_dict(nested_dict, keys, safe=True):
    if not type(keys) is tuple: keys = (keys,)
    if safe and not has_key(nested_dict, keys): return None

    if(len(keys) > 1):
        return get_from_nested_dict(nested_dict[keys[0]], keys[1:], False)
    else:
        return nested_dict[keys[0]]

def has_key(nested_dict, keys):
    ''' search through nested dictionary to fine the elements '''
    if not type(keys) is tuple: keys = (keys,)
    if not type(nested_dict) == dict: return False

    if(len(keys) > 1):
        has_it = nested_dict.has_key(keys[0])
        return has_key(nested_dict[keys[0]], keys[1:]) if has_it else False
    else:
        return nested_dict.has_key(keys[0])

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeException("Can't convert 'str' object to 'boolean'")

def merge_dict(a, b, path=None, clean = True):
    '''
    merges b into a

    ref: https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge"
    '''
    if path is None: path = []

    for key in b:
        if key in a:
            if isinstance(a[key], dict):
                if isinstance(b[key], dict):
                    merge_dict(a[key], b[key], path + [str(key)])
                else:
                    a[key] = b[key]
            elif isinstance(a[key], np.ndarray) and np.array_equal(a[key], b[key]):
                pass
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]

    if clean: del b

    return a

