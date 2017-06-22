cimport cython

def big_sum():
    cdef int a[10000]

    for i in range(10000):
        a[i] = i
    # <==================== I want to put a break here
    cdef int my_sum
    my_sum = 0
    for i in range(1000):
        my_sum += a[i]
    return my_sum
