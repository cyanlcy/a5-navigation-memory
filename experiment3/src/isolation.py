"""Linux-only isolation for one submitted decision, internal to the DSW evaluation container."""
import ctypes
import errno
import os
def no_network():
    lib=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_release.argtypes=[ctypes.c_void_p]
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    ctx=lib.seccomp_init(0x7fff0000)
    if not ctx:raise RuntimeError('seccomp_init')
    try:
        for name in (b'socket',b'connect',b'socketpair',b'io_uring_setup'):
            n=lib.seccomp_syscall_resolve_name(name)
            if n>=0 and lib.seccomp_rule_add(ctx,0x00050000|errno.EPERM,n,0):raise RuntimeError('seccomp rule')
        if lib.seccomp_load(ctx):raise RuntimeError('seccomp_load')
    finally:lib.seccomp_release(ctx)

def single_question():
    no_network()
    os.chmod('.',0o555);os.chroot('.');os.chdir('/')
    os.setgroups([]);os.setgid(65534);os.setuid(65534)
    if os.geteuid()!=65534 or os.listdir('/'):raise RuntimeError('candidate root must be empty')
