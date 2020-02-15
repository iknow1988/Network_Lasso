import sys
import os
import re # Perl-style regular expressions
from functools import partial
import multiprocessing
from multiprocessing.managers import SyncManager
import time
from Queue import Queue as _Queue
#from factorize import factorize_naive
#from eblib.utils import Timer

IP = '127.0.0.1'
PORTNUM = 5000
AUTHKEY = 'abcdefg'

# class Queue(_Queue):
#     """ A picklable queue. """   
#     def __getstate__(self):
#         # Only pickle the state we care about
#         return (self.maxsize, self.queue, self.unfinished_tasks)
# 
#     def __setstate__(self, state):
#         # Re-initialize the object, then overwrite the default state with
#         # our pickled state.
#         Queue.__init__(self)
#         self.maxsize = state[0]
#         self.queue = state[1]
#         self.unfinished_tasks = state[2]
# 
# 
# def get_q(q):
#     return q

class JobQueueManager(SyncManager):
        pass

def factorize_naive(n):
    print"factorizing",n,"!"
    return 1

def make_server_manager(port, authkey):
#     job_q = Queue()
#     result_q = Queue()
#     job_q = multiprocessing.Queue
#     result_q = multiprocessing.Queue

    #JobQueueManager.register('get_job_q',  callable=partial(get_q, job_q))
    #JobQueueManager.register('get_result_q', callable=partial(get_q, result_q))

    manager = JobQueueManager(address=(IP, port), authkey=authkey)
    manager.start()
    print 'Server started at port %s' % port
    return manager

def make_client_manager(ip, port, authkey):
    class ServerQueueManager(SyncManager):
        pass

    ServerQueueManager.register('get_job_q')
    ServerQueueManager.register('get_result_q')

    manager = ServerQueueManager(address=(ip, port), authkey=authkey)
    manager.connect()

    print 'Client connected to %s:%s' % (ip, port)
    return manager

def factorizer_worker(job_q, result_q):
    myname = multiprocessing.current_process().name
    outdict={}
    while True:
        try:
            job = job_q.get_nowait()
            #print '%s got %s nums...' % (myname, len(job))
            for n in job:
                #outdict.append(factorize_naive(n))
                outdict[n] = factorize_naive(n)
            #outdict = {n: factorize_naive(n) for n in job}
            result_q.put(outdict)
            #print '  %s done' % myname
        except Queue.Empty:
            return

def mp_factorizer(shared_job_q, shared_result_q, nprocs):
    procs = []
    for i in range(nprocs):
        p = multiprocessing.Process(
                target=factorizer_worker,
                args=(shared_job_q, shared_result_q))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

def make_nums(N):
    nums = [9999]
    for i in xrange(N):
        nums.append(nums[-1] + 2)
    return nums

def runserver():
    manager = make_server_manager(PORTNUM, AUTHKEY)
    shared_job_q = manager.get_job_q()
    shared_result_q = manager.get_result_q()

    N = 102
    nums = make_nums(N)

    chunksize = 43
    for i in range(0, len(nums), chunksize):
        #print 'putting chunk %s:%s in job Q' % (i, i + chunksize)
        shared_job_q.put(nums[i:i + chunksize])

    #with Timer('howlong...'):
#     if 1:
#         mp_factorizer(shared_job_q, shared_result_q, 8)
# 
    numresults = 0
    resultdict = {}
    while numresults < N:
        outdict = shared_result_q.get()
        resultdict.update(outdict)
        numresults += len(outdict)

    for num, factors in resultdict.iteritems():
        print("answer is ", num, factors)
        #product = reduce(lambda a, b: a * b, factors, 1)
        #if num != product:
        #    assert False, "Verification failed for number %s" % num

    print '--- DONE ---'
    time.sleep(2)
    manager.shutdown()

def runclient():
    manager = make_client_manager(IP, PORTNUM, AUTHKEY)
    job_q = manager.get_job_q()
    result_q = manager.get_result_q()

    mp_factorizer(job_q, result_q, 4)

if __name__ == '__main__':
    job_q = multiprocessing.Queue
    result_q = multiprocessing.Queue
    JobQueueManager.register('get_job_q', callable=lambda: job_q)
    JobQueueManager.register('get_result_q', callable=lambda: result_q)
    import sys
    if len(sys.argv) > 1:
        runclient()
    else:
        runserver()