import threading
import time

# threading.Event is a flag shared between threads.
#
#   event.wait() — block here until the flag is True
#   event.set()  — flip the flag to True, unblocking anyone waiting
#
# Use case: "T1, don't proceed until T2 has done X."

step_a_done = threading.Event()
step_b_done = threading.Event()


def thread_1():
    print("[T1] doing step A ...")
    time.sleep(0.5)
    print("[T1] step A done, signalling T2")
    step_a_done.set()             # flip flag → unblocks T2's wait()

    print("[T1] waiting for T2 to finish step B ...")
    step_b_done.wait()            # block until T2 calls step_b_done.set()
    print("[T1] T2 finished, T1 continues")


def thread_2():
    print("[T2] waiting for T1 to finish step A ...")
    step_a_done.wait()            # block until T1 calls step_a_done.set()
    print("[T2] T1 finished, now doing step B ...")
    time.sleep(0.5)
    print("[T2] step B done, signalling T1")
    step_b_done.set()             # flip flag → unblocks T1's wait()


th1 = threading.Thread(target=thread_1)
th2 = threading.Thread(target=thread_2)
th1.start()
th2.start()
th1.join()
th2.join()

print("\nForced order: A → B → T1 resumes")
print("Without Event, the order would be random.")
