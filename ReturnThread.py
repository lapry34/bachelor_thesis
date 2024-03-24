from threading import Thread

# custom thread
class ReturnThread(Thread):
    # constructor
    def __init__(self, target, args):
        # execute the base constructor
        super().__init__(group=None, target=target, args=args)
        # set a default value
        self._value = None
    # function executed in a new thread
    def run(self):
        self._value = self._target(self._args)

    def getValue(self):
        return self._value