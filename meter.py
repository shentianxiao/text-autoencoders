
class AverageMeter(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.cnt = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        self.cnt += n
        self.sum += val * n
        self.avg = self.sum / self.cnt
