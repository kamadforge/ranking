

def mama(i):



def _fisher1(i, notused1, notused2, grad_output):
    act1 = self.act1.detach()
    grad = grad_output[0].detach()
    print(grad_output[0].shape)

    g_nk = (act1 * grad).sum(-1).sum(-1)
    del_k = g_nk.pow(2).mean(0).mul(0.5)
    print(del_k.shape)
    self.running_fisher[0] += del_k
