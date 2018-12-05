from my_callbacks import Step

def onetenth_4_8_12(lr):
    steps = [4, 8,12]
    lrs = [lr, lr/2, lr/4,lr/5]
    return Step(steps, lrs)

def onetenth_50_75(lr):
    steps = [25, 40]
    lrs = [lr, lr/10, lr/100]
    return Step(steps, lrs)

def wideresnet_step(lr):
    steps = [60, 120, 160]
    lrs = [lr, lr/5, lr/25, lr/125]
    return Step(steps, lrs)
