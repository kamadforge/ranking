import torch
from torchnet import meter
from torch import nn
from tqdm import tqdm
import data.dataset as dataset
import models
from utils.visualization import Visualizer
import numpy as np
import time
from time import localtime
from config import opt
import os
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt






def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


device = None
cudnn.benchmark = True
current_time = time.strftime('%Y-%m-%d %H:%M:%S', localtime())
print(current_time)
vis = None


def gradi(module):
    module[0] = 0
    #print(module[0])
    #print(module[1])

def gradi_all(module):
    module[:]=0


def mc_sampling(vector):

    sigmoid = 1 / (1 + np.exp(-np.array(vector)))

    a=np.random.uniform(0,1, len(sigmoid))

    vector_sampled=[int(i) for i in sigmoid > a]

    print("vector_sampled: ", vector_sampled)

    return  vector_sampled







def get_external_weights(model, run_type):


    # model.state_dict()['convs.0.weights'][:] = torch.load("../results_compression/models/arrays/99.27_c1.weight").to(
    #     device)
    # model.state_dict()['convs.0.bias'][:] = torch.load("../results_compression/models/arrays/99.27_c1.bias").to(device)
    # model.state_dict()['convs.3.weights'][:] = torch.load("../results_compression/models/arrays/99.27_c3.weight").to(
    #     device)
    # model.state_dict()['convs.3.bias'][:] = torch.load("../results_compression/models/arrays/99.27_c3.bias").to(device)

    model.state_dict()['conv1.weights'][:] = torch.load("../results_compression/models/arrays/99.27_c1.weight").to(
        device)
    model.state_dict()['conv1.bias'][:] = torch.load("../results_compression/models/arrays/99.27_c1.bias").to(device)
    model.state_dict()['conv2.weights'][:] = torch.load("../results_compression/models/arrays/99.27_c3.weight").to(
        device)
    model.state_dict()['conv2.bias'][:] = torch.load("../results_compression/models/arrays/99.27_c3.bias").to(device)

    model.state_dict()['fcs.0.weights'][:] = torch.load("../results_compression/models/arrays/99.27_c5.weight").t().to(
        device)
    model.state_dict()['fcs.0.bias'][:] = torch.load("../results_compression/models/arrays/99.27_c5.bias").to(device)
    model.state_dict()['fcs.2.weights'][:] = torch.load("../results_compression/models/arrays/99.27_f6.weight").t().to(
        device)
    model.state_dict()['fcs.2.bias'][:] = torch.load("../results_compression/models/arrays/99.27_f6.bias").to(device)

    test_phi = 10
    model.state_dict()['conv1.z_phi'][:] = (torch.ones(10) * test_phi).to(device)

    if run_type=='test':

        z_phis=[[1.0000e+05, -4.3877e-01, 6.5680e-01, 7.2174e-02, -3.5518e-01,-2.0797e-01, -1.1507e-01, -3.9121e-01, -4.0855e-01, 2.3960e-01],
        [-1.5193e-01, 1.0000e+05, 5.6770e-01, -3.8061e-02, -3.2549e-01, -1.3520e-01, -2.3034e-02, -4.9315e-01, -3.1906e-01, 2.7365e-01],
        [-1.3943e-01, -4.2179e-01, 1.0000e+05, -6.6652e-02, -3.7092e-01,-1.8476e-01, -3.6386e-01, -3.4232e-01, -4.0615e-01, 1.9351e-01],
        [-1.4723e-01, -4.1427e-01, 5.9977e-01, 1.0000e+05, -3.0970e-01,-2.3160e-01, -1.4783e-01, -3.0962e-01, -4.1965e-01, 2.9672e-01],
        [-1.3354e-01, -3.8455e-01, 6.1556e-01, -5.7763e-02, 1.0000e+05, -1.1570e-01, -8.4818e-02, -3.6836e-01, -4.4823e-01, 3.0249e-01],
        [3.7496e-03, -3.4185e-01, 5.9810e-01, -5.5021e-02, -2.8466e-01,1.0000e+05, -9.3396e-02, -3.5507e-01, -4.5072e-01, 3.1868e-01],
        [-1.9322e-01, -3.9075e-01, 5.9369e-01, -1.7286e-01, -2.3313e-01,-1.3478e-01, 1.0000e+05, -3.6746e-01, -4.6805e-01, 2.2758e-01],
        [-4.5005e-02, -4.8510e-01, 4.6703e-01, 1.2722e-01,-3.0572e-01,-1.7002e-01, -2.3059e-01, 1.0000e+05, -4.3838e-01,2.0406e-01],
        [-9.8894e-02, -4.0972e-01, 6.4475e-01, -1.1214e-01,-3.1715e-01, -1.9220e-01, -2.1514e-01, -4.0147e-01, 1.0000e+05,3.1069e-01],
        [-4.4040e-02, -3.6494e-01, 5.1821e-01, 9.7703e-02, -2.5698e-01, -1.5923e-01, -1.8586e-01, -2.8659e-01, -3.3394e-01, 1.0000e+05]]

        if opt.node_remove==True:
            z_phis[opt.node][opt.node]=-10000

        vector=mc_sampling(z_phis[opt.node])

        model.state_dict()['conv1.z_phi'][:]=torch.tensor(vector).to(device)
        model.state_dict()['conv2.z_phi'][:] = (torch.ones(20) * test_phi).to(device)
        model.state_dict()['fcs.0.z_phi'][:] = (torch.ones(320) * test_phi).to(device)
        model.state_dict()['fcs.2.z_phi'][:] = (torch.ones(100) * test_phi).to(device)
        # model.state_dict()['conv1.z_phi'][opt.node].detach()
        # model.state_dict()['conv1.z_phi'][opt.node]=100000.



    # model.convs[0].z_phi.register_hook(gradi)
    model.conv1.weights.register_hook(lambda grad: grad * 0)
    model.conv1.bias.register_hook(lambda grad: grad * 0)
    model.conv2.weights.register_hook(lambda grad: grad * 0)
    model.conv2.bias.register_hook(lambda grad: grad * 0)
    model.fcs[0].weights.register_hook(lambda grad: grad * 0)
    model.fcs[0].bias.register_hook(lambda grad: grad * 0)
    model.fcs[2].weights.register_hook(lambda grad: grad * 0)
    model.fcs[2].bias.register_hook(lambda grad: grad * 0)

    model.conv2.z_phi.register_hook(lambda grad: grad * 0)
    model.fcs[0].z_phi.register_hook(lambda grad: grad * 0)
    model.fcs[2].z_phi.register_hook(lambda grad: grad * 0)

    return model




def train(**kwargs):
    global device, vis
    if opt.seed is not None:
        setup_seed(opt.seed)
    config_str = opt.parse(kwargs)


    device = torch.device("cuda" if opt.use_gpu else "cpu")

    vis = Visualizer(opt.log_dir, opt.model, current_time, opt.title_note)
    # log all configs
    vis.log('config', config_str)

    # load data set
    train_loader, val_loader, num_classes = getattr(dataset, opt.dataset)(opt.batch_size * opt.gpus)
    # load model
    model = getattr(models, opt.model)(lambas=opt.lambas, num_classes=num_classes, weight_decay=opt.weight_decay).to(
        device)


    model=get_external_weights(model, "train")

    if opt.gpus > 1:
        model = nn.DataParallel(model)

    # define loss function
    def criterion(output, target_var):
        loss = nn.CrossEntropyLoss().to(device)(output, target_var)
        reg_loss = model.regularization() if opt.gpus <= 1 else model.module.regularization()
        total_loss = (loss + reg_loss).to(device)
        return total_loss

    # load optimizer and scheduler
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters() if opt.gpus <= 1 else model.module.parameters(), opt.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=opt.lr_decay, patience=15)
        scheduler = None
        print('Optimizer: Adam, lr={}'.format(opt.lr))
    elif opt.optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters() if opt.gpus <= 1
                                    else model.module.parameters(), opt.lr, momentum=opt.momentum, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.schedule_milestone,
                                                         gamma=opt.lr_decay)
        print('Optimizer: Momentum, lr={}, momentum'.format(opt.lr, opt.momentum))
    else:
        print('No optimizer')
        return

    loss_meter = meter.AverageValueMeter()
    accuracy_meter = meter.ClassErrorMeter(accuracy=True)
    # create checkpoints dir
    directory = '{}/{}_{}'.format(opt.checkpoints_dir, opt.model, current_time)
    if not os.path.exists(directory):
        os.makedirs(directory)
    total_steps = 0
    counter_plot=0
    for epoch in range(opt.start_epoch, opt.max_epoch) if opt.verbose else tqdm(range(opt.start_epoch, opt.max_epoch)):
        model.train() if opt.gpus <= 1 else model.module.train()
        loss_meter.reset()
        accuracy_meter.reset()
        for ii, (input_, target) in enumerate(train_loader):
            input_, target = input_.to(device), target.to(device)
            optimizer.zero_grad()
            score, diff = model(input_, target, opt.node)
            #model.conv1.z_phi[0]=-10000

            #loss = criterion(score, target)
            #loss.backward()
            optimizer.step()



            #loss_meter.add(loss.cpu().data)
            accuracy_meter.add(score.data, target.data)

            e_fl, e_l0 = model.get_exp_flops_l0() if opt.gpus <= 1 else model.module.get_exp_flops_l0()
            vis.plot('stats_comp/exp_flops', e_fl, total_steps)
            vis.plot('stats_comp/exp_l0', e_l0, total_steps)
            total_steps += 1

            if (ii % 500 ==0):
                counter_plot+=1
                #plt.scatter(counter_plot, diff.detach().cpu().numpy())
                #plt.pause(0.05)
                #print(model.conv1.z_phi.grad)
                print(model.conv1.z_phi)

            if (model.beta_ema if opt.gpus <= 1 else model.module.beta_ema) > 0.:
                model.update_ema() if opt.gpus <= 1 else model.module.update_ema()

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('train/loss', loss_meter.value()[0])
                vis.plot('train/accuracy', accuracy_meter.value()[0])
                if opt.verbose:
                    print("epoch:{epoch},lr:{lr},loss:{loss:.2f},train_acc:{train_acc:.2f}"
                      .format(epoch=epoch, loss=loss_meter.value()[0],
                              train_acc=accuracy_meter.value()[0],
                              lr=optimizer.param_groups[0]['lr']))


        # save model
        if epoch % 10 == 0 or epoch == opt.max_epoch - 1:
            torch.save(model.state_dict(), directory + '/{}.model'.format(epoch))
        # validate model
        # val_accuracy, val_loss = val(model, val_loader, criterion)
        #
        # vis.plot('val/loss', val_loss)
        # vis.plot('val/accuracy', val_accuracy)

        # update lr
        # if scheduler is not None:
        #     if isinstance(optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         scheduler.step(val_loss)
        #     else:
        #         scheduler.step(epoch)
        # if opt.verbose:
        #     print("epoch:{epoch},lr:{lr},loss:{loss:.2f},val_acc:{val_acc:.2f},prune_rate:{pr:.2f}"
        #           .format(epoch=epoch, loss=loss_meter.value()[0], val_acc=val_accuracy, lr=optimizer.param_groups[0]['lr'],
        #                   pr=model.prune_rate() if opt.gpus <= 1 else model.module.prune_rate()))
        # for (i, num) in enumerate(model.get_expected_activated_neurons() if opt.gpus <= 1
        #                           else model.module.get_expected_activated_neurons()):
        #     vis.plot("Training_layer/{}".format(i), num)
        # vis.plot('lr', optimizer.param_groups[0]['lr'])
    #plt.show()


def val(model, dataloader, criterion):
    model.eval() if opt.gpus <= 1 else model.module.eval()
    loss_meter = meter.AverageValueMeter()
    accuracy_meter = meter.ClassErrorMeter(accuracy=True)
    for ii, data in enumerate(dataloader):
        input_, label = data
        input_, label = input_.to(device), label.to(device)
        score = model(input_)
        accuracy_meter.add(score.data.squeeze(), label.long())
        loss = criterion(score, label)
        loss_meter.add(loss.cpu().data)

    for (i, num) in enumerate(model.get_activated_neurons() if opt.gpus <= 1 else model.module.get_activated_neurons()):
        vis.plot("val_layer/{}".format(i), num)

    for (i, z_phi) in enumerate(model.z_phis()):
        if opt.hardsigmoid:
            vis.hist("hard_sigmoid(phi)/{}".format(i), F.hardtanh(opt.k * z_phi / 7. + .5, 0, 1).cpu().detach().numpy())
        else:
            vis.hist("sigmoid(phi)/{}".format(i), torch.sigmoid(opt.k * z_phi).cpu().detach().numpy())

    vis.plot("prune_rate", model.prune_rate() if opt.gpus <= 1 else model.module.prune_rate())
    return accuracy_meter.value()[0], loss_meter.value()[0]



def test(**kwargs):
    opt.parse(kwargs)
    global device, vis
    device = torch.device("cuda" if opt.use_gpu else "cpu")
    vis = Visualizer(opt.log_dir, opt.model, current_time)
    # load model
    model = getattr(models, opt.model)(lambas=opt.lambas).to(device)
    # load data set
    train_loader, val_loader, num_classes = getattr(dataset, opt.dataset)(opt.batch_size * opt.gpus)

    # define loss function
    def criterion(output, target_var):
        loss = nn.CrossEntropyLoss().to(device)(output, target_var)
        total_loss = (loss + model.regularization() if opt.gpus <= 1 else model.module.regularization()).to(device)
        return total_loss

    if len(opt.load_file) > 0:
        #model.load_state_dict(torch.load(opt.load_file))
        for name, param in model.named_parameters():
            print (name, param.shape)
            #param.data[0]=0.5 #example how change

        model=get_external_weights(model, "test")

        #-1 everything is pruned

        print(model.state_dict()['conv1.z_phi'][:])

        val_accuracy, val_loss = val(model, val_loader, criterion)
        print("loss:{loss:.2f},val_acc:{val_acc:.2f},prune_rate:{pr:.2f}"
              .format(loss=val_loss, val_acc=val_accuracy,
                      pr=model.prune_rate() if opt.gpus <= 1 else model.module.prune_rate()))

        val_accuracy1=val_accuracy

        model.conv1.z_phi[opt.node]=torch.tensor(0)

        print(model.state_dict()['conv1.z_phi'][:])

        val_accuracy, val_loss = val(model, val_loader, criterion)
        print("loss:{loss:.2f},val_acc:{val_acc:.2f},prune_rate:{pr:.2f}"
              .format(loss=val_loss, val_acc=val_accuracy,
                      pr=model.prune_rate() if opt.gpus <= 1 else model.module.prune_rate()))

        val_accuracy0 = val_accuracy
        difference=val_accuracy1-val_accuracy0
        with open("results/shapley_switches.txt", 'a') as file:
            file.write(str(difference)+ "\n")






def help():
    '''help'''
    print('''
    usage : python main.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --model=ARMLeNet5 --dataset=mnist --lambas="[.1,.1,.1,.1]" --optimizer=adam --lr=0.001
            python {0} test --model=ARMLeNet5 --dataset=mnist --lambas="[.1,.1,.1,.1]" --load_file="checkpoints/ARMLeNet5_2019-06-19 14:27:03/0.model"
            python {0} train --model=ARMWideResNet --dataset=cifar10 --lambas=.001 --optimizer=momentum --lr=0.1 --schedule_milestone="[60,120]"
            python {0} help
            
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire
    fire.Fire({'train': train, 'test': test, 'help': help})
