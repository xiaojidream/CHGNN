# coding=utf-8
import sys
import pandas as pd
sys.path.append("..")
import matplotlib as mpl
mpl.use('Agg')

from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.encoder import *
from patch_dataset import Dataset as PatchDataset
from models.Encoder_diffusion import Diffusion, Encoder

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--num_classes', default=2, type=int,  # num_classes
                    help='numbers of classes (default: 1)')
parser.add_argument('--optimizer', default='Adam', type=str,  # Adam
                    help='optimizer (Adam)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--tensorboard', default=True,
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--use_cuda', default=True,
                    help='whether to use_cuda(default: True)')
parser.add_argument('--batch_size', default=500, type=int,
                    help='mini-batch size (default: 4000)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--epoch', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--seed', default=2, type=int, help='random seed(default: 1)')  #
parser.add_argument("--gpu", type=str, default="0", metavar="N", help="input visible devices for training (default: 0)")
parser.add_argument("--dstDir", default="./results/1-23", type=str)


args = parser.parse_args()



def main():
    global use_cuda, writer
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.tensorboard:
        writer = SummaryWriter(args.dstDir+'/tensorboard/')

    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")

    if args.seed >= 0:
        seed_torch(args.seed)

    model = Encoder(time_emb_channels=128)
    if use_cuda:
        model = model.cuda()

    model_path = os.path.join(args.dstDir, 'encoder')
    if not os.path.exists(model_path):
        os.makedirs(model_path)


    weights = torch.FloatTensor([2.0, 1.0]).cuda()
    criterion = nn.CrossEntropyLoss(weight=weights)

    # define optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    nesterov=True, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    else:
        print('Please choose true optimizer.')
        return 0


    # 获取训练集和验证集的数据patch
    train_datasets = PatchDataset(path=r'.\data\encoder_data\train.xlsx')
    val_datasets = PatchDataset(path=r'.\data\encoder_data\val.xlsx')
    test_datasets = PatchDataset(path=r'.\data\encoder_data\test.xlsx')
    train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=args.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_datasets, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=args.batch_size, shuffle=False)

    best_prec = 0
    for epoch in range(args.start_epoch, args.epoch):
        # train for one epoch
        train_losses, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.fold_index, device)
        val_losses, val_acc, prec1 = validate(val_loader, model, criterion, epoch, args.fold_index, device)

        if args.tensorboard:
            writer.add_scalars('data' + str(args.fold_index) + '/loss',
                               {'train_loss': train_losses.avg, 'val_loss': val_losses.avg}, epoch)
            writer.add_scalars('data' + str(args.fold_index) + '/Accuracy',
                               {'train_acc': train_acc.avg, 'val_acc': val_acc.avg}, epoch)


        is_best = prec1 > best_prec
        print('prec:', prec1)
        print('best_prec:', best_prec)
        print('is best:', is_best)
        if is_best == 1:
            best_prec = max(prec1, best_prec)  #
            torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec,
        }, os.path.join(model_path, 'best_model.pt'))






def train(train_loader, model, criterion, optimizer, epoch, fold, device):
    """Train for one epoch on the training set"""
    train_losses = AverageMeter()
    train_acc = AverageMeter()
    # switch to train mode
    model.train()
    with tqdm(train_loader, ncols=130) as t:
        for i, (input, target) in enumerate(t):
            t.set_description("train epoch %s" % epoch)
            if use_cuda:
                target = target.type(torch.FloatTensor).cuda()
                input = input.type(torch.FloatTensor).cuda()

            diffusion = Diffusion(noise_steps=50, img_size=50, device=device)
            time_step = diffusion.sample_timesteps(input.shape[0]).to(device)
            input, noise = diffusion.noise_images(input, time_step)

            optimizer.zero_grad()
            output = model(input, time_step)
            train_loss = criterion(output.squeeze(), target.long())

            train_losses.update(train_loss.item(), input.size(0))

            acc = accuracy(output.data, target)
            train_acc.update(acc, input.size(0))

            # compute gradient
            train_loss.backward()
            optimizer.step()

            t.set_postfix({
                'loss': '{loss.val:.4f}({loss.avg:.4f})'.format(loss=train_losses),
                'Acc': '{acc.val:.4f}({acc.avg:.4f})'.format(acc=train_acc)}
            )

    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('data' + str(fold) + '/train_loss', train_losses.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/train_acc', train_acc.avg, epoch)
    return train_losses, train_acc


def validate(val_loader, model, criterion, epoch, fold, device):
    """Perform validation on the validation set"""
    val_losses = AverageMeter()
    val_acc = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        with tqdm(val_loader, ncols=130) as t:
            for i, (input, target) in enumerate(t):
                t.set_description("valid epoch %s" % epoch)

                if use_cuda:
                    target = target.type(torch.FloatTensor).cuda()
                    input = input.type(torch.FloatTensor).cuda()

                diffusion = Diffusion(noise_steps=50, img_size=50, device=device)
                time_step = diffusion.sample_timesteps(input.shape[0]).to(device)
                input, noise = diffusion.noise_images(input, time_step)
                # compute output
                # output = model(input.to(torch.float32))
                output = model(input, time_step)
                val_loss = criterion(output.squeeze(), target.to(torch.float32).long())
                val_losses.update(val_loss.item(), input.size(0))

                # -------------------------------------Accuracy--------------------------------- #
                acc = accuracy(output.data, target)
                val_acc.update(acc, input.size(0))

                t.set_postfix({
                    'loss': '{loss.val:.4f}({loss.avg:.4f})'.format(loss=val_losses),
                    'Acc': '{acc.val:.4f}({acc.avg:.4f})'.format(acc=val_acc)}
                )

    if args.tensorboard:
        writer.add_scalar('data' + str(fold) + '/val_loss', val_losses.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/val_acc', val_acc.avg, epoch)
    return val_losses, val_acc, val_acc.avg


def test(test_loader, model, mode):
    checkpoint = torch.load(r'.\encoder\best_model.pt')
    pretrained_dict = checkpoint['state_dict']
    model.load_state_dict(pretrained_dict)
    model.eval()
    y_true = []
    pred = []
    ids = []
    locs = []

    features = []
    device = torch.device("cuda:0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")

    with torch.no_grad():
        with tqdm(test_loader, ncols=130) as t:
            for i, (input, target, id, loc) in enumerate(t):
                t.set_description("test:{}".format(mode))
                if use_cuda:
                    target = target.type(torch.FloatTensor).cuda()
                    input = input.type(torch.FloatTensor).cuda()


                # compute output
                # output = model(input)
                diffusion = Diffusion(noise_steps=50, img_size=50, device=device)
                time_step = diffusion.sample_timesteps(input.shape[0]).to(device)
                input, noise = diffusion.noise_images(input, time_step)
                # compute output
                # output = model(input.to(torch.float32))
                output = model(input, time_step)

                feature = model.feature(input, time_step)
                feature = feature.detach().cpu().numpy()
                features.extend(feature)
                y_true.extend(target)
                pred.extend(output.data.cpu().numpy())
                ids.extend(id)
                locs.extend(loc)


    acc = accuracy(np.array(pred), np.array(y_true))
    features = np.array(features)
    probs = np.array(pred)
    ids = np.array(ids)
    locs = np.array(locs)
    y_true = np.array(torch.tensor(y_true).tolist())
    print(acc)
    return features, probs, ids, locs, y_true


def save_patch_probs_pred_test(features, probs, ids, locs, labels):
    patch_prob_filename = os.path.join(r'E:\yyx\RanK\graph-slide-copy\data\encoder_data', 'test_probs.csv')
    feat_head = ['feat'+str(x) for x in range(32)]
    info = []
    for i in range(len(ids)):
        loc = locs[i]
        # file_name = loc.split('-')[0]
        file_name = loc
        xmin = loc.split('-')[2]
        xmax = loc.split('-')[3]
        ymin = loc.split('-')[4]
        ymax = loc.split('-')[5]
        z = loc.split('-')[6].split('.')[0]
        label = labels[i]

        info.append([ids[i], file_name, str(probs[i][1])] + [xmin, xmax, ymin, ymax, z, label] + features[i].tolist())

    df = pd.DataFrame(info, columns=['id', 'file_name', 'prob', 'x_min', 'x_max', 'y_min', 'y_max', 'z', 'label']+feat_head)
    df.to_csv(patch_prob_filename, index=False)







def accuracy(output, target):
    # output_np = output.cpu().numpy()
    output_np = output

    # output_np[output_np > 0.5] = 1
    # output_np[output_np <= 0.5] = 0
    pred = np.zeros(output_np.shape[0])
    pred[output_np[:, 0] < output_np[:, 1]] = 1
    # target_np = target.cpu().numpy()
    target_np = target

    right = (pred == target_np)
    acc = np.sum(right) / output.shape[0]

    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




if __name__ == '__main__':
    args = parser.parse_args()
    main()

    # test_datasets = PatchDataset(path=r'.\data\encoder_data\stride30\test.xlsx')
    # test_loader = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=args.batch_size, shuffle=False)
    # # 用于测试生成特征向量
    # features, probs, ids, locs, labels = test(test_loader, model, 'test')
    # save_patch_probs_pred_test(features, probs, ids, locs, labels)
