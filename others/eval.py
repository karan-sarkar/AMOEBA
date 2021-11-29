import torch.optim as optim
import matplotlib.pyplot as plt
import time
import argparse
from sklearn.manifold import TSNE



from others.bdd import *
from others.ssd import *


EPOCHS = 100
NUM_CLASSES = 10
root_anno_path = "bdd100k_labels_detection20"


parser = argparse.ArgumentParser()
parser.add_argument('--attribute', type=str, default = 'timeofday')
parser.add_argument('--source', type=str, default = 'daytime')
parser.add_argument('--target', type=str, default = 'night')
parser.add_argument('--batch_size', type=int, default = 32)
parser.add_argument('--lr', type=float, default = 0.001)
parser.add_argument('--momentum', type=float, default = 0.9)
parser.add_argument('--decay', type=float, default = 0.0005)
parser.add_argument('--clipping', type=float, default = 5.0)
parser.add_argument('--iter', type=int, default = 4)
parser.add_argument('--source_iter', type=int, default = 1)
parser.add_argument('--ckpt', type=int, default = -1)
parser.add_argument('--sample', type=int, default = -1)
parser.add_argument('--source_sample', type=int, default = -1)
parser.add_argument('--test_first', type=int, default = -1)

args = parser.parse_args()

ATTRIBUTE = args.attribute
SOURCE_FLAG = args.source
TARGET_FLAG = args.target

BATCH_SIZE = args.batch_size
lr = args.lr
momentum = args.momentum
weight_decay = args.decay
clipping = args.clipping
iterations = args.iter
class_iterations = args.source_iter
mod = args.ckpt
max_comp = 1
min_comp = 1
sample = args.sample
source_sample = args.source_sample
args.ckpt =None
path = str((SOURCE_FLAG, source_sample, TARGET_FLAG, sample))
torch.cuda.empty_cache()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

test_first = False
if args.test_first > 0:
    test_first = True

root_img_path = "../fcos-pytorch/bdd100k/images/100k"
root_anno_path = "bdd100k_labels_detection20/bdd100k/labels/detection20"

train_img_path = root_img_path + "/train/"
val_img_path = root_img_path + "/val/"

train_anno_json_path = root_anno_path + "/det_v2_train_release.json"
val_anno_json_path = root_anno_path + "/det_v2_val_release.json"

with open(train_anno_json_path, "r") as file:
    train_data = json.load(file)
print(len(train_data))
with open(val_anno_json_path, "r") as file:
    test_data = json.load(file)
print(len(test_data))

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def make_dataset(train, flag):
    if train:
        data = train_data
        json_file = train_anno_json_path
        header = train_img_path
    else:
        data = test_data
        json_file = val_anno_json_path
        header = val_img_path
    
    img_list = []
    idx = []
    for i in tqdm.tqdm(range(len(data))):
        if data[i]['attributes'][ATTRIBUTE] == flag and data[i]['labels'] != None:
            img_list.append(header + data[i]['videoName'] + '.jpg')
            idx.append(i)
    if sample > 0 and flag == TARGET_FLAG and train:
        np.random.seed(0)
        perm = np.random.choice(len(idx), sample)
        idx = [idx[p] for p in perm]
        img_list = [img_list[p] for p in perm]
    if source_sample > 0 and flag == SOURCE_FLAG and train:
        np.random.seed(0)
        perm = np.random.choice(len(idx), source_sample)
        idx = [idx[p] for p in perm]
        img_list = [img_list[p] for p in perm]
    dset = BDD(img_list, idx, json_file, train)
    return dset

source_train = make_dataset(True, SOURCE_FLAG)
source_test = make_dataset(False, SOURCE_FLAG)
target_train = make_dataset(True, TARGET_FLAG)
target_test = make_dataset(False, TARGET_FLAG)

def load(dset, sample, batch_size=0):
    if batch_size == 0:
        batch_size = BATCH_SIZE
    return torch.utils.data.DataLoader(dset,batch_size=batch_size,shuffle=True, collate_fn=dset.collate_fn)

def get_model(num_classes):
    model = SSD300(num_classes)
    return model.to(device)
        
        
jm = get_model(NUM_CLASSES)
jm = nn.DataParallel(jm)
opt = None
if mod >= 0:
    jm,opt = torch.load('mcd_bdd100k-9_' + path + str(mod) + ".pth")

        
params = list(jm.parameters()) 
if opt is None or type(opt) is not tuple:
    g = [p for n, p in jm.named_parameters() if 'pred_convs2' not in n]
    c = [p for n, p in jm.named_parameters() if 'pred_convs1' in n]
    d = [p for n, p in jm.named_parameters() if 'pred_convs2' in n]
    g_opt = optim.SGD(g, lr=lr, momentum=momentum, weight_decay=weight_decay)
    c_opt = optim.SGD(c, lr=lr, momentum=momentum, weight_decay=weight_decay)
    d_opt = optim.SGD(d, lr=lr, momentum=momentum, weight_decay=weight_decay)
    opt = (g_opt, c_opt, d_opt)

crit = MultiBoxLoss(priors_cxcy=jm.module.priors_cxcy.clone()).to(device)

def train(train_loader, test_loader, model, crit, optimizer, epoch, print_freq):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout
    criterion = crit
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    dlosses = AverageMeter()  # loss
    blosses = AverageMeter()
    tlosses = AverageMeter()
    start = time.time()
    
    (g_opt, c_opt, d_opt) = optimizer
    # Batches
    test_loader = load(test_loader, False)
    train_loader = load(train_loader, False)
    test = iter(test_loader)
    train = iter(train_loader)
    dlosses1 = []
    blosses1 = []
    dlosses2 = []
    blosses2 = []
    tsne = TSNE(n_components=2, random_state=0, verbose = 10)
    generator = nn.DataParallel(model.module.base)
    source_f = []
    target_f = []
    
    with torch.no_grad():
        for i in range(min(len(test_loader), len(train_loader))-2):
            '''
            try:
            '''
            (target_images, target_boxes, target_labels) = next(test)
            (source_images, source_boxes, source_labels) = next(train)
            '''
            except:
                break
                test = iter(load(test_loader, False))
                (target_images, target_boxes, target_labels) = next(test)
            '''
            data_time.update(time.time() - start)
            
            # Move to default device
            source_images = source_images.to(device)  # (batch_size (N), 3, 300, 300)
            source_boxes = [b.to(device) for b in source_boxes]
            source_labels = [l.to(device) for l in source_labels]
            
            target_images = target_images.to(device)  # (batch_size (N), 3, 300, 300)
            target_boxes = [b.to(device) for b in target_boxes]
            target_labels = [l.to(device) for l in target_labels]
            
            source_features = generator(source_images)
            source_features = torch.cat([x.view(BATCH_SIZE, -1) for x in source_features], 1).cpu().numpy()
            target_features = generator(target_images)
            target_features = torch.cat([x.view(BATCH_SIZE, -1) for x in target_features], 1).cpu().numpy()
            source_f.append(source_features)
            target_f.append(target_features)
            
            
            
            predicted_source_locs1, predicted_source_scores1, predicted_source_locs2, predicted_source_scores2 = model(source_images)  # (N, 8732, 4), (N, 8732, n_classes)
            predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2 = model(target_images)  # (N, 8732, 4), (N, 8732, n_classes)
            loss = criterion(predicted_source_locs1, predicted_source_scores1, source_boxes, source_labels)  # scalar
            loss += criterion(predicted_source_locs2, predicted_source_scores2, source_boxes, source_labels)
            bloss, dloss = criterion.discrep(predicted_target_locs1.detach(), predicted_target_scores1.detach(), predicted_target_locs2, predicted_target_scores2)
            bloss2, dloss2 = criterion.discrep(predicted_source_locs1.detach(), predicted_source_scores1.detach(), predicted_source_locs2, predicted_source_scores2)
            loss -= max_comp * (bloss + dloss - bloss2 - dloss2)
            dlosses1.append(float(dloss))
            blosses1.append(float(bloss))
            dlosses2.append(float(dloss2))
            blosses2.append(float(bloss2))
            
            dlosses.update(dloss.item(), target_images.size(0))
            blosses.update(bloss.item(), target_images.size(0))
            del predicted_source_locs1, predicted_source_scores1, predicted_source_locs2, predicted_source_scores2, loss
            del predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2, dloss, bloss
            
            
            del source_images, source_boxes, source_labels, target_images, target_boxes, target_labels
            
            batch_time.update(time.time() - start)
            start = time.time()

            optimizer = (g_opt, c_opt, d_opt)
            # Print status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})'
                      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})'
                      'Source Loss {loss.val:.4f} ({loss.avg:.4f})'
                      'Discrep Loss {dloss.val:.4f} ({dloss.avg:.4f})'
                      'Box Loss {bloss.val:.4f} ({bloss.avg:.4f})'
                      'Target Loss {tloss.val:.4f} ({tloss.avg:.4f})'.format(epoch, i, min(len(train), len(test)),
                                                                      batch_time=batch_time,
                                                                      data_time=data_time, loss=losses, dloss = dlosses, bloss = blosses, tloss = tlosses))
            if i % 5 == 0:
                torch.save((jm, optimizer), 'mcd_bdd100k-9_' + path +  str(epoch + mod + 1) + '.pth')
            
            if i > 100:
                break
    '''
    source_f = np.concatenate(source_f, 0)
    target_f = np.concatenate(target_f, 0)
    
    source_2d = tsne.fit_transform(source_f)
    target_2d = tsne.fit_transform(target_f)
    
    plt.scatter(source_2d[:, 0], source_2d[:, 1], c='r', label='Source Features')
    plt.scatter(target_2d[:, 0], target_2d[:, 1], c='b', label='Target Features')
    plt.legend()
    plt.savefig('AMOEBA-features.pdf')
    '''
    '''
    plt.clf()
    plt.style.use('seaborn-deep')

    plt.hist([np.array(dlosses2), np.array(dlosses1)], 30, label=['Source Class Discrepancy', 'Target Class Discrepancy'])
    plt.legend(loc='upper right', fontsize = 20)
    plt.xlabel('')
    plt.savefig('Supervised-class-discrep.pdf')
   '''
    plt.clf()
    
    plt.style.use('seaborn-deep')
    bins = np.linspace(0, 0.5, 30)

    plt.hist([np.array(blosses2), np.array(blosses1)], 30, label=[SOURCE_FLAG+' Box Discrepancy', TARGET_FLAG+' Box Discrepancy'])
    plt.xlabel('Bounding Box Discrepancy', fontsize = 20)
    plt.ylabel('Frequency', fontsize = 20)
    plt.legend(loc='upper right', fontsize = 20)

    plt.savefig('Supervised' +SOURCE_FLAG + TARGET_FLAG+ '-discrep.pdf')
                                                          

def test(test_loader, model, criterion, epoch):
    model.eval()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    tlosses = AverageMeter()
    start = time.time()
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    
    i = 1
    predicted_cars = 0
    true_cars = 0
    pbar = tqdm.tqdm(load(test_loader, False, batch_size=BATCH_SIZE))
    with torch.no_grad():# Batches
        for (target_images, target_boxes, target_labels) in pbar:
            #if i > 5:
            #    break
            target_images = target_images.to(device)  # (batch_size (N), 3, 300, 300)
            target_boxes = [b.to(device) for b in target_boxes]
            target_labels = [l.to(device) for l in target_labels]
            
            predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2 = model(target_images)
            
            det_boxes_batch, det_labels_batch, det_scores_batch = model.module.detect_objects(predicted_target_locs1, predicted_target_scores1,
                                                                                           min_score=0.01, max_overlap=0.45,
                                                                                           top_k=200)
            true_cars += sum([float(l.eq(5).sum()) for l in target_labels])
            predicted_cars += sum([float(l.eq(5).sum()) for l in det_labels_batch])
            pbar.set_description(str(predicted_cars/true_cars))
            
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(target_boxes)
            true_labels.extend(target_labels)
            
            del predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2
            del target_images, target_boxes, target_labels
            del det_boxes_batch, det_labels_batch, det_scores_batch
            i += 1
        
    
    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)
    print(APs)
    print(mAP)
                                                              
    
for epoch in range(EPOCHS):
    temp = MultiBoxLoss(priors_cxcy=crit.priors_cxcy.clone()).to(device)
    #if (epoch % 2 == 0 and epoch > 0) or (test_first and epoch == 0):
    #   test(target_test, jm, crit, epoch)
    #   test(source_test, jm, crit, epoch)
    train(source_train, target_train, jm, crit, opt, epoch, 1)
    torch.save((jm, opt), 'mcd_bdd100k-9_' + path + str(epoch + mod + 1) + '.pth')
    jm,opt = torch.load('mcd_bdd100k-9_' + path+ str(epoch + mod + 1) + ".pth")
    crit = temp
    source_train = make_dataset(True, SOURCE_FLAG)
    source_test = make_dataset(False, SOURCE_FLAG)
    target_train = make_dataset(True, TARGET_FLAG)
    target_test = make_dataset(False, TARGET_FLAG)
        
    
