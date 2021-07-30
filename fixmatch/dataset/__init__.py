import sys
sys.path.append('/nethome/jbang36/k_amoeba/')

from fixmatch.dataset.cifar import get_cifar10, get_cifar100
from fixmatch.dataset.bdd import get_bdd_fcos, get_bdd_fcos_new
from fixmatch.dataset.pascal_voc import (get_pascal_voc, get_pascal_bdd,
                                         get_pascal_bdd_day_night, get_pascal_bdd_day_night_cyclegan)



DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'bdd_fcos': get_bdd_fcos_new,
                   'pascal_bdd': get_pascal_bdd,
                   'pascal_voc': get_pascal_voc,
                   'pascal_bdd_day_night': get_pascal_bdd_day_night,
                   'pascal_bdd_day_night_cyclegan': get_pascal_bdd_day_night_cyclegan
                   }


