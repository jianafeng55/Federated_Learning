import logging
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import datasets


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class MNIST_truncated(data.Dataset):
    def __init__(
        self,
        root,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        cifar_dataobj = datasets.MNIST(
            self.root,
            self.train,
            self.transform,
            self.target_transform,
            self.download,
        )

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)



def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


def load_mnist_data(args):
    # train_transform, test_transform = _data_transforms_cifar10()
    mnist_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )
    train_dataset = datasets.MNIST(
        args.data_dir,
        train=True,
        download=True,
        transform=mnist_transform,
    )

    # validation_dataset = datasets.MNIST(
    #     args.data_dir, train=False, transform=transforms.ToTensor()
    # )
    X_train, y_train = train_dataset.data, train_dataset.targets
    # X_test, y_test = validation_dataset.data, validation_dataset.targets

    return X_train, y_train


def partition_data(args):
    logging.info("*********partition data***************")
    X_train, y_train = load_mnist_data(args)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]
    num_classes = len(np.unique(y_train))

    if args.partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, args.total_num_clients)
        net_dataidx_map = {
            i: batch_idxs[i] for i in range(args.total_num_clients)
        }

    elif args.partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(args.total_num_clients)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(
                    np.repeat(args.alpha, args.total_num_clients)
                )
                ## Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / args.total_num_clients)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(
                    int
                )[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(
                        idx_batch, np.split(idx_k, proportions)
                    )
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(args.total_num_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return (
        num_classes,
        net_dataidx_map,
        traindata_cls_counts,
    )


def get_local_train_data_loader_MNIST(args, dataidxs_train):
    mnist_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )
    train_dataset = MNIST_truncated(
        args.data_dir,
        train=True,
        download=True,
        dataidxs=dataidxs_train,
        transform=mnist_transform,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    return train_loader


def get_valid_data_loader_MNIST(args):
    mnist_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )
    validation_dataset = MNIST_truncated(
        args.data_dir, train=False, transform=mnist_transform, download=True
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=args.batch_size, shuffle=False
    )
    return validation_loader


def load_partition_data_mnist(args):
    """
    Paritions the MNIST data based on the specified args
    Returns:
        train_data_num: Number of data points in overall Train dataset.
        test_data_num: Number of data points in overall Test dataset.
        test_data_global: Data loader for global test data.
        data_local_num_dict: Dictionary where key is client_id and value is number of train data points for client's parition.
        train_data_local_dict: Dictionary where key is client_id and value is data loader of client's parition.
        class_num: Number of classes in the data.
    """
    class_num, net_dataidx_map, traindata_cls_counts = partition_data(args)
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum(
        [len(net_dataidx_map[r]) for r in range(args.total_num_clients)]
    )

    test_data_global = get_valid_data_loader_MNIST(args)
    # logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()

    for client_idx in range(args.total_num_clients):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info(
            "client_idx = %d, local_sample_number = %d"
            % (client_idx, local_data_num)
        )

        # training batch size = 64; algorithms batch size = 32
        train_data_local = get_local_train_data_loader_MNIST(args, dataidxs)
        logging.info(
            "client_idx = %d, batch_num_train_local = %d"
            % (client_idx, len(train_data_local))
        )
        train_data_local_dict[client_idx] = train_data_local
    return (
        train_data_num,
        test_data_num,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        class_num,
    )
