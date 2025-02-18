from PIL import Image
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from utils import AverageMeter
from glob import glob
import os, sys
os.chdir(sys.path[0])
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Import Sionna
import sionna
from sionna.mimo import lmmse_equalizer
import tensorflow as tf
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
from einops import rearrange
from torchvision import io, datasets
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

plt.rcParams["savefig.bbox"] = 'tight'

# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)
seed = np.random.RandomState(42)

def calculate_psnr(X, Y):
    mse = torch.nn.MSELoss()(X*1. , Y*1.)
    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
    return psnr

def save_image(raw_img, de_img, name):
        raw_img = raw_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        de_img = de_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        raw_img = raw_img / 255. if raw_img.max() > 1.1 else raw_img
        de_img = de_img / 255. if de_img.max() > 1.1 else de_img

        plt.figure(figsize=(7,15))

        plt.subplot(2,1,1)
        plt.imshow(raw_img)
        plt.axis('off')

        plt.subplot(2,1,2)
        plt.imshow(de_img)
        plt.axis('off')

        plt.savefig('%s.png' % name)
        plt.close()

def dec2bin(x, bits):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float().flatten()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


class Datasets(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = []
        for dir in self.data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        self.imgs.sort()


    def __getitem__(self, item):
        image_ori = self.imgs[item]
        image = Image.open(image_ori).convert('RGB')
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).byte())])
        transformed = self.transform(image)
        # transforms.ToPILImage()(transformed).save('test.png')
        return transformed
    def __len__(self):
        return len(self.imgs)
    
# 将 PIL 图像转换为 Tensor
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为 [0, 1] 范围的浮点数
    transforms.Lambda(lambda x: (x * 255).byte())  # 转换为 [0, 255] 范围的 uint8
])
multiple_snr = [10, 7]

test_dataset = datasets.CIFAR10(root="../../datasets/CIFAR10/",
                                train=False,
                                transform=transform,
                                download=False)
indices = np.random.choice(len(test_dataset), 100, replace=False)
subset_dataset = Subset(test_dataset, indices)

kodak_dataset = Datasets(["../../datasets/Kodak/"])

test_loader = DataLoader(dataset=subset_dataset,
                            batch_size=1,
                            shuffle=False)
test_loader_all = DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=False)
kodak_loader = DataLoader(dataset=kodak_dataset,
                            batch_size=1,
                            shuffle=False)

demapping_method = "app" # try "max-log"
ldpc_cn_type = "boxplus" # try also "minsum"
constellation = sionna.mapping.Constellation("qam", 4)
mapper = sionna.mapping.Mapper(constellation=constellation)
channel_type = 'rayleigh'
if channel_type == 'awgn':
    channel = sionna.channel.AWGN()
else:
    channel = sionna.channel.FlatFadingChannel(1, 1, return_channel=True)
demapper = sionna.mapping.Demapper(demapping_method,
                                   constellation=constellation)


def cifar_after_channel():
    for i in range(len(multiple_snr)):
        psnrs, cbrs = [AverageMeter() for _ in range(2)]
        for j, data in enumerate(tqdm(test_loader)):
            raw_img, _ = data
            raw_img = torch.squeeze(raw_img, 0)
            # JPEG encode
            en_img = io.encode_jpeg(raw_img, 75)
            en_img_b = dec2bin(en_img, 8)
            en_img_b_tf = tf.convert_to_tensor(en_img_b.numpy())
            k_ldpc = en_img_b_tf.shape[0]
            coderate = 0.5
            n_ldpc = int(k_ldpc / coderate)
            en_img_b_tf = tf.reshape(en_img_b_tf, (1, en_img_b_tf.shape[0]))

            # ldpc config
            ldpc_encoder = sionna.fec.ldpc.encoding.LDPC5GEncoder(k_ldpc, n_ldpc)
            ldpc_decoder = sionna.fec.ldpc.decoding.LDPC5GDecoder(ldpc_encoder,
                                                            hard_out=True, cn_type=ldpc_cn_type,
                                                            num_iter=20)
            # communication process
            c = ldpc_encoder(en_img_b_tf)
            x = mapper(c)
            no = sionna.utils.ebnodb2no(multiple_snr[i], 4, coderate)
            if channel_type == 'rayleigh':
                y, h = channel([tf.reshape(x, (x.shape[1],x.shape[0])), no])
                s = tf.cast(no*tf.eye(1, 1), y.dtype)
                x_hat, no_eff = lmmse_equalizer(y, h, s)
                llr = demapper([tf.reshape(x_hat,(x_hat.shape[1],x_hat.shape[0])), no])
            if channel_type == 'awgn':
                y = channel([x, no])
                llr = demapper([y, no])
            b_hat = ldpc_decoder(llr)

            # calculate BERs
            c_hat = tf.cast(tf.less(0.0, llr), tf.float32) # hard-decided bits before dec.
            ber_uncoded = sionna.utils.metrics.compute_ber(c, c_hat)
            ber_coded = sionna.utils.metrics.compute_ber(en_img_b_tf, b_hat)
            print("BER uncoded = {:.3f} at EbNo = {:.1f} dB".format(ber_uncoded, multiple_snr[i]))
            print("BER after decoding = {:.3f} at EbNo = {:.1f} dB".format(ber_coded, multiple_snr[i]))
            b_hat = tf.squeeze(b_hat, axis=0)
            img_c = torch.tensor(b_hat.numpy())
            img_c = rearrange(img_c, '(x b) -> x b', b=8)
            img_c = bin2dec(img_c, 8)
            img_c = img_c.to(torch.uint8)

            try:
                de_img = io.decode_jpeg(img_c)
                cbrs.update(en_img.nbytes / raw_img.nbytes / coderate)
                psnrs.update(calculate_psnr(raw_img, de_img))
            except:
                cbrs.update(en_img.nbytes / raw_img.nbytes / coderate)
                psnrs.update(0)

        print("snr:", multiple_snr[i])
        print("cbrs:", cbrs.avg)
        print("psnrs:", psnrs.avg)

def cifar_no_channel():
    for i in range(30, 100, 10):
        psnrs, cbrs = [AverageMeter() for _ in range(2)]
        for j, data in enumerate(tqdm(test_loader_all)):
            raw_img, _ = data
            raw_img = torch.squeeze(raw_img, 0)
            # JPEG encode
            en_img = io.encode_jpeg(raw_img, i)

            de_img = io.decode_jpeg(en_img)
            cbrs.update(en_img.nbytes / raw_img.nbytes / 0.5)
            psnrs.update(calculate_psnr(raw_img, de_img))
        print("quality:", i)
        print("cbrs:", cbrs.avg)
        print("psnrs:", psnrs.avg)
        cbrs.clear()
        psnrs.clear()



def kodak_no_channel():
    for i in range(1, 100, 10):
        psnrs, cbrs = [AverageMeter() for _ in range(2)]
        for j, data in enumerate(tqdm(kodak_loader)):
            raw_img = data
            raw_img = torch.squeeze(raw_img, 0)
            # JPEG encode
            en_img = io.encode_jpeg(raw_img, i)
            de_img = io.decode_jpeg(en_img)
            cbrs.update(en_img.nbytes / raw_img.nbytes / 0.5)
            psnrs.update(calculate_psnr(raw_img, de_img))
            save_image(raw_img, de_img, 'test')
            
        print("snr:", i)
        print("cbrs:", cbrs.avg)
        print("psnrs:", psnrs.avg)

def test_channel():
    # system parameters
    n_ldpc = 500 # LDPC codeword length
    k_ldpc = 250 # number of info bits per LDPC codeword
    coderate = k_ldpc / n_ldpc
    num_bits_per_symbol = 4 # number of bits mapped to one symbol (cf. QAM)
    demapping_method = "app" # try "max-log"
    ldpc_cn_type = "boxplus" # try also "minsum" 
    binary_source = sionna.utils.BinarySource()
    encoder = sionna.fec.ldpc.encoding.LDPC5GEncoder(k_ldpc, n_ldpc)
    constellation = sionna.mapping.Constellation("qam", num_bits_per_symbol)
    mapper = sionna.mapping.Mapper(constellation=constellation)
    channel = sionna.channel.AWGN()
    demapper = sionna.mapping.Demapper(demapping_method,
                                    constellation=constellation)
    decoder = sionna.fec.ldpc.decoding.LDPC5GDecoder(encoder,
                                                    hard_out=True, cn_type=ldpc_cn_type,
                                                    num_iter=20)
    # simulation parameters
    batch_size = 1000
    ebno_db = 4

    # Generate a batch of random bit vectors
    b = binary_source([batch_size, k_ldpc])

    # Encode the bits using 5G LDPC code
    print("Shape before encoding: ", b.shape)
    c = encoder(b)
    print("Shape after encoding: ", c.shape)

    # Map bits to constellation symbols
    x = mapper(c)
    print("Shape after mapping: ", x.shape)

    # Transmit over an AWGN channel at SNR 'ebno_db'
    no = sionna.utils.ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
    y = channel([x, no])
    print("Shape after channel: ", y.shape)

    # Demap to LLRs
    llr = demapper([y, no])
    print("Shape after demapping: ", llr.shape)

    # LDPC decoding using 20 BP iterations
    b_hat = decoder(llr)
    print("Shape after decoding: ", b_hat.shape)

    # calculate BERs
    c_hat = tf.cast(tf.less(0.0, llr), tf.float32) # hard-decided bits before dec.
    ber_uncoded = sionna.utils.metrics.compute_ber(c, c_hat)

    ber_coded = sionna.utils.metrics.compute_ber(b, b_hat)

    print("BER uncoded = {:.3f} at EbNo = {:.1f} dB".format(ber_uncoded, ebno_db))
    print("BER after decoding = {:.3f} at EbNo = {:.1f} dB".format(ber_coded, ebno_db))
    print("In total {} bits were simulated".format(np.size(b.numpy())))


if __name__ == "__main__":
    cifar_after_channel()