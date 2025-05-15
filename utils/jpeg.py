from PIL import Image
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from utils import AverageMeter
from glob import glob
import subprocess
import os, sys
os.chdir(sys.path[0])
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 获取当前文件（jpeg.py）的目录，并找到项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)  # 将项目根目录添加到模块搜索路径
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
from models.distortion import *
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


def bpg_compress(input_file, output_file, quality=30, bit_depth=8, color_space='ycbcr', extra_options=None):
    """
    调用 bpgenc，把 input_file 压缩成 BPG 存到 output_file。
    :param input_file: 输入图像路径，如 'input.png'
    :param output_file: 输出 BPG 文件路径，如 'output.bpg'
    :param quality: bpgenc 的 -q 参数，值越大质量越高，文件也越大
    :param bit_depth: bpgenc 的 -b 参数，默认 8 位，最大可到 14
    :param color_space: bpgenc 的 -c 参数，可选 'ycbcr', 'rgb', 'ycgco' 等
    :param extra_options: 列表，包含要额外传给 bpgenc 的其他选项
    """
    if extra_options is None:
        extra_options = []
    
    # bpgenc 常见选项:
    #   -o <outfile> : 指定输出文件
    #   -q <quality> : 质量 [0..51]
    #   -b <bit_depth>: 指定位深度 [8..14]
    #   -c <color_space> : 指定色彩空间，ycbcr(默认)/rgb/ycgco
    #   -f <format> : 指定像素格式(4:4:4,4:2:2,4:2:0等)
    #   -m <level>  : HEVC 编码复杂度 [0..9], 数值越大压缩速度越慢但效果更好
    #   -lossless   : 无损压缩(此时 -q,-b 可能失效，需同时指定rgb)
    #   其他更多选项见 bpgenc -h

    cmd = [
        'bpgenc',
        '-o', output_file,
        '-q', str(quality),
        '-b', str(bit_depth),
        '-c', color_space
    ] + extra_options + [input_file]
    
    # print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # print("BPG compression done. Output =>", output_file)

def bpg_decompress(input_bpg, output_file, extra_options=None):
    """
    调用 bpgdec，把 BPG 文件解压为普通图像，如 PNG/PPM/PGM 等。
    :param input_bpg: BPG 格式文件路径
    :param output_file: 解压后的图像文件路径
    :param extra_options: 其他选项
    """
    if extra_options is None:
        extra_options = []
    
    # bpgdec 常见选项:
    #   -o <outfile>: 指定输出文件
    #   -b <bit_depth>: 输出时使用多少位
    #   -f <format> : 输出图像格式： 'png'/'ppm'/'pgm'
    #   -alpha : 单独输出 alpha 通道
    #   -csky : (很少用)
    #   其他更多选项见 bpgdec -h

    cmd = [
        'bpgdec',
        '-o', output_file
    ] + extra_options + [input_bpg]

    # print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # print("BPG decompression done. Output =>", output_file)

def split_into_blocks(bits_tensor, max_k):
    """将比特流分割成最大为max_k的小块"""
    num_bits = bits_tensor.shape[1]
    num_blocks = (num_bits + max_k - 1) // max_k
    blocks = []
    for i in range(num_blocks):
        start = i * max_k
        end = min((i+1)*max_k, num_bits)
        block = bits_tensor[:, start:end]
        blocks.append(block)
    return blocks

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
multiple_snr = [7]

test_dataset = datasets.CIFAR10(root="../../datasets/CIFAR10/",
                                train=False,
                                transform=transform,
                                download=False)
# indices = np.random.choice(len(test_dataset), 100, replace=False)
indices = list(range(9000, 9050))
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
channel_type = 'awgn'  # 'awgn' or 'rayleigh'
if channel_type == 'awgn':
    channel = sionna.channel.AWGN()
else:
    channel = sionna.channel.FlatFadingChannel(1, 1, return_channel=True)
demapper = sionna.mapping.Demapper(demapping_method,
                                   constellation=constellation)

def ldpc_encode_blocks(blocks, ldpc_rate):
    encoded_blocks , ks, ns= [], [], []
    for block in blocks:
        k = block.shape[1]
        ks.append(k)
        n = int(k / ldpc_rate)
        ns.append(n)
        encoder = sionna.fec.ldpc.encoding.LDPC5GEncoder(k, n)
        encoded = encoder(block)
        encoded_blocks.append(encoded)
    return tf.concat(encoded_blocks, axis=1), ks, ns

def ldpc_decode_blocks(received_blocks, ldpc_rate, ks, ns):
    decoded_blocks = []
    start = 0
    for i in range(len(ks)):
        k = ks[i]
        n = ns[i]
        decoder = sionna.fec.ldpc.decoding.LDPC5GDecoder(
            sionna.fec.ldpc.encoding.LDPC5GEncoder(k, n),
            hard_out=True
        )
        decoded = decoder(received_blocks[0][:, start:start+n])
        decoded_blocks.append(decoded)
        start += n
    return tf.concat(decoded_blocks, axis=1)

def test_bpg_ldpc_channel_single_image(
    raw_img_tensor,
    bpg_quality=30,
    ldpc_rate=0.5,
    ebno_db=6.0,
    channel_type=channel_type):
    """
    对单张图执行:
      1) 保存PNG -> 调用bpgenc压缩 -> 读入BPG字节流
      2) BPG比特流 -> LDPC + 信道 -> BPG比特流
      3) 写回BPG文件 -> bpgdec 解码 -> 计算PSNR
    :param raw_img_tensor: 形状(C,H,W)，像素范围[0,255]的uint8 Tensor
    :param bpg_quality: bpgenc的 -q 参数
    :param ldpc_rate: LDPC 编码码率
    :param ebno_db: 信道Eb/No (dB)
    :param channel_type: 'awgn' 或 'rayleigh'
    :return: psnr, cbr
    """
    # 0) 先把 raw_img_tensor 存为临时PNG (bpgenc 读PNG更方便)
    #    如果原图本身就是一个文件，可以省略这步。
    temp_input_png = "temp_input.png"
    temp_bpg = "temp_output.bpg"
    temp_bpg_received = "temp_received.bpg"
    temp_recon_png = "temp_recon.png"

    # 保存图像为PNG文件
    pil_img = transforms.ToPILImage()(torch.squeeze(raw_img_tensor, 0))
    pil_img.save(temp_input_png)
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3)
    # 1) 调用 bpgenc 压缩成 BPG 文件
    bpg_compress(
        input_file=temp_input_png,
        output_file=temp_bpg,
        quality=bpg_quality,
        bit_depth=8,
        color_space='ycbcr',
        extra_options=['-f','444']  # 比如强制4:4:4
    )

    # 2) 读取 BPG 文件到内存
    with open(temp_bpg, 'rb') as f:
        bpg_data = f.read()  # bytes 对象
    header_bytes = bpg_data[:32]  # 保护前 32 字节
    hevc_bytes = bpg_data[32:]    # 仅对 HEVC 码流做 LDPC
    # 转成 torch.uint8
    bpg_data_tensor = torch.tensor(list(hevc_bytes), dtype=torch.uint8)

    # 3) 把BPG字节流转换成比特 (每字节8bit)
    bpg_bits = dec2bin(bpg_data_tensor, 8)  # shape: [num_bytes*8]

    # 4) 建立 sionna LDPC + 信道模型
    #    - 假设 bpg_bits_tf shape: [1, K], K = bpg_bits_tf.shape[1]
    bpg_bits_tf = tf.convert_to_tensor(bpg_bits.unsqueeze(0).numpy(), dtype=tf.float32)
    blocks  = split_into_blocks(bpg_bits_tf, 8448)
    c, ks, ns = ldpc_encode_blocks(blocks, ldpc_rate)
    x = mapper(c)              # [1, N/4]
    no = sionna.utils.ebnodb2no(ebno_db, 4, ldpc_rate)

    if channel_type == 'rayleigh':
        y, h = channel([tf.reshape(x, (x.shape[1],x.shape[0])), no])
        s = tf.cast(no*tf.eye(1, 1), y.dtype)
        x_hat, no_eff = lmmse_equalizer(y, h, s)
        llr = demapper([tf.reshape(x_hat,(x_hat.shape[1],x_hat.shape[0])), no])
    if channel_type == 'awgn':
        y = channel([x, no])
        llr = demapper([y, no])
    b_hat = ldpc_decode_blocks([llr], ldpc_rate, ks, ns)
    # 计算BER（可选）
    ber_coded = sionna.utils.metrics.compute_ber(bpg_bits_tf, b_hat)
    # print(f"EbNo={ebno_db}dB, BPG+LDPC BER={ber_coded:.4e}")

    # 6) 重组BPG字节流并写回文件
    b_hat = torch.from_numpy(b_hat.numpy()).to(torch.uint8).squeeze(0)    # shape: [K]
    b_hat_bytes = bin2dec(b_hat.view(-1, 8), 8)                           # [num_bytes]
    # b_hat_bytes 还是 torch.uint8，转成 list 然后变成真正的bytearray
    final_bpg_data = header_bytes + bytes(b_hat_bytes.tolist())
    with open(temp_bpg_received, 'wb') as f:
        f.write(bytearray(final_bpg_data))

    # 7) 调用 bpgdec 解码 -> temp_recon.png
    bpg_decompress(
        input_bpg=temp_bpg_received,
        output_file=temp_recon_png
    )

    # 8) 读回 temp_recon.png 做 PSNR
    recon_img = Image.open(temp_recon_png).convert('RGB')
    recon_tensor = transforms.ToTensor()(recon_img) * 255  # 回到[0,255]范围

    psnr_val = calculate_psnr(torch.squeeze(raw_img_tensor, 0).float(), recon_tensor.float())
    ms_ssim = 1 - CalcuSSIM(raw_img_tensor/255., (recon_tensor.unsqueeze(0)/255.).clamp(0., 1.)).mean().item()
    # 计算压缩率 (可自定义)
    #   raw_img_tensor.nbytes = C*H*W (uint8)
    #   bpg_data_tensor.nbytes = BPG大小(字节)
    #   有LDPC -> 实际传输长度 ~ bpg_data_tensor.nbytes / ldpc_rate
    cbr = (bpg_data_tensor.nelement()) / (raw_img_tensor.nelement()) / ldpc_rate
    # 清理临时文件
    # （如果想调试保留文件，可注释掉）
    # for fn in [temp_input_png, temp_bpg, temp_bpg_received, temp_recon_png]:
    #     if os.path.exists(fn):
    #         os.remove(fn)

    return psnr_val, cbr, ms_ssim

def bpg_cifar_after_channel():
    for i in range(len(multiple_snr)):
        psnrs, cbrs, ssim = [AverageMeter() for _ in range(3)]
        for j, data in enumerate(tqdm(kodak_loader)):
            if j != 6:
                continue
            img = data
            psnr, cbr, ms_ssim = test_bpg_ldpc_channel_single_image(
                                    img,
                                    bpg_quality=30,          #  35 30 20 15 10
                                    ldpc_rate=0.5,
                                    ebno_db=multiple_snr[i],
                                    channel_type=channel_type)
            cbrs.update(cbr)
            psnrs.update(psnr)
            ssim.update(ms_ssim)

        print("snr:", multiple_snr[i])
        print("cbrs:", cbrs.avg)
        print("psnrs:", psnrs.avg)
        print("ssim:", ssim.avg)

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
    # bpg_cifar_after_channel()
