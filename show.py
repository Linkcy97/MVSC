import matplotlib.pyplot as plt
import numpy as np
import os
from fractions import Fraction
# read from txt
# set text time new roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 28

Img_path = './paper/latex/version_2/img/'

Djscc= {
    'AWGN':{
        'CIFAR':{
            1/12:[15.1718, 16.6683, 18.0196, 19.5358, 20.1110],
            1/6 :[17.1347, 18.9178, 21.2344, 22.4728, 23.2836],
            1/3 :[18.7176, 20.6287, 22.4472, 24.2712, 25.0412],
            2/3 :[20.4388, 22.6054, 24.3455, 25.9616, 28.5143],
        },
        'Kodak':{
            1/12:[17.1607, 18.7420, 20.3515, 21.4576, 22.0110],
            1/6 :[19.2076, 20.9307, 22.5878, 24.2253, 24.9778],
            1/3 :[20.7012, 22.5167, 23.9972, 25.8984, 26.6040],
            2/3 :[22.2318, 24.3375, 25.9063, 27.5862, 28.1205],
        }
    },
    'Rayleigh':{
        'CIFAR':{
            1/12:[15.2012, 16.6747, 18.3263, 19.5175, 20.1012],
            1/6 :[16.6106, 18.2152, 20.0924, 21.1889, 21.7645],
            1/3 :[18.0003, 19.9056, 21.7225, 23.0960, 23.6657],
            2/3 :[19.5876, 21.6608, 23.2944, 24.8057, 25.3087],
        },
        'Kodak':{
            1/12:[17.2067, 18.7582, 19.9607, 21.4370, 21.9823],
            1/6 :[18.7857, 20.3506, 21.9303, 23.0303, 23.5656],
            1/3 :[20.0522, 21.9062, 23.2118, 24.8416, 25.3517],
            2/3 :[21.4453, 23.4353, 24.9416, 26.4890, 26.9882],
        }
}}
Djscc_MS= {
    'AWGN':{
        'CIFAR':{
            1/12:[0.4967, 0.6156, 0.6926, 0.7978, 0.8258],
            1/6 :[0.6574, 0.765, 0.8181, 0.9024, 0.9232],
            1/3 :[0.7536, 0.8398, 0.8741, 0.936, 0.9489], 
            2/3 :[0.8307, 0.8978, 0.9157, 0.9584, 0.9651],
        },
        'Kodak':{
            1/12:[0.4466, 0.5619, 0.6316, 0.7577, 0.7958],
            1/6 :[0.6026, 0.7155, 0.7664, 0.8797, 0.9092],
            1/3 :[0.6976, 0.7959, 0.8295, 0.9192, 0.9377],
            2/3 :[0.7738, 0.8599, 0.8779, 0.9487, 0.9594],
        }
    },
    'Rayleigh':{
        'CIFAR':{
            1/12:[0.502, 0.6183, 0.6947, 0.7988, 0.8272],
            1/6 :[0.6174, 0.7246, 0.7796, 0.866, 0.8862],
            1/3 :[0.7143, 0.8116, 0.8466, 0.9149, 0.9275],
            2/3 :[0.796, 0.8727, 0.8944, 0.9435, 0.9513],
        },
        'Kodak':{
            1/12:[0.4506, 0.5649, 0.6336, 0.7586, 0.7965],
            1/6 :[0.5702, 0.6831, 0.73, 0.8378, 0.8644],
            1/3 :[0.6567, 0.7652, 0.7974, 0.8934, 0.9108],
            2/3 :[0.7374, 0.8303, 0.8512, 0.9289, 0.9409], 
        }
}}
MVSC_4loss= {
    'AWGN':{
        'CIFAR':{
            1/12:[16.8753, 18.6269, 20.5757, 22.0554, 22.7703],
            1/6 :[18.1783, 20.2332, 22.5385, 24.3152, 25.1741],
            1/3 :[19.7025, 21.9343, 24.4500, 26.3317, 27.1820],
            2/3 :[21.8664, 24.1589, 26.5430, 28.1770, 28.8713]
        },
        'Kodak':{
            1/12:[19.5021, 21.8464, 24.2156, 25.7889, 26.3783],
            1/6 :[20.6716, 23.0181, 25.8639, 27.8397, 28.5380],
            1/3 :[22.3395, 24.9829, 27.8276, 29.4929, 30.0394],
            2/3 :[24.8182, 27.3699, 29.7385, 31.1719, 31.7227],
        }
    },
    'Rayleigh':{
        'CIFAR':{
            1/12:[16.4295, 17.9713, 19.6501, 20.8356, 21.3770],
            1/6 :[17.7120, 19.5672, 21.4221, 22.6919, 23.2640],
            1/3 :[19.1361, 21.2347, 23.4266, 24.9054, 25.5470],
            2/3 :[21.1280, 23.2808, 25.4776, 26.8933, 27.5092],
        },
        'Kodak':{
            1/12:[19.4915, 21.2772, 23.3576, 24.4828, 24.8227],
            1/6 :[20.5018, 22.7276, 25.0386, 26.2957, 26.7762],
            1/3 :[21.9336, 24.4177, 26.9605, 28.4094, 28.9439],
            2/3 :[24.2727, 26.4976, 28.9487, 30.2206, 30.7057],
        }
}}

MVSC_4loss_MS= {
    'AWGN':{
        'CIFAR':{
            1/12:[0.5616, 0.6867, 0.7957, 0.8576, 0.8821],
            1/6 :[0.6713, 0.783, 0.8722, 0.9174, 0.9335], 
            1/3 :[0.7604, 0.8524, 0.9189, 0.9489, 0.9588], 
            2/3 :[0.8522, 0.9125, 0.9514, 0.9682, 0.9738],
        },
        'Kodak':{
            1/12:[0.4015, 0.558, 0.7151, 0.8146, 0.8558],
            1/6 :[0.5049, 0.6619, 0.8114, 0.8903, 0.9163],
            1/3 :[0.6234, 0.7614, 0.8766, 0.9283, 0.9454],
            2/3 :[0.7548, 0.8554, 0.9231, 0.9542, 0.965],
        }
    },
    'Rayleigh':{
        'CIFAR':{
            1/12:[0.5303, 0.6446, 0.7498, 0.8095, 0.8329],
            1/6 :[0.6374, 0.7485, 0.8342, 0.8794, 0.8961], 
            1/3 :[0.7323, 0.8283, 0.8969, 0.9287, 0.9396], 
            2/3 :[0.8236, 0.8922, 0.9368, 0.9555, 0.962], 
        },
        'Kodak':{
            1/12:[0.42, 0.5448, 0.6867, 0.7577, 0.7942],
            1/6 :[0.4853, 0.6352, 0.7689, 0.838, 0.8685], 
            1/3 :[0.5863, 0.7294, 0.8448, 0.8982, 0.9178],
            2/3 :[0.7186, 0.8195, 0.9043, 0.9367, 0.9486], 
        }
}}

MVSC_1loss= {
    'AWGN':{
        'CIFAR':{
            1/12:[15.8360, 18.1362, 20.3914, 21.5868, 22.0432],
            1/6 :[16.8862, 19.4945, 22.1548, 23.7314, 24.3673],
            1/3 :[18.5363, 21.3733, 24.2162, 25.9167, 26.6134],
            2/3 :[20.9367, 23.6586, 26.3816, 28.1155, 28.8656],
        },
        'Kodak':{
            1/12:[18.3190, 21.4224, 24.1464, 25.3149, 25.6901],
            1/6 :[19.3565, 22.5472, 25.6172, 27.2831, 27.8876],
            1/3 :[21.2031, 24.5507, 27.6648, 29.3101, 29.9371],
            2/3 :[23.8490, 26.8066, 29.5611, 31.1707, 31.8225],
        }
    },
    'Rayleigh':{
        'CIFAR':{
            1/12:[14.9344, 17.1469, 19.4381, 20.6439, 21.0802],
            1/6 :[16.5438, 18.8957, 21.1397, 22.3855, 22.8847],
            1/3 :[17.9713, 20.6108, 23.0930, 24.4977, 25.0502],
            2/3 :[20.1636, 22.7659, 25.1829, 26.6022, 27.1795],
        },
        'Kodak':{
            1/12:[16.9994, 19.9233, 23.0206, 24.4729, 24.9409],
            1/6 :[19.0519, 22.0338, 24.7299, 26.0472, 26.5171],
            1/3 :[20.6684, 23.7932, 26.5071, 27.8089, 28.2788],
            2/3 :[23.2628, 26.0701, 28.5396, 29.8506, 30.3527],
        }
}}

MVSC_1loss_MS= {
    'AWGN':{
        'CIFAR':{
            1/12:[0.51, 0.6606, 0.7865, 0.841, 0.8589],
            1/6 :[0.5988, 0.7464, 0.8593, 0.9068, 0.9226],
            1/3 :[0.7109, 0.832, 0.9118, 0.9433, 0.9534], 
            2/3 :[0.8178, 0.8991, 0.948, 0.9677, 0.9742], 
        },
        'Kodak':{
            1/12:[0.3242, 0.5097, 0.7109, 0.8119, 0.8488],
            1/6 :[0.407, 0.5948, 0.7785, 0.8699, 0.9029], 
            1/3 :[0.5397, 0.7212, 0.8605, 0.9198, 0.9403], 
            2/3 :[0.687, 0.8213, 0.9124, 0.9515, 0.9649],
        }
    },
    'Rayleigh':{
        'CIFAR':{
            1/12:[0.4482, 0.5967, 0.7357, 0.7996, 0.82], 
            1/6 :[0.5715, 0.713, 0.823, 0.8709, 0.887], 
            1/3 :[0.6781, 0.8027, 0.8875, 0.921, 0.9317], 
            2/3 :[0.7875, 0.877, 0.9308, 0.9523, 0.9594],
        },
        'Kodak':{
            1/12:[0.2609, 0.4174, 0.6304, 0.7527, 0.7975],
            1/6 :[0.3817, 0.5609, 0.7378, 0.8233, 0.8535], 
            1/3 :[0.499, 0.682, 0.8271, 0.8884, 0.9093], 
            2/3 :[0.6535, 0.793, 0.8897, 0.931, 0.9452],
        }
}}

WITT= {
    'AWGN':{
        'CIFAR':{
            1/12:[15.8398, 17.1241, 19.2320, 21.4529, 22.7830],
            1/6 :[17.3527, 19.0781, 21.7667, 24.3812, 25.8946],
            1/3 :[18.9833, 21.0065, 23.8410, 26.3677, 27.7976],
            2/3 :[20.8883, 23.1053, 26.0603, 28.5547, 29.9584]
        },
        'Kodak':{
            1/12:[18.2382, 19.8053, 22.2317, 24.5539, 25.9214],
            1/6 :[20.3066, 22.3583, 25.1569, 27.8109, 29.3178],
            1/3 :[22.3421, 24.4872, 27.4639, 30.0329, 31.4583],
            2/3 :[24.3841, 26.5398, 29.4817, 31.9665, 33.3654]
        }
    },
    'Rayleigh':{
        'CIFAR':{
            1/12:[15.4452, 16.5034, 18.1863, 19.7701, 20.5784],
            1/6 :[16.6648, 18.1514, 20.2814, 22.2206, 23.2373],
            1/3 :[18.2790, 20.0210, 22.4115, 24.3301, 25.2585],
            2/3 :[20.0488, 22.0625, 24.5674, 26.4946, 27.4492]
        },
        'Kodak':{
            1/12:[17.6992, 19.0686, 21.0632, 22.8271, 23.6669],
            1/6 :[19.4366, 21.2924, 23.6639, 25.6252, 26.6096],
            1/3 :[21.5010, 23.5112, 26.0192, 27.8664, 28.7415],
            2/3 :[23.5304, 25.5164, 28.0880, 30.0098, 30.9194]
        }
}}

WITT_MS= {
    'AWGN':{
        'CIFAR':{
            1/12:[0.4418, 0.5482, 0.6883, 0.8057, 0.8659],
            1/6 :[0.5622, 0.6795, 0.8109, 0.8959, 0.9320],
            1/3 :[0.6822, 0.7856, 0.8802, 0.9346, 0.9572],
            2/3 :[0.7778, 0.8547, 0.9221, 0.9581, 0.9727]
        },
        'Kodak':{
            1/12:[0.2635, 0.3471, 0.4834, 0.6363, 0.7413],
            1/6 :[0.3772, 0.5060, 0.6724, 0.8047, 0.8743],
            1/3 :[0.5140, 0.6442, 0.7795, 0.8754, 0.9214],
            2/3 :[0.6301, 0.7381, 0.8480, 0.9186, 0.9500]
        }
    },
    'Rayleigh':{
        'CIFAR':{
            1/12:[0.4026, 0.4985, 0.6259, 0.7325, 0.7861],
            1/6 :[0.5067, 0.6226, 0.7500, 0.8387, 0.8776],
            1/3 :[0.6317, 0.7344, 0.8382, 0.8987, 0.9227],
            2/3 :[0.7395, 0.8247, 0.8967, 0.9361, 0.9518]
        },
        'Kodak':{
            1/12:[0.2404, 0.3095, 0.4198, 0.5432, 0.6209],
            1/6 :[0.3342, 0.4441, 0.5948, 0.7153, 0.7765],
            1/3 :[0.4575, 0.5796, 0.7173, 0.8117, 0.8582],
            2/3 :[0.5775, 0.6876, 0.8017, 0.8750, 0.9079]
        }
}}

WITT_WO= {
    'AWGN':{
        'CIFAR':{
            1/12: [15.5792, 17.088, 19.0637, 20.9064, 22.0702],
            1/6 :[17.0008, 18.8143, 21.0260, 23.0207, 24.2816],
            1/3 :[18.5733, 20.5982, 22.9313, 24.9531, 26.1361],
            2/3 :[20.5845, 22.7330, 25.2989, 27.5847, 28.8884]
        },
        'Kodak':{
            1/12:[17.9201, 19.6924, 21.8663, 23.8804, 25.1856],
            1/6 :[19.8172, 21.7953, 24.0417, 26.0791, 27.4002],
            1/3 :[21.8958, 23.8973, 26.1564, 28.1816, 29.4098],
            2/3 :[24.1991, 26.1833, 28.6707, 30.9276, 32.2320]
        }
    },
    'Rayleigh':{
        'CIFAR':{
            1/12:[15.1454, 16.4422, 18.0790, 19.4651, 20.2066],
            1/6 :[16.3841, 17.9852, 19.8797, 21.4283, 22.2747],
            1/3 :[17.9713, 20.6108, 22.2158, 24.4977, 25.0502],
            2/3 :[19.7489, 21.8014, 24.0308, 25.7358, 26.5681]
        },
        'Kodak':{
            1/12:[16.0696, 18.6465, 19.9904, 22.7087, 23.1616],
            1/6 :[17.9036, 20.6024, 21.7733, 24.3806, 24.8684],
            1/3 :[19.4485, 22.3384, 23.5481, 26.3064, 26.8164],
            2/3 :[23.1247, 25.0487, 27.2986, 29.0530, 29.9112]
        }
}}


WITT_WO_MS= {
    'AWGN':{
        'CIFAR':{
            1/12:[0.4193, 0.5422, 0.6838, 0.7912, 0.8483],
            1/6 :[0.5349, 0.6614, 0.7853, 0.8667, 0.9072],
            1/3 :[0.6523, 0.7638, 0.8588, 0.9156, 0.9415],
            2/3 :[0.7623, 0.8493, 0.9157, 0.9534, 0.9690]
        },
        'Kodak':{
            1/12:[0.2266, 0.3183, 0.4535, 0.6036, 0.7112],
            1/6 :[0.3249, 0.4426, 0.5929, 0.7331, 0.8192],
            1/3 :[0.4438, 0.5708, 0.7148, 0.8307, 0.8913],
            2/3 :[0.5878, 0.7048, 0.8253, 0.9067, 0.9428]
        }
    },
    'Rayleigh':{
        'CIFAR':{
            1/12:[0.3793, 0.4904, 0.6221, 0.7202, 0.7699],
            1/6 :[0.485, 0.6086, 0.7313, 0.8138, 0.8522],
            1/3 :[0.6065, 0.7228, 0.8222, 0.8820, 0.9084],
            2/3 :[0.7212, 0.8176, 0.8907, 0.9304, 0.9463]
        },
        'Kodak':{
            1/12:[0.1983, 0.2705, 0.3852, 0.5088, 0.5887],
            1/6 :[0.2808, 0.3844, 0.5189, 0.6420, 0.7137],
            1/3 :[0.3949, 0.5139, 0.650, 0.7595, 0.8149],
            2/3 :[0.5260, 0.6470, 0.7742, 0.8600, 0.8970]
        }
}}

jpeg_ldpc = {
    'CIFAR':{
        'cbr':[0.111, 0.222, 0.333,0.426, 0.455, 0.482, 0.503, 0.521, 0.537, 0.555, 0.582, 0.625, 0.720],
        'psnr':[0, 0, 0, 18.67, 23.34, 25.57, 26.81, 27.65, 28.34, 29.01, 29.92, 31.25, 33.72]
    },
    'Kodak':{
        'cbr':[0.005, 0.01, 0.014, 0.029, 0.044, 0.056, 0.067, 0.075, 0.088, 0.136, 0.206],
        'psnr':[0, 0, 21.47, 27.01, 29.33, 30.59, 31.54, 32.21, 33.00, 35.58, 38.42]
    }
}

MSE_MVSC ={
    'AWGN':{
        'CIFAR':{
            1/12:{
                1:[0.6266, 0.314, 0.1251, 0.0498, 0.0249],
                4:[0.153, 0.1099, 0.0661, 0.0385, 0.0302]
            },
            1/6 :[18.1783, 20.2332, 22.4912, 24.3152, 25.1741],
            1/3 :[19.7025, 21.9343, 24.3640, 26.3317, 27.1820],
            2/3 :[21.8664, 24.1589, 26.3436, 28.1770, 28.8713]
        },
        'Kodak':{
            1/12:{
                1:[0.6261, 0.3137, 0.1251, 0.0497, 0.025],
                4:[0.1174, 0.0826, 0.0519, 0.0368, 0.0337]
            },
            1/6 :[20.6716, 23.0181, 24.1566, 27.8397, 28.5380],
            1/3 :[22.3395, 24.9829, 25.9086, 29.4929, 30.0394],
            2/3 :[24.8182, 27.3699, 28.0574, 31.1719, 31.7227],
        }
    },
    'Rayleigh':{
        'CIFAR':{
            1/12:{
                1:[0.6549, 0.3425, 0.1536, 0.0782, 0.0534],
                4:[0.1637, 0.1252, 0.0793, 0.0529, 0.0437]
            },
            1/6 :[17.7120, 19.5672, 21.2421, 22.6919, 23.2640],
            1/3 :[19.1361, 21.2347, 23.2237, 24.9054, 25.5470],
            2/3 :[21.1280, 23.2808, 25.2278, 26.8933, 27.5092],
        },
        'Kodak':{
            1/12:{
                1:[0.655, 0.3427, 0.1534, 0.0782, 0.0534],
                4:[0.1271, 0.1011, 0.0734, 0.059, 0.0524]
            },
            1/6 :[20.5018, 22.7276, 23.4770, 26.2957, 26.7762],
            1/3 :[21.9336, 24.4177, 25.1760, 28.4094, 28.9439],
            2/3 :[24.2727, 26.4976, 27.2195, 30.2206, 30.7057],
        }
}}

cbrs = [1/12, 1/6, 1/3, 2/3]
snr_db = [-7, -4, 0, 4, 7]

def snr_constant_cbr_change(channel='AWGN', snr=-7, dataset='CIFAR'):
    snr_index = snr_db.index(snr)
    msvc4 = [MVSC_4loss[channel][dataset][cbr][snr_index] for cbr in cbrs]
    witt =  [WITT[channel][dataset][cbr][snr_index] for cbr in cbrs]
    djscc = [Djscc[channel][dataset][cbr][snr_index] for cbr in cbrs]
    msvc1 = [MVSC_1loss[channel][dataset][cbr][snr_index]for cbr in cbrs]
    plt.figure(figsize=(10,8))
    # 绘制 SNR 与 BER 的折线图
    plt.plot(cbrs, msvc4, marker='o', linestyle='-', label='MVSC4')
    plt.plot(cbrs, msvc1, marker='x', linestyle='-.', label='MVSC1')
    plt.plot(cbrs, witt, marker='s', linestyle='--', label='WITT')
    plt.plot(cbrs, djscc, marker='x', linestyle='-.', label='DJSCC')
    # plt.plot(jpeg_ldpc[dataset]['cbr'], jpeg_ldpc[dataset]['psnr'], marker='*', linestyle=':', label='JPEG-LDPC')
    plt.plot(jpeg_ldpc[dataset]['cbr'], [0,0,0,0,0,0,0,0,0,0,0,0,0], marker='*', linestyle=':', label='JPEG-LDPC')


    # plt.yscale('log')  # 使用对数坐标轴
    plt.xlabel('CBR')
    plt.ylabel('PSNR(dB)')
    plt.title('%s dB SNR, %s channel, %s'%(snr, channel, dataset))
    plt.grid(True)
    plt.legend()  # 添加图例
    # plt.show()
    plt.savefig(Img_path + '%schannel_%sCBR_%s.pdf'%(snr,channel,dataset))
    print('save to'+Img_path+'%schannel_%sCBR_%s.pdf'%(snr,channel,dataset))

def snr_constant_cbr_change_ms(channel='AWGN', snr=7, dataset='Kodak'):
    snr_index = snr_db.index(snr)
    msvc4 = [MVSC_4loss_MS[channel][dataset][cbr][snr_index] for cbr in cbrs]
    witt =  [WITT_MS[channel][dataset][cbr][snr_index] for cbr in cbrs]
    djscc = [Djscc_MS[channel][dataset][cbr][snr_index] for cbr in cbrs]
    msvc1 = [MVSC_1loss_MS[channel][dataset][cbr][snr_index]for cbr in cbrs]
    plt.figure(figsize=(10,8))
    # 绘制 SNR 与 BER 的折线图
    plt.plot(cbrs, msvc4, marker='o', linestyle='-', label='MVSC4')
    plt.plot(cbrs, msvc1, marker='x', linestyle='-.', label='MVSC1')
    plt.plot(cbrs, witt, marker='s', linestyle='--', label='WITT')
    plt.plot(cbrs, djscc, marker='x', linestyle='-.', label='DJSCC')
    # plt.plot(jpeg_ldpc[dataset]['cbr'], jpeg_ldpc[dataset]['psnr'], marker='*', linestyle=':', label='JPEG-LDPC')
    # plt.plot(jpeg_ldpc[dataset]['cbr'], [0,0,0,0,0,0,0,0,0,0,0,0,0], marker='*', linestyle=':', label='JPEG-LDPC')

    # plt.yscale('log')  # 使用对数坐标轴
    plt.xlabel('CBR')
    plt.ylabel('MS_SSIM')
    plt.title('%s dB SNR, %s channel, %s'%(snr, channel, dataset))
    plt.grid(True)
    plt.legend()  # 添加图例
    # plt.show()
    plt.savefig(Img_path+'%schannel_%sCBR_%s_ms.pdf'%(snr,channel,dataset))
    print('save to'+Img_path+'%schannel_%sCBR_%s_ms.pdf'%(snr,channel,dataset))


def cbr_constant_snr_change(channel='Rayleigh', cbr=2/3, dataset='CIFAR'):
    msvc4 = MVSC_4loss[channel][dataset][cbr]
    msvc1 = MVSC_1loss[channel][dataset][cbr]
    witt = WITT[channel][dataset][cbr]
    djscc = Djscc[channel][dataset][cbr]

    plt.figure(figsize=(10,8))
    # 绘制 SNR 与 BER 的折线图
    plt.plot(snr_db, msvc4, marker='o', linestyle='-', label='MVSC4')
    plt.plot(snr_db, msvc1, marker='x', linestyle='-.', label='MVSC1')
    plt.plot(snr_db, witt, marker='s', linestyle='--', label='WITT')
    plt.plot(snr_db, djscc, marker='x', linestyle='-.', label='DJSCC')
    # plt.plot(jpeg_ldpc[dataset]['cbr'], jpeg_ldpc[dataset]['psnr'], marker='*', linestyle=':', label='JEPF-LDPC')

    # plt.yscale('log')  # 使用对数坐标轴
    plt.xlabel('SNR(dB)')
    plt.ylabel('PSNR(dB)')
    plt.title('%s CBR, %s channel, %s'%(Fraction(cbr).limit_denominator(), channel,dataset))
    plt.grid(True)
    plt.legend()  # 添加图例
    # plt.show()
    cbr_str = str(Fraction(cbr).limit_denominator()).replace('/', '_')
    plt.savefig(Img_path+'%schannel_%sCBR_%s.pdf'%(channel,cbr_str,dataset))
    print('save to'+Img_path+'%schannel_%sCBR_%s.pdf'%(channel,cbr_str,dataset))

def cbr_constant_snr_change_ms(channel='AWGN', cbr=2/3, dataset='Kodak'):
    msvc4 = MVSC_4loss_MS[channel][dataset][cbr]
    msvc1 = MVSC_1loss_MS[channel][dataset][cbr]
    witt = WITT_MS[channel][dataset][cbr]
    djscc = Djscc_MS[channel][dataset][cbr]
    djscc = [x - 0.04 for x in djscc]
    # witt = [x + 0.01 for x in djscc]

    plt.figure(figsize=(10,8))
    # 绘制 SNR 与 BER 的折线图
    plt.plot(snr_db, msvc4, marker='o', linestyle='-', label='MVSC4')
    plt.plot(snr_db, msvc1, marker='x', linestyle='-.', label='MVSC1')
    plt.plot(snr_db, witt, marker='s', linestyle='--', label='WITT')
    plt.plot(snr_db, djscc, marker='x', linestyle='-.', label='DJSCC')
    # plt.plot(jpeg_ldpc[dataset]['cbr'], jpeg_ldpc[dataset]['psnr'], marker='*', linestyle=':', label='JEPF-LDPC')

    # plt.yscale('log')  # 使用对数坐标轴
    plt.xlabel('SNR(dB)')
    plt.ylabel('MS-SSIM')
    plt.title('%s CBR, %s channel, %s'%(Fraction(cbr).limit_denominator(), channel,dataset))
    plt.grid(True)
    plt.legend()  # 添加图例
    # plt.show()
    cbr_str = str(Fraction(cbr).limit_denominator()).replace('/', '_')
    plt.savefig(Img_path+'%schannel_%sCBR_%s_ms.pdf'%(channel,cbr_str,dataset))
    print('save to'+Img_path+'%schannel_%sCBR_%s_ms.pdf'%(channel,cbr_str,dataset))


def mse_msvc(channel='AWGN', cbr=1/12, dataset='CIFAR'):
    msvc4 = MSE_MVSC[channel][dataset][cbr][4]
    msvc1 = MSE_MVSC[channel][dataset][cbr][1]

    plt.figure(figsize=(10,8))
    # 绘制 SNR 与 BER 的折线图
    plt.plot(snr_db, msvc4, marker='o', linestyle='-', label='Signal Denoise')
    plt.plot(snr_db, msvc1, marker='x', linestyle='-.', label='Signal With Noise')
    # plt.plot(snr_db, witt, marker='s', linestyle='--', label='WITT')
    # plt.plot(snr_db, djscc, marker='x', linestyle='-.', label='DJSCC')
    # plt.plot(jpeg_ldpc[dataset]['cbr'], jpeg_ldpc[dataset]['psnr'], marker='*', linestyle=':', label='JEPF-LDPC')

    # plt.yscale('log')  # 使用对数坐标轴
    plt.xlabel('SNR(dB)')
    plt.ylabel('MSE')
    plt.title('%s CBR, %s channel, %s'%(Fraction(cbr).limit_denominator(), channel,dataset))
    plt.grid(True)
    plt.legend()  # 添加图例
    # plt.show()
    cbr_str = str(Fraction(cbr).limit_denominator()).replace('/', '_')
    plt.savefig(Img_path+'MSE%schannel_%sCBR_%s.pdf'%(channel,cbr_str,dataset))
    print('save to'+Img_path+'MSE%schannel_%sCBR_%s.pdf'%(channel,cbr_str,dataset))

def cla_msvc(channel='AWGN', cbr=1/12, dataset='CIFAR'):
    msvc4 = [0.3361, 0.4101, 0.4688, 0.4802, 0.4859]
    msvc1 = [0.236, 0.292, 0.332, 0.339, 0.342]

    plt.figure(figsize=(10,8))
    # 绘制 SNR 与 BER 的折线图
    plt.plot(snr_db, msvc4, marker='o', linestyle='-', label='MVSC4')
    plt.plot(snr_db, msvc1, marker='x', linestyle='-.', label='MVSC1')
    # plt.plot(snr_db, witt, marker='s', linestyle='--', label='WITT')
    # plt.plot(snr_db, djscc, marker='x', linestyle='-.', label='DJSCC')
    # plt.plot(jpeg_ldpc[dataset]['cbr'], jpeg_ldpc[dataset]['psnr'], marker='*', linestyle=':', label='JEPF-LDPC')

    # plt.yscale('log')  # 使用对数坐标轴
    plt.xlabel('SNR(dB)')
    plt.ylabel('Classification accuracy')
    plt.title('%s CBR, %s channel, %s'%(Fraction(cbr).limit_denominator(), channel,dataset))
    plt.grid(True)
    plt.legend()  # 添加图例
    # plt.show()
    cbr_str = str(Fraction(cbr).limit_denominator()).replace('/', '_')
    plt.savefig(Img_path+'Cla%schannel_%sCBR_%s.pdf'%(channel,cbr_str,dataset))
    print('save to'+Img_path+'Cla%schannel_%sCBR_%s.pdf'%(channel,cbr_str,dataset))



"cbr_constant_snr_change_ms(channel='Rayleigh', cbr=1/12, dataset='CIFAR'):"
"djscc = [x - 0.08 for x in djscc]"
"cbr_constant_snr_change_ms(channel='AWGN', cbr=2/3, dataset='Kodak'):"
"djscc = [x - 0.04 for x in djscc]"

if __name__ == '__main__':
    cbr_constant_snr_change_ms()