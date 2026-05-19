## encode2.py :
新的嵌入三圈的方法
## watermark_decoder3.py：
vit解码
## watermark_trainer.py
数据集目录：


 parser.add_argument('--train_dir', type=str, default="/mnt/ylyu/COCO-train2017/", help='Training data directory')
parser.add_argument('--val_dir', type=str, default="/mnt/ylyu/COCO-val2017/", help='Validation data directory')
parser.add_argument('--test_dir', type=str, default="/mnt/ylyu/COCO-test2017/", help='Test data directory')
parser.add_argument('--output_dir', type=str, default='/home/ylu2024/workspace/fftmask/output_60', help='Output directory for models and logs')


训练设置：


parser.add_argument('--batch_size', type=int, default=40, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--block_size', type=int, default=512, help='Block size for watermark decoding')


gpu设备：


parser.add_argument('--device', type=str, default='2,3', help='Device to use for training')


水印嵌入相关设置:


parser.add_argument('--num_bits', type=int, default=60, help='Number of bits for watermark decoding')
parser.add_argument('--r', type=list, default=[5,9,13], help='Radius for watermark decoding')
parser.add_argument('--bitsf', type=list, default=[5,15,40], help='Bits for each radius')
parser.add_argument('--alpha_embed', type=float, default=1, help='Embedding strength')