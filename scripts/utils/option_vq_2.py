import configargparse

def get_args_parser():
    parser = configargparse.ArgParser()

    parser.add('-c', '--config', required=True, is_config_file=True, help='Config file path')
    parser.add('--gpu_num', type=int, default="cuda:0", help='gpu number')
    parser.add('--num_gpu', type=int, default="1", help='number of gpus')
    ## dataloader  
    parser.add('--data_path', type=str, default='kit', help='data directory')
    parser.add('--dataset', type=str, default='kit', help='dataset directory')
    parser.add('--dataset_name', type=str, default='kit', help='dataset name')
    parser.add('--batch_size', default=128, type=int, help='batch size')
    parser.add('--window-size', type=int, default=64, help='training motion length')
    parser.add("--train_data_path", type=str, default='train', help='train directory')
    parser.add("--val_data_path", type=str, default='val', help='val directory')
    parser.add("--test_data_path", type=str, default='test', help='test directory')
    
    parser.add("--embed_evaluator_path", type=str, default=None)
    parser.add("--video_sample_path", type=str, default=None)
    parser.add("--mean_vectors_path", type=str, default=None)

    parser.add("--subject", type=str, default='both')

    ## optimization
    parser.add('--epochs', default=10, type=int, help='number of total iterations to run')
    parser.add('--lr', default=2e-4, type=float, help='max learning rate body')
    parser.add('--lr_scheduler', default=[50000, 400000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add("--contrastive", type=float, default=0.02, help="hyper-parameter for the contrastive loss")
    parser.add('--recons-loss', type=str, default='l2', help='reconstruction loss')
    
    ## vqvae arch
    parser.add("--code_dim", type=int, default=512, help="embedding dimension")
    parser.add("--nb_code", type=int, default=512, help="nb of embedding")
    parser.add("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add("--down_t", type=int, default=2, help="downsampling rate")
    parser.add("--stride", type=int, default=2, help="stride size")
    parser.add("--width", type=int, default=512, help="width of the network")
    parser.add("--depth", type=int, default=3, help="depth of the network")
    parser.add("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    
    ## quantizer
    parser.add("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## resume
    parser.add("--resume_path", type=str, default=None, help='resume pth for VQ')
    
    
    ## output directory 
    parser.add('--out_dir', type=str, default='output_vqfinal/', help='output directory')
    parser.add('--exp_name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add('--eval_iter', default=1000, type=int, help='evaluation frequency')
    parser.add('--seed', default=123, type=int, help='seed for initializing training.')
    parser.add('--save_top_k', default=1, type=int, help='Save top k number for checkpointing')
    parser.add('--monitor', default="FGD", type=str, help='Monitor metric for checkpointing')
    parser.add('--mode', default="min", type=str, help='Monitor metric mode for chechpointing')

    #pose
    parser.add("--in_pose_dim", type=int)
    parser.add("--out_pose_dim", type=int)
    parser.add("--motion_resampling_framerate", type=int, default=24)
    parser.add("--n_poses", type=int, default=50)
    parser.add("--n_pre_poses", type=int, default=5)
    parser.add("--subdivision_stride", type=int, default=5)
    parser.add("--loader_workers", type=int, default=0)

    parser.add('--temperature', type=float, default=0.5, help='contrastive temperature')
    
    return parser.parse_args()