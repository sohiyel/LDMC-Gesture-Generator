import configargparse

def get_args_parser():
    parser = configargparse.ArgParser()

    parser.add('-c', '--config', required=True, is_config_file=True, help='Config file path')
    parser.add('--gpu_num', type=str, default="cuda:0", help='gpu number')
    parser.add('--num_gpu', type=int, default="1", help='gpu number')
    ## dataloader  
    parser.add('--data_path', type=str, default='kit', help='data directory')
    parser.add('--dataset', type=str, default='kit', help='dataset directory')
    parser.add('--dataset_name', type=str, default='kit', help='dataset name')
    parser.add('--batch_size', default=128, type=int, help='batch size')
    parser.add("--train_data_path", type=str, default='train', help='train directory')
    parser.add("--val_data_path", type=str, default='val', help='val directory')
    parser.add("--test_data_path", type=str, default='test', help='test directory')
    
    parser.add("--embed_evaluator_path", type=str, default=None)
    parser.add("--vqvae_path", type=str, default=None)
    parser.add("--resume_path", type=str, default=None)

    ## optimization
    parser.add('--epochs', default=10, type=int, help='number of total iterations to run')
    parser.add('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add('--lr_scheduler', default=[50000, 400000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add('--recons_loss', type=str, default='l2', help='reconstruction loss')
    parser.add('--weight_decay', default=0.0, type=float, help='weight decay')

    ## output directory 
    parser.add('--out_dir', type=str, default='output_vqfinal/', help='output directory')
    parser.add('--exp_name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    parser.add('--monitor', type=str, default='FGD', help='Monitor metric for validation')
    parser.add('--save_top_k', type=int, default=1, help='Save top k models')
    parser.add('--mode', type=str, default='min', help='Mode for checkpoint callback')
    ## other
    parser.add('--eval_iter', default=1000, type=int, help='evaluation frequency')
    parser.add('--seed', default=123, type=int, help='seed for initializing training.')
    
    ## diffusion
    parser.add("--nb_steps", type=int, default=10, help="nb of steps")
    parser.add("--beta_schedule", type=str, default='linear', help="beta")
    parser.add("--subject", type=str, default='both', help="Subject to train")

    #pose
    parser.add("--in_pose_dim", type=int)
    parser.add("--out_pose_dim", type=int)
    parser.add("--motion_resampling_framerate", type=int, default=24)
    parser.add("--n_poses", type=int, default=50)
    parser.add("--n_pre_poses", type=int, default=5)
    parser.add("--subdivision_stride", type=int, default=5)
    parser.add("--loader_workers", type=int, default=0)
    
    return parser.parse_args()