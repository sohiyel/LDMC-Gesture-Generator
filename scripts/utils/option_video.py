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
    parser.add("--body_vqvae_path", type=str, default=None)
    parser.add("--hand_vqvae_path", type=str, default=None)
    parser.add("--body_diffusion_path", type=str, default=None)
    parser.add("--hand_diffusion_path", type=str, default=None)
    parser.add("--video_sample_path", type=str, default=None)
    parser.add("--mean_vectors_path", type=str, default=None)
    
    ## output directory 
    parser.add('--out_dir', type=str, default='output_vqfinal/', help='output directory')
    parser.add('--exp_name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add('--clip_duration', default=[50, 90], nargs="+", type=int, help="Clip duration")
    parser.add('--loader_workers', default=8, type=int, help="Number of loader workers")
    
    parser.add('--nb_videos', default=20, type=int, help='nb of videos')
    parser.add('--model_type', default='diffsuion', type=str, help='model to evaluate')

    #pose
    parser.add("--in_pose_dim", type=int)
    parser.add("--out_pose_dim", type=int)
    parser.add("--motion_resampling_framerate", type=int, default=24)
    parser.add("--n_poses", type=int, default=50)
    parser.add("--n_pre_poses", type=int, default=5)
    parser.add("--subdivision_stride", type=int, default=5)

    parser.add("--subject", type=str, default='both', help="Subject to train")
    
    return parser.parse_args()