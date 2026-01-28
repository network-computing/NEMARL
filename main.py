import os
from argument import *


if __name__ == '__main__':
    args = get_args()
    from train import run
    args.dqn_save_dir = os.path.join(args.dqn_save_dir, "basic" if not args.is_irl else "irl")
    out_graphs_dir = os.path.join(args.out_graph_dir, "basic" if not args.is_irl else "irl")
    if args.is_train:
        args.out_graph_dir = os.path.join(out_graphs_dir, "train")
        run(args)
    else:
        assert args.mode in ['static_sw', 'static_degree', 'dynamic_luq', 'dynamic_degree', 'dynamic_lcc_eff']
        args.total_episodes = 1
        if 'static' in args.mode:
            if args.mode == 'static_sw':
                from model.fitter import SwFitter
                print('[INFO] Begin the static sw tasks!')
                args.out_graph_dir = os.path.join(out_graphs_dir, "static_sw")
                sw_fitter = SwFitter(output_file_name="sw_summary_basic.csv" if not args.is_irl else "sw_summary_irl.csv")
                sw_fitter.fit(args, run)
            elif args.mode == 'static_degree':
                from model.fitter import DegreeFitter
                print('[INFO] Begin the static degree tasks!')
                args.out_graph_dir = os.path.join(out_graphs_dir, "static_degree")
                degree_fitter = DegreeFitter(output_file_name="degree_summary_basic.csv" if not args.is_irl else "degree_summary_irl.csv")
                degree_fitter.fit(args, run)
            else:
                assert False, "[Error] Unknown static task!"
        else:
            if args.mode == 'dynamic_luq':
                from model.fitter import DynamicLuqFitter
                print('[INFO] Begin the dynamic link update quantity tasks')
                args.out_graph_dir = os.path.join(out_graphs_dir, "dynamic_luq")
                dynamic_luq_fitter = DynamicLuqFitter(output_file_name="dynamic_luq_summary_basic.csv" if not args.is_irl else "dynamic_luq_summary_irl.csv")
                dynamic_luq_fitter.fit(args, run)
            elif args.mode == 'dynamic_degree':
                from model.fitter import DynamicDegreeFitter
                print('[INFO] Begin the dynamic degree tasks')
                args.out_graph_dir = os.path.join(out_graphs_dir, "dynamic_degree")
                dynamic_degree_fitter = DynamicDegreeFitter(output_file_name="dynamic_degree_summary_basic.csv" if not args.is_irl else "dynamic_degree_summary_irl.csv")
                dynamic_degree_fitter.fit(args, run)
            elif args.mode == 'dynamic_lcc_eff':
                from model.fitter import DynamicLccEffFitter
                print('[INFO] Begin the dynamic LCC/EFF tasks')
                args.out_graph_dir = os.path.join(out_graphs_dir, "dynamic_lcc_eff")
                dynamic_lcc_eff_fitter = DynamicLccEffFitter(output_file_name="dynamic_lcc_eff_summary_basic.csv" if not args.is_irl else "dynamic_lcc_eff_summary_irl.csv")
                dynamic_lcc_eff_fitter.fit(args, run)
            else:
                assert False, "[Error] Unknown dynamic task!"

