from params import param
import os
def gen_train_data():
    args = param()()
    from data import mixSPAC, mixSPAT
    root_path = '../thermal_coherent/'
    target_nbar = 1.3
    if os.path.exists(root_path):
        os.system('rm -rf '+root_path)
    os.makedirs(root_path)

    args.quantum_efficiency = 1.0
    args.nbar = 2.0
    args.mixRatio = 1.0
    nbar_th_spac = 2.0
    nbar_th_spat = 2.0
    args.path = root_path

    for i in range(11):
        _ = False
        args.nbar = nbar_th_spac
        while not _:
            s = mixSPAC(args)
            s()
            s.estimate_nbar()
            _, obs_nbar = s.save(10, 200, target_nbar= target_nbar, key = str(i))
            # if abs(obs_nbar - target_nbar) < 0.05:
            #     s.save(10, 200, key = str(i))
            args.nbar -= (obs_nbar - target_nbar)
            print(f'[INFO] nbar = {args.nbar}, obs_nbar = {obs_nbar}, i = {i}')
        nbar_th_spac = args.nbar
        _ = False
        args.nbar = nbar_th_spat
        while not _:
            s = mixSPAT(args)
            s()
            s.estimate_nbar()
            _, obs_nbar = s.save(10, 200, target_nbar= target_nbar, key = str(i))
            # if abs(obs_nbar - target_nbar) < 0.05:
            #     s.save(10, 200, key = str(i))
            args.nbar -= (obs_nbar - target_nbar)
            print(f'[INFO] nbar = {args.nbar}, obs_nbar = {obs_nbar}, i = {i}')
        # args.mixRatio -= 0.1





def gen_test_data():
    args  = param()()
    from data import mixSPAC, mixSPAT
    root_path = '../test_data_nbarObs_1.3/'
    target_nbar = 1.31
    # if os.path.exists(root_path):
    #     os.system('rm -rf '+root_path)
    # os.makedirs(root_path)

    args.quantum_efficiency = 1.0
    args.nbar = 2.0
    args.mixRatio = 1.0
    nbar_th_spac = 2.0
    nbar_th_spat = 2.0


    print(f'[INFO] nbar = {args.nbar}, mixRatio = {args.mixRatio}')
    for i in range(11):
        args.path = os.path.join(root_path, f'QE_{args.quantum_efficiency}_mixRatio_{args.mixRatio}')
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        _ = False
        args.nbar = nbar_th_spac
        while not _:
            s = mixSPAC(args)
            s()
            s.estimate_nbar()
            _, obs_nbar = s.save(10, 200, target_nbar= target_nbar, key= str(i))
            args.nbar -= (obs_nbar - target_nbar)
            print(f'[INFO] nbar = {args.nbar}, obs_nbar = {obs_nbar}, i = {i}')
        nbar_th_spac = args.nbar
        _ = False
        args.nbar = nbar_th_spat
        while not _:
            s = mixSPAT(args)
            s()
            s.estimate_nbar()
            _, obs_nbar = s.save(10, 200, target_nbar= target_nbar, key= str(i))
            args.nbar -= (obs_nbar - target_nbar)
            print(f'[INFO] nbar = {args.nbar}, obs_nbar = {obs_nbar}')
        nbar_th_spat = args.nbar
        args.mixRatio -= 0.1
        args.mixRatio = round(args.mixRatio, 1)



if __name__ == "__main__":
    gen_test_data()
    gen_train_data()