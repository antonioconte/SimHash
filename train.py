from SimHashModel import SimHashModel


# SimHashModel(type="", k='3', t=10, isDataProc=False)


def train_all(k = '3',tolerance=30,SGN_L = 128):
    type = ['paragraph', 'section', 'phrase']
    for t in type:
        model = SimHashModel(type=t, k=k, T=tolerance,sign=SGN_L,isDataProc=True)
        model.train()
        import gc
        gc.collect()
    # model = Minhash('trigram',k='3')
    # model.train(config.filepath)
    # exit()


if __name__ == '__main__':
    TOLERANCE = 30
    SGN_L = 128
    K = '3'
    # ===== TRAIN ALL ======================================
    # config.DEBUG = True
    #  k = { '1','2', '3'}
    train_all(k=K,tolerance=TOLERANCE,SGN_L=SGN_L)
