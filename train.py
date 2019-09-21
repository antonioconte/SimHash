from SimHashModel import SimHashModel


# SimHashModel(type="", k='3', t=10, isDataProc=False)


def train_all(k = '3'):
    type = ['paragraph', 'section', 'phrase']
    for t in type:
        model = SimHashModel(type=t, k=k, isDataProc=True)
        model.train()
        import gc
        gc.collect()
    # model = Minhash('trigram',k='3')
    # model.train(config.filepath)
    exit()


if __name__ == '__main__':

    # ===== TRAIN ALL ======================================
    # config.DEBUG = True
    #  k = { '1','2', '3'}
    train_all(k='3')
