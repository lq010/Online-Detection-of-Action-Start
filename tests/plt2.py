import matplotlib.pyplot as plt


def plot_history(history):

    plt.plot(history['acc'], marker='.')
    plt.plot(history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    # plt.axis([0, len(history['acc'])])
    plt.xticks([i for i in range(len(history['acc']))][::2])
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig('model_accuracy.png')
    plt.close()

    # plt.ylim(0, 2.5)
    plt.plot(history['loss'], marker='.')
    plt.plot(history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.xticks([i for i in range(len(history['acc']))][::2])
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig('model_loss.png')
    plt.close()

if __name__ == '__main__':
    history = dict()
    with open("//media/lq/C13E-1ED0/dataset/UCF_Crimes/results/adam_c3d_zz_10lr_pre_ABG/log.csv", "r") as f:
        name = f.readline().strip().split(",")
        for n in name:
            history[n] = []
        history['val_loss'] =[]
        history['acc'] =[]
        history['val_acc']=[]
        print (history)
        for i in range(16):
            v = f.readline().strip().split(",")
            print(v)
            # history['epoch'].append(int(v[0]))
            # history['loss'].append(float(v[2]))
            # history['acc'].append(float(v[1]))
            # history['val_loss'].append(float(v[4]))
            # history['val_acc'].append(float(v[3]))
            
            history['epoch'].append(int(v[0]))
            
            history['acc'].append(float(v[1]))
            
            history['loss'].append(float(v[3]))

            history['val_acc'].append(float(v[6]))
            history['val_loss'].append(float(v[8]))

    plot_history(history)