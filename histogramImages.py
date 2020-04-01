from __main__ import *


def orangeHist():
    count=0
    redhistr = []
    bluehistr = []
    greenhstr = []
    Otrain=[]
    for filename in os.listdir("orangeSamples"):
        image = cv2.imread(os.path.join("orangeSamples", filename))

        # fig, (ax1, ax2, ax3) = plt.subplots(3)
        # fig.suptitle('orangeSample'+str(count))
        color = ('b', 'g', 'r')
        rgbcount=0

        for i, col in enumerate(color):
            histr = cv2.calcHist([image], [i], None, [256], [0, 256])
            # print(rgbcount)
            rgbcount+=1

            if col=='b':
                # ax1.plot(histr, color=col)
                if count>0:
                    bluehistr=bluehistr+histr
                else:
                    bluehistr=histr
            elif col=='g':
                # ax2.plot(histr, color=col)
                if count >0:
                    greenhstr = greenhstr+ histr
                else:
                    greenhstr = histr
            elif col=='r':
                # ax3.plot(histr, color=col)
                if count>0:
                    redhistr=redhistr+histr
                else:
                    redhistr=histr


            # plt.xlim([0, 256])
        # print('\n')
        # print(bluehistr)
        count += 1
        for j in range(image.shape[0]):
            Otrain.append(image[i, :])
    Otrain = np.array(Otrain)

    np.save('Otrain', Otrain)



    # plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('orangeSample All')
    ax1.plot(bluehistr, color='b')
    ax2.plot(greenhstr, color='g')
    ax3.plot(redhistr, color='r')

    plt.xlim([0, 256])
    plt.show()

    print("Done")
    return bluehistr,greenhstr,redhistr

def greenHist():
    count=0
    redhistr = []
    bluehistr = []
    greenhstr = []
    Gtrain=[]
    for filename in os.listdir("greenSamples"):
        image = cv2.imread(os.path.join("greenSamples", filename))

        # fig, (ax1, ax2, ax3) = plt.subplots(3)
        # fig.suptitle('greenSamples'+str(count))
        color = ('b', 'g', 'r')
        rgbcount=0

        for i, col in enumerate(color):
            histr = cv2.calcHist([image], [i], None, [256], [0, 256])
            # print(rgbcount)
            rgbcount+=1

            if col=='b':
                # ax1.plot(histr, color=col)
                if count>0:
                    bluehistr=bluehistr+histr
                else:
                    bluehistr=histr
            elif col=='g':
                # ax2.plot(histr, color=col)
                if count >0:
                    greenhstr = greenhstr+ histr
                else:
                    greenhstr = histr
            elif col=='r':
                # ax3.plot(histr, color=col)
                if count>0:
                    redhistr=redhistr+histr
                else:
                    redhistr=histr


            # plt.xlim([0, 256])
        # print('\n')
        # print(bluehistr)
        count += 1
        for j in range(image.shape[0]):
            Gtrain.append(image[i, :])
    Gtrain = np.array(Gtrain)

    # np.save('Gtrain', Gtrain)
    # plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('greenSamples All')
    ax1.plot(bluehistr, color='b')
    ax2.plot(greenhstr, color='g')
    ax3.plot(redhistr, color='r')

    plt.xlim([0, 256])
    plt.show()

    print("Done")

    return bluehistr,greenhstr,redhistr


def yellowHist():
    count=0
    redhistr = []
    bluehistr = []
    greenhstr = []
    Ytrain=[]
    for filename in os.listdir("yellowSamples"):
        image = cv2.imread(os.path.join("yellowSamples", filename))

        # fig, (ax1, ax2, ax3) = plt.subplots(3)
        # fig.suptitle('yellowSamples'+str(count))
        color = ('b', 'g', 'r')
        rgbcount=0

        for i, col in enumerate(color):
            histr = cv2.calcHist([image], [i], None, [256], [0, 256])
            # print(rgbcount)
            rgbcount+=1

            if col=='b':
                # ax1.plot(histr, color=col)
                if count>0:
                    bluehistr=bluehistr+histr
                else:
                    bluehistr=histr
            elif col=='g':
                # ax2.plot(histr, color=col)
                if count >0:
                    greenhstr = greenhstr+ histr
                else:
                    greenhstr = histr
            elif col=='r':
                # ax3.plot(histr, color=col)
                if count>0:
                    redhistr=redhistr+histr
                else:
                    redhistr=histr


            # plt.xlim([0, 256])
        # print('\n')
        # print(bluehistr)
        count += 1
        for j in range(image.shape[0]):
            Ytrain.append(image[i, :])
    Ytrain = np.array(Ytrain)

    # np.save('Ytrain', Ytrain)
    # plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('yellowSamples All')
    ax1.plot(bluehistr, color='b')
    ax2.plot(greenhstr, color='g')
    ax3.plot(redhistr, color='r')

    plt.xlim([0, 256])
    plt.show()

    print("Done")

    return bluehistr,greenhstr,redhistr
