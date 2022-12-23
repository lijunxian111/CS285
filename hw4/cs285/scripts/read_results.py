import glob
import tensorflow as tf

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

if __name__ == '__main__':
    import glob

    logdir = 'data/hw4_q1_cheetah_n5_arch2x250_cheetah-cs285-v0_23-12-2022_11-16-55/event*'
    #eventfile = glob.glob(logdir)[0]

    eventfile='D:\PycharmProjects\pythonProject3\data\hw4_q3_obstacles_obstacles-cs285-v0_23-12-2022_13-29-12\events.out.tfevents.1671773352.LAPTOP-FK0K2T47'

    X, Y = get_section_results(eventfile)
    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))