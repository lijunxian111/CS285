# -*- coding: utf-8 -*-
import glob
import tensorflow as tf

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

if __name__ == '__main__':
    #logdir = 'data/q1_lb_no_rtg_dsa_CartPole-v0_12-11-2020_23-05-55/events*'
    eventfile = 'D:\PycharmProjects\pythonProject3\data\q2_pg_q5_b2000_r0.001_lambda_0.99_Hopper-v3_24-12-2022_23-43-13\events.out.tfevents.1671896593.LAPTOP-FK0K2T47'

    X, Y = get_section_results(eventfile)
    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))