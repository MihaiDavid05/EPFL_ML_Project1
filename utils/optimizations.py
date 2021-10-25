import numpy as np
from utils.data import prepare_train_data, do_cross_validation


def argmax(d):
    max_val = max(d.values())
    return [k for k in d if d[k] == max_val][0], max_val


def find_best_poly_lambda(x, y, config, args):
    """
    Find best polynomial degree-lambda pair using cross validation.
    :param x: Features.
    :param y: Labels.
    :param config: Configuration parameters.
    :param args: Command line arguments provided when run.
    :return:
    """
    # Set grid values
    lambdas = np.logspace(-4, 0, 5)
    degrees = list(range(3, 8))

    # Be sure to check config given as cli param before setting other parameters here
    config["max_iters"] = 4000
    config["reg_threshold"] = 0.5

    res_dict_tr = {}
    res_dict_te = {}
    for i, ld in enumerate(lambdas):
        config["lambda"] = ld
        for index_degree, degree in enumerate(degrees):
            config["degree"] = degree
            # Cross validation
            tr_feats, tr_labels, _ = prepare_train_data(config, args, y, x)
            final_val_f1, final_train_f1, final_val_acc, _ = do_cross_validation(tr_feats, tr_labels, config)
            print("Validation f1 score is {:.2f} % and accuracy is {:.2f} %".format(final_val_f1 * 100,
                                                                                    final_val_acc * 100))

            res_dict_tr[(ld, degree)] = final_train_f1
            res_dict_te[(ld, degree)] = final_val_f1
        print("Finished for lambda {}".format(i))

    print("For train, best lambda-degree pair is {} - {}".format(argmax(res_dict_tr)[0][0],
                                                                 argmax(res_dict_tr)[0][1]))
    print("For test, best lambda-degree pair is {} - {}".format(argmax(res_dict_te)[0][0],
                                                                argmax(res_dict_te)[0][1]))


def find_best_reg_threshold(x, y, config, args):
    """
    Find best regression threshold using cross validation.
    :param x: Features.
    :param y: Labels.
    :param config: Configuration parameters.
    :param args: Command line arguments provided when run.
    :return:
    """
    # Set grid values
    thresholds = np.linspace(0.01, 0.05, 5)

    # Be sure to check config given as cli param before setting other parameters here
    config["max_iters"] = 4000
    config["lambda"] = 1
    config["degree"] = 6

    res_dict_tr = {}
    res_dict_te = {}
    for i, th in enumerate(thresholds):
        config["reg_threshold"] = th
        tr_feats, tr_labels, _ = prepare_train_data(config, args, y, x)
        final_val_f1, final_train_f1, final_val_acc, _ = do_cross_validation(tr_feats, tr_labels, config)
        print("Validation f1 score is {:.2f} % and accuracy is {:.2f} %".format(final_val_f1 * 100,
                                                                                final_val_acc * 100))

        res_dict_tr[th] = final_train_f1
        res_dict_te[th] = final_val_f1
        print("Finished for th {}".format(i))

    print("For train, best threshold is {}".format(argmax(res_dict_tr)[0][0]))
    print("For test, best threshold is {}".format(argmax(res_dict_te)[0][0]))
