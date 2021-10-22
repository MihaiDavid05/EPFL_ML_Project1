import numpy as np
from utils.vizualization import bias_variance_decomposition_visualization
from utils.data import split_data, prepare_test_data, prepare_train_data, do_cross_validation
from utils.algo import predict_labels, get_f1
from utils.implementations import reg_logistic_regression, logistic_regression, ridge_regression


def argmax(d):
    if not d:
        return None
    max_val = max(d.values())
    return [k for k in d if d[k] == max_val][0], max_val


def find_best_poly_lambda(x, y, config, args, output_filename):
    """
    Find best polynoial degree - lambda pair using data splits.
    :param x:
    :param y:
    :param config:
    :param args:
    :param output_filename:
    :return:
    """
    lambdas = np.logspace(-4, 0, 5)
    degrees = list(range(3, 10))
    seeds = range(3)
    # Be sure to check config given as cli param before setting other parameters here
    ratio_train = 0.8
    config["build_poly"] = True

    f1_tr = np.empty((len(seeds), len(degrees)))
    f1_te = np.empty((len(seeds), len(degrees)))

    res_dict_tr = {}
    res_dict_te = {}
    for i, ld in enumerate(lambdas):
        config["lambda"] = ld
        for index_seed, seed in enumerate(seeds):
            x_tr, x_te, y_tr, y_te = split_data(x, y, ratio_train, seed=index_seed)
            for index_degree, degree in enumerate(degrees):

                config["degree"] = degree
                tr_feats, tr_labels, tr_stat = prepare_train_data(config, args, y_tr, x_tr)
                te_feats, te_labels, te_stat = prepare_test_data(config, tr_stat[0], tr_stat[1], x_te, y_te)
                if config['lambda'] is not None:
                    if config['model'] == 'ridge':
                        weights, tr_loss = ridge_regression(tr_labels, tr_feats, config['lambda'])
                    else:
                        weights, tr_loss = reg_logistic_regression(tr_labels, tr_feats, config['lambda'],
                                                                   np.zeros((tr_feats.shape[1], 1)),
                                                                   config['max_iters'], config['gamma'])
                else:
                    weights, tr_loss = logistic_regression(tr_labels, tr_feats, np.zeros((tr_feats.shape[1], 1)),
                                                           config['max_iters'],
                                                           config['gamma'])
                tr_preds = predict_labels(weights, tr_feats, config["reg_threshold"])
                te_preds = predict_labels(weights, te_feats, config["reg_threshold"])

                f1_score_tr = get_f1(tr_preds, tr_labels)
                f1_score_te = get_f1(te_preds, te_labels)
                res_dict_tr[(ld, degree)] = f1_score_tr
                res_dict_te[(ld, degree)] = f1_score_te

                f1_tr[index_seed, index_degree] = f1_score_tr
                f1_te[index_seed, index_degree] = f1_score_te
        print("Finished for lambda {}".format(i))

    output_path = config["viz_path"] + output_filename
    bias_variance_decomposition_visualization(degrees, f1_tr, f1_te, output_path)
    print("For training, best lambda is {} and best degree is {}".format(argmax(res_dict_tr)[0][0],
                                                                         argmax(res_dict_tr)[0][1]))
    print("For test, best lambda is {} and best degree is {}".format(argmax(res_dict_te)[0][0],
                                                                     argmax(res_dict_te)[0][1]))


def find_best_poly_lambda_cross_val(x, y, config, args):
    """
    Find best polynoial degree - lambda pair using cross validation.
    :param x:
    :param y:
    :param config:
    :param args:
    :return:
    """
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
            final_val_f1, final_train_f1 = do_cross_validation(tr_feats, tr_labels, config['lambda'], config)
            print("Validation f1 score is {:.2f} %".format(final_val_f1 * 100))

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
    :param x:
    :param y:
    :param config:
    :param args:
    :return:
    """
    thresholds = np.linspace(0.01, 0.05, 5)
    # TODO: Check this: Best regression threshold seems to be at 0.01 !!!
    # Be sure to check config given as cli param before setting other parameters here
    config["max_iters"] = 4000
    config["lambda"] = 1  # or 0.001
    config["degree"] = 6

    res_dict_tr = {}
    res_dict_te = {}
    for i, th in enumerate(thresholds):
        config["reg_threshold"] = th
        tr_feats, tr_labels, _ = prepare_train_data(config, args, y, x)
        final_val_f1, final_train_f1 = do_cross_validation(tr_feats, tr_labels, config['lambda'], config)
        print("Validation f1 score is {:.2f} %".format(final_val_f1 * 100))

        res_dict_tr[th] = final_train_f1
        res_dict_te[th] = final_val_f1
        print("Finished for th {}".format(i))

    print("For train, best threshold is {}".format(argmax(res_dict_tr)[0][0]))
    print("For test, best threshold is {}".format(argmax(res_dict_te)[0][0]))
