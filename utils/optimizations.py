import numpy as np
from utils.data import prepare_train_data, do_cross_validation
from utils.vizualization import plot_acc_lambdas


def argmax(d):
    max_val = max(d.values())
    return [k for k in d if d[k] == max_val][0], max_val


def find_best_poly_lambda(x, y, degrees, lambdas, config, args, x_name, model_key=''):
    """
    Find best polynomial degree-lambda pair using cross validation.
    :param x: Features.
    :param y: Labels.
    :param degrees: Poly degree grid
    :param lambdas: Lambdas grid.
    :param config: Configuration parameters.
    :param model_key: String name if sub_model is used.
    :param x_name: Features names
    :param args: Command line arguments provided when run.
    :return:
    """
    res_tr, res_te = np.zeros((len(lambdas), len(degrees))), np.zeros((len(lambdas), len(degrees)))
    for i, ld in enumerate(lambdas):
        config["lambda"] = ld
        train_accs = []
        val_accs = []
        for index_degree, degree in enumerate(degrees):
            config["degree"] = degree
            # Cross validation
            tr_feats, tr_labels, _ = prepare_train_data(config, args, y, x, x_name=x_name, model_key=model_key)
            final_val_f1, final_train_f1, final_val_acc, final_train_acc = do_cross_validation(tr_feats, tr_labels,
                                                                                               config)
            print("Validation accuracy is {:.2f} %  % for degree {}".format(final_val_acc * 100, degree))

            train_accs.append(final_train_acc)
            val_accs.append(final_val_acc)
        res_tr[i] = train_accs
        res_te[i] = val_accs

        print("Finished for lambda {}".format(ld))

    # Get indices of best degree-lambda pair
    res_tr = res_tr.T
    res_te = res_te.T
    best_degree_tr, best_lambda_tr = np.unravel_index(np.argmax(res_tr, axis=None), res_tr.shape)
    best_degree_te, best_lambda_te = np.unravel_index(np.argmax(res_te, axis=None), res_te.shape)

    for i, d in enumerate(degrees):
        output_filename = config["viz_path"] + 'acc_vs_lambdas_' + str(model_key) + '_degree_' + str(d)
        plot_acc_lambdas(lambdas, res_tr[i], res_te[i], degree=d, model_key=model_key,
                         output_path=output_filename)

    print("For train, best lambda-degree pair is {} - {}".format(lambdas[best_lambda_tr], degrees[best_degree_tr]))
    print("For test, best lambda-degree pair is {} - {}".format(lambdas[best_lambda_te], degrees[best_degree_te]))
    return (degrees[best_degree_tr], lambdas[best_lambda_tr]), (degrees[best_degree_te], lambdas[best_lambda_te])


def find_best_reg_threshold(x, y, thresholds, config, args, x_name, model_key=''):
    """
    Find best regression threshold using cross validation.
    :param x: Features.
    :param y: Labels.
    :param thresholds: Thresholds grid.
    :param config: Configuration parameters.
    :param args: Command line arguments provided when run.
    :param x_name: Features names
    :param model_key: String name if sub_model is used.
    :return:
    """
    res_dict_tr = {}
    res_dict_te = {}
    for i, th in enumerate(thresholds):
        config["reg_threshold"] = th
        tr_feats, tr_labels, _ = prepare_train_data(config, args, y, x, x_name=x_name, model_key=model_key)
        final_val_f1, final_train_f1, final_val_acc, final_train_acc = do_cross_validation(tr_feats, tr_labels,
                                                                                           config)
        print("Validation f1 score is {:.2f} % and accuracy is {:.2f} % for thresh {}".format(final_val_f1 * 100,
                                                                                              final_val_acc * 100, th))

        res_dict_tr[th] = final_train_f1
        res_dict_te[th] = final_val_f1
        print("Finished for th {}".format(i))

    print("For train, best threshold is {}".format(argmax(res_dict_tr)[0][0]))
    print("For test, best threshold is {}".format(argmax(res_dict_te)[0][0]))
