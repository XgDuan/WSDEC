import os
import time
import json
import sys
import multiprocessing
import torch

class ModelSaver(object):
    """
    Manager model saving process and afterward eval process
    Saver will construct folders in the following structure:
        + params['runs']
            + params['alias']
                + model
                    + alias_time0.pkl
                    + alias_time1.pkl
                    + ...
                + submits
                    + alias_time0.json
                    + alias_time1.json
                    + ...
                + params.json
                + eval_result.json
    """

    def __init__(self, params, evaluator_path):
        """
        :param params:
        :param evaluator_path: absolute path
        """
        self.params = params
        self.root_folder = os.path.join(params['runs'], params['alias'])
        self.model_folder = os.path.join(self.root_folder, 'model')
        self.submits_folder = os.path.join(self.root_folder, 'submits')
        sys.path.append(evaluator_path)
        self._init_saver()
        self.time_format = '%m-%d-%H-%M-%S'
        with open(os.path.join(self.root_folder, 'params.json'), 'w') as file:
            json.dump(params, file)

    def _init_saver(self):
        if os.path.exists(self.root_folder):
            if self.params['alias'].startswith('test'):
                os.system('rm %s -rf'%self.root_folder)
                print('warning: remove test(%s) folder'%self.root_folder)
            else:
                print('error: alias already in use, abort')
                exit()
        if not os.path.exists(self.root_folder):
            os.makedirs(self.root_folder)
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        if not os.path.exists(self.submits_folder):
            os.makedirs(self.submits_folder)
        self.eval_result_path = os.path.join(self.root_folder, 'eval_result.txt')

    def _dump_eval_result(self, file_name):
        temp_file = file_name + '.tmp'
        os.system('sh scripts/evaluate_helper.sh %s %s'%(
            os.path.abspath(file_name), os.path.abspath(temp_file)))
        with open(temp_file, 'r') as result_file:
            line_number = 12 * 5  # 12 for each result, 5 for ios=[0.3, 0.5, 0.7, 0.9, ] and overall
            true_result = result_file.readlines()[-line_number:]
            with open(self.eval_result_path, 'a') as eval_result_file:
                eval_result_file.write('-'*80 + '\n')
                eval_result_file.write(file_name + '\n')
                eval_result_file.writelines(true_result)
                eval_result_file.write('\n')
        os.system('rm %s'%temp_file)

    def save_model(self, model, step, dynamic_params):
        """
        :param model: model.state_dict() to be saved
        :param dynamic_params: dynamic params that used to reconstruct training process
        :return: None
        """
        model_path = os.path.join(self.model_folder,
                            '%s_%05d_%s.ckp' % (self.params['alias'], step,
                                                time.strftime(self.time_format, time.localtime())))
        torch.save({'state_dict': model.state_dict(),
                    'dynamic_params': dynamic_params,
                    'params': self.params},
                   open(model_path, 'w'))

    def save_submits(self, submits, step):
        file_name = os.path.join(self.submits_folder,
                                 '%s_%05d_%s_%s.json' % (self.params['alias'],
                                    step, self.params['val_data'].split('/')[-1],
                                    time.strftime(self.time_format, time.localtime())))
        with open(file_name, 'w') as file:
            json.dump(submits, file)

        # construct
        process = multiprocessing.Process(target=self._dump_eval_result, args=(file_name,))
        process.start()

    def load_model(self, model_path):
        file_obj = torch.load(open(model_path, 'r'))
        return file_obj['state_dict'], file_obj['params']

    def load_model_slcg(self, model_path):
        file_obj = torch.load(open(model_path, 'r'))
        model_sl = file_obj['dynamic_params']['model_sl']
        model_cg = file_obj['dynamic_params']['model_cg']

        return model_sl, model_cg, file_obj['params']
