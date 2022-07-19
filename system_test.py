import configparser
import yaml
from multiprocessing import Process, Queue
from MCCache.sys_op import WorkloadTest, SystemOperator
import os
import time


def backend_config_parse(filename='config/backend.ini'):
    """Parse backend connection configurations."""
    parser = configparser.ConfigParser()
    parser.read(filename)
    db_conn_param, cache_conn_param = {}, {}
    if parser.has_section("system"):
        params = parser.items(section="system")
        for param in params:
            if param[0] == "db":
                db_conn_param[param[0]] = param[1]
            if param[0] == "cache":
                cache_conn_param[param[0]] = param[1]
    else:
        print('[system] section not found in {}, exiting.'.format(filename))
        exit(1)
    db_params = parser.items(section=db_conn_param["db"])
    for param in db_params:
        db_conn_param[param[0]] = param[1]
    cache_params = parser.items(section=cache_conn_param["cache"])
    for param in cache_params:
        cache_conn_param[param[0]] = param[1]
    return db_conn_param, cache_conn_param


def alg_config_parse(filename='config/sysTest.yaml'):
    with open(filename, 'r') as fp:
        try:
            alg_dict = yaml.safe_load(fp)
            alg_dict['dataset_dir'] = alg_dict['dataset_root'] + "/" + alg_dict['dataset_name']
            return alg_dict
        except yaml.YAMLError as exc:
            print(exc)


def system_process(config_file, queue, sys_test):
    """Process DB and Cache operations."""
    # ---------- debug only ----------
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

    db_conn_param, cache_conn_param = backend_config_parse(config_file)
    if sys_test:
        sys_op = SystemOperator(db_conn_param, cache_conn_param)
        print('SYS Start Time: {}'.format(time.time()))
        sys_op.operation_exe(op_queue=queue)
        print('SYS End Time: {}'.format(time.time()))
        sys_op.cleanup()


def alg_process(alg_dict, queue):
    alg_test = WorkloadTest(dataset_dir=alg_dict['dataset_dir'], sys_test=alg_dict['sys_test'])
    print('ALG Start time: {}'.format(time.time()))
    alg_test.batch_test(queue, csize=alg_dict['csize'], cache_scheme=alg_dict['cache_scheme'], alg_name=alg_dict['alg_name'], batch_size=alg_dict['batch_size'], opt_len=alg_dict['opt_len'], fetch_strategy=alg_dict['fetch_strategy'], staleness_bound=alg_dict['staleness_bound'], ob_acc=alg_dict['ob_acc'])
    print('ALG End Time: {}'.format(time.time()))


if __name__ == '__main__':
    alg_dict = alg_config_parse('config/sysTest.yaml')
    op_queue = Queue()  # system operation queue
    sys_test = alg_dict['sys_test']
    p = Process(target=system_process, args=('config/backend.ini', op_queue, sys_test))
    p.start()
    alg_process(alg_dict, op_queue)
    p.join()
