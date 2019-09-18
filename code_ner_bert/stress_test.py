import os
import time


tracer_input = '/home/lawrench/OPERA/workspace/input/tracer_corpus/output/raw_text_ltf_lang_corrected/eng/'
tracer_output = '/home/lawrench/OPERA/workspace/output/tracer_corpus/raw_text_output/xianyang_entity_out/'
E01_input = '/home/lawrench/OPERA/workspace/output/LDC2018E01/ldc_ltf_origin_lang_corrected/'
E01_output = '/home/lawrench/OPERA/workspace/output/LDC2018E01/ltf_output/xianyang_entity_out/'
E52_input = '/home/lawrench/OPERA/workspace/output/LDC2018E52/ldc_ltf_origin_lang_corrected/'
E52_output = '/home/lawrench/OPERA/workspace/output/LDC2018E52/ltf_output/xianyang_entity_out/'

with open('stress_log', 'w') as f:
    for i in range(100):
        time1 = time.time()
        os.system('python2 main.py {} {}'.format(tracer_input, tracer_output))
        time2 = time.time()
        f.write('tracer {}: {}'.format(i, time2 - time1))
        os.system('python2 main.py {} {}'.format(E01_input, E01_output))
        time3 = time.time()
        f.write('E01 {}: {}'.format(i, time3 - time2))
        os.system('python2 main.py {} {}'.format(E52_input, E52_output))
        time4 = time.time()
        f.write('E52 {}: {}'.format(i, time4 - time3))