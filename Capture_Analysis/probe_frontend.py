#!/usr/bin/env python3
"""Probe the USRP front-end: gain range, DC-offset support, current settings."""
from gnuradio import uhd

src = uhd.usrp_source(','.join(('', 'addr=192.168.10.4')),
                      uhd.stream_args(cpu_format='fc32', args='', channels=[0]))
src.set_subdev_spec('A:AB', 0)
src.set_antenna('RXA', 0)

try:
    gr = src.get_gain_range(0)
    print(f'gain range : {gr.start()} .. {gr.stop()} dB  step {gr.step()}')
except Exception as e:
    print('gain range : ERR', e)
try:
    print(f'gain names : {src.get_gain_names(0)}')
except Exception as e:
    print('gain names : ERR', e)
try:
    print(f'cur gain   : {src.get_gain(0)} dB   (normalized {src.get_normalized_gain(0):.3f})')
except Exception as e:
    print('cur gain   : ERR', e)

# Hardware DC-offset auto-correction support
for label, fn in (('set_dc_offset(True)', lambda: src.set_dc_offset(True, 0)),):
    try:
        fn(); print(f'{label}: accepted')
    except Exception as e:
        print(f'{label}: ERR {e}')
