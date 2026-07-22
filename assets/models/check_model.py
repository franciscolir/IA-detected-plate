import tensorflow as tf
import sys

interpreter = tf.lite.Interpreter(model_path=sys.argv[1])
interpreter.allocate_tensors()

for kind, details in [("Input", interpreter.get_input_details()), ("Output", interpreter.get_output_details())]:
    for i, d in enumerate(details):
        print(f'{kind}[{i}]: shape={d["shape"]}, dtype={d["dtype"]}, name={d["name"]}')