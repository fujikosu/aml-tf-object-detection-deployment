def init():
    import tensorflow as tf
    global sess, image_tensor, boxes, scores, classes, num_detections, tensor_dict

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('./test_ckpt/ssd_inception_v2.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name
                for op in ops for output in op.outputs
            }
            tensor_dict = {}
            for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph(
                    ).get_tensor_by_name(tensor_name)
                image_tensor = tf.get_default_graph().get_tensor_by_name(
                    'image_tensor:0')
                sess = tf.Session(graph=detection_graph)


def run(reqStr):
    import json
    import numpy as np
    import base64
    from PIL import Image
    from io import BytesIO

    # decode the image
    images = json.loads(reqStr)
    base64ImgString = images[0]
    if base64ImgString.startswith('b\''):
        base64ImgString = base64ImgString[2:-1]
    base64Img = base64ImgString.encode('utf-8')
    decoded_img = base64.b64decode(base64Img)
    img_buffer = BytesIO(decoded_img)
    image = Image.open(img_buffer)
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((im_height, im_width,
                                                  3)).astype(np.uint8)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Run inference
    output_dict = sess.run(
        tensor_dict, feed_dict={image_tensor: image_np_expanded})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][
        0].astype(np.uint8).tolist()
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0].tolist()
    output_dict['detection_scores'] = output_dict['detection_scores'][
        0].tolist()

    return (json.dumps(output_dict))
