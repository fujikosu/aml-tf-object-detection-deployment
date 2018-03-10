# TensorFlow Object Detection model deployment by Azure Machine Learning
The [full tutorial](https://github.com/Azure/MachineLearningSamples-tf/tree/RuonanO16N) contains all the steps and code to train and operationalize a model using Azure ML. [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) page contains all details about TensorFlow Object Detection API. The abbreviated instructions here focus on creating the Docker container with a pre-trained object detection model and deploying it as API by Azure Machine Learning.

1. Open the [Azure ML CLI](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-service-deploy) and set your environment

2. You need these files:
* test_ckpt/ssd_inception_v2.pb
* conda_dependencies.yml
* webservice_driver.py

This below is the command to deploy the trained model by Azure ML CLI.
```
az ml service create realtime -m ./test_ckpt/ssd_inception_v2.pb -f webservice_driver.py -n [your service name] -r python -c conda_dependencies.yml
```

You can get other pre-trained models from here [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), or you can bring your own trained model.

3. Test the model

First, use ```az ml service usage realtime -i [your service name]``` and check your scoring URL.

Update YOUR_PORT in webservice_invoke.py with your port number from scoring URL and test the API.

```
python webservice_invoke.py
```

4. Your Docker image is now stored in Azure Container Registry and ready for [deployment to an Azure IoT Edge device](https://docs.microsoft.com/en-us/azure/machine-learning/preview/deploy-to-iot-edge-device) or [consumption](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-consumption).