import base64
import time
import json
import os
from dd_client import DD
import cStringIO
from PIL import Image
from PIL import ExifTags


from scipy import spatial


class DDImage(object):

    mllib = 'caffe'

    def __init__(self, host):

        self.host = host
        self.sname = ''
        self.description = ''
        self.mltype = ''
        self.extract_layer = ''
        self.nclasses = 0
        self.width = self.height = 224
        self.binarized = False
        self.GPU = True
        self.model_repo = ''
        self.dd = None
        self.model_name = ''
        self.batch_size = None
        self.GOOGLENET_BATCH_SIZE = 40
        self.RESNET152_BATCH_SIZE = 4
        self.model_repo

    def init_for_classification(self, model_name, model_repo):
        self.sname = "_classification"
        self.description = self.sname + ' for classification'
        self.mltype = 'supervised'
        self.nclasses = 1000
        self.model_name = model_name
        self.model_repo=model_repo
        self.dd = DD(self.host)
        self.dd.set_return_format(self.dd.RETURN_PYTHON)

    def init_for_extraction(self, model_name, model_repo):
        self.sname = model_name + "_extraction"
        self.description = self.sname + ' for feature extraction'
        self.mltype = 'unsupervised'
        self.model_name = model_name
        self.model_repo = model_repo

        if model_name == 'resnet152':
            self.extract_layer = 'fc1000'
            self.nclasses = 1000
            self.batch_size = self.RESNET152_BATCH_SIZE
        elif model_name == 'googlenet':
            self.extract_layer = 'pool5/7x7_s1'
            self.nclasses = 1000
            self.batch_size = self.GOOGLENET_BATCH_SIZE
        else:
            print("ERROR IN INIT FOR MODEL " + model_name)
            return

        self.dd = DD(self.host)
        self.dd.set_return_format(self.dd.RETURN_PYTHON)

    def create_service(self):
        model = {'repository': self.model_repo, 'templates': '../templates/caffe/'}
        parameters_input = {'connector': 'image', 'width': self.width, 'height': self.height}
        parameters_mllib = {'nclasses': self.nclasses}
        parameters_output = {}

        response = self.dd.put_service(self.sname, model, self.description, self.mllib, parameters_input,
                                       parameters_mllib, parameters_output,
                                       self.mltype)
        print (response)

    def classify_image_from_file(self, image_file_path):
        img = Image.open(image_file_path)
        img = self._rotate_image_if_needed(img, image_file_path)

        base64_img = self._convert_img_to_base64(img)
        img.close()

        img_data = [base64_img]

        classes = self._make_image_classification_call(img_data)
        if classes is not None:
            print(image_file_path + ': ' + classes['cat'] + ' with probability ' + str(classes['prob']))
            return classes
        else:
            print("error making classification call for " + image_file_path)
            return None

    def extract_features_from_image_file(self, image_file_path):
        image_data = []
        img = Image.open(image_file_path)
        img = self._rotate_image_if_needed(img, image_file_path)

        base64_img = self._convert_img_to_base64(img)
        img.close()
        image_data.append(base64_img)

        response = self._make_feature_extraction_call(image_data)

        if response is not None:
            print("feature vector for " + image_file_path + " is:")
            print(response['vals'])
            return response['vals']
        else:
            print ("failed to extract features for " + image_file_path)
            return None

    def extract_features_from_url(self, url):
        url_input_list = [url]
        response = self._make_feature_extraction_call(url_input_list)

        if response is not None:
            print("vector for " + response['uri'] + " is:")
            print(response['vals'])
            return response['vals']
        else:
            print("error extracting features for " + url)
            return None

    def extract_features_from_url_file(self, data_file, target_file):
        print("processing " + data_file + '\n')

        urls = []
        url_file = open(data_file, 'r')
        for url in url_file:
            urls.append(url.rstrip('\n'))

        print("total number of image URL's to process " + str(len(urls)))

        uri_to_features = {}
        image_processed = 0
        num_of_erros = 0

        start_time = time.time()
        while len(urls) > 0:
            print("remaining images to process: " + str(len(urls)))
            if self.batch_size > len(urls):
                self.batch_size = len(urls)

            urls_to_process = []
            for i in range(0, self.batch_size):
                urls_to_process.append(urls.pop())

            t0 = time.time()
            responses = self._make_feature_extraction_call(urls_to_process)

            if responses is not None:
                for response in responses:
                    uri_to_features[response['uri']] = response['vals']
                else:
                    num_of_erros += 1

            t1 = time.time()
            image_processed = image_processed + self.batch_size
            print("total time it took to process " + str(i) + " images was " + str(t1 - t0) + ' seconds')

        full_file = open(target_file, 'w')
        json.dump(uri_to_features, full_file)
        full_file.close()

        end_time = time.time()
        total_time = (end_time - start_time) / 60

        self._report_stats(image_processed, num_of_erros, target_file, total_time)

        return uri_to_features

    def extract_features_from_files_in_dir(self, dir_path, target_file_path):

        image_files = self._fetch_files_from_dir(dir_path)
        image_vectors = {}

        while len(image_files) > 0:
            img_path = image_files.pop()
            img_data = self._pre_process_image(img_path)
            response = self._make_feature_extraction_call(img_data)
            image_vectors[img_path] = response['vals']
            print("files left to process " + str(len(image_files)))

        target_file = open(target_file_path, 'w')
        json.dump(image_vectors, target_file)
        target_file.close()

    def classify_images_from_dir(self, dir_path):

        image_files = self._fetch_files_from_dir(dir_path)

        while (len(image_files)) > 0:
            image_file_path = image_files.pop()
            img_data = self._pre_process_image(image_file_path)

            classes = self._make_image_classification_call(img_data)
            if classes is not None:
                print(image_file_path + ': ' + classes['cat'] + ' with probability ' + str(classes['prob']))
            else:
                print("error making classification call for " + image_file_path)

    def delete_service(self):
        self.dd.delete_service(self.sname, clear='')

    def _make_image_classification_call(self, img_data):
        parameters_input = {}
        parameters_mllib = {}
        parameters_output = {'best': 1}
        prediction_call = None
        # noinspection PyBroadException
        try:
            t0 = time.time()
            prediction_call = self.dd.post_predict(self.sname, img_data, parameters_input, parameters_mllib,
                                                   parameters_output)
            classes = prediction_call['body']['predictions']['classes']
            t1 = time.time()
            print ("received response in " + str(t1 - t0) + " seconds")

            return classes
        except Exception:
            print("Error in response!! " + str(prediction_call['status']))
            return None

    def _make_feature_extraction_call(self, image_data):
        parameters_input = {}
        parameters_mllib = {'gpu': self.GPU, 'extract_layer': self.extract_layer}
        parameters_output = {'binarized': self.binarized}
        predict_call = None
        try:
            t0 = time.time()
            predict_call = self.dd.post_predict(self.sname, image_data, parameters_input, parameters_mllib,
                                                parameters_output)
            response = predict_call['body']['predictions']
            t1 = time.time()
            print ("received responses in " + str(t1 - t0) + " seconds")
            return response
        except Exception:
            if predict_call is not None:
                print("Error in server response " + str(predict_call['status']))
            else:
                print("Error in server response")
        return None

    def _convert_img_to_base64(self, img):
        tmp_buffer = cStringIO.StringIO()
        img.save(tmp_buffer, format="JPEG")
        base64_img_str = base64.b64encode(tmp_buffer.getvalue())
        tmp_buffer.close()
        return base64_img_str

    def _rotate_image_if_needed(self, img, path):
        try:
            exif = dict((ExifTags.TAGS[k], v) for k, v in img._getexif().items() if k in ExifTags.TAGS)
            if exif['Orientation'] == 6:
                print("rotating image " + path)
                img = img.rotate(270, expand=True)
        except Exception:
            print(path + " probably does not need rotation")
        return img

    def _pre_process_image(self, image_path):
        image_data = []
        img = Image.open(image_path)
        img = self._rotate_image_if_needed(img, image_path)
        base64_img = self._convert_img_to_base64(img)
        image_data.append(base64_img)
        img.close()

        return image_data

    def _fetch_files_from_dir(self, dir_path):
        image_files = []

        for file_name in os.listdir(dir_path):
            if file_name.endswith(".jpg") or file_name.endswith(".JPG"):
                image_files.append(dir_path + "/" + file_name)

        print("fetched a total of " + str(len(image_files)) + " from " + dir_path)
        return image_files

    def _report_stats(self, image_processed, num_of_erros, target_file, total_time):
        print ("\nfinished successfully and wrote final file " + target_file)
        print("Total images processed: " + str(image_processed))
        print("Total errors: " + str(num_of_erros))
        print("Total time was " + str(total_time) + " minutes")


# USAGE EXAMPLES
# ----------------

# create the dede client with the appropriate host
client = DDImage('host')

# Initlize for classification and classify images from local file system
client.init_for_classification("googlenet",PATH_TO_MODEL_REPO)
client.create_service()
client.classify_image_from_file('PATH_TO_FILE')
client.classify_images_from_dir('DIR_PATH')
client.delete_service()

# Initilize for extraction, and then extract from various sources. compute distance between two vectors
client.init_for_extraction("resnet152",PATH_TO_MODEL_REPO)
client.create_service()
feature_vector = client.extract_features_from_image_file(PATH_TO_IMAGE_FILE)
feature_vector2 = client.extract_features_from_url(URL)
result = spatial.distance.cosine(feature_vector, feature_vector2)
print("distance between two vectors is " + str(result))

#This will extract features from urls or files and output feature vectors to a JSON file
client.extract_features_from_url_file(PATH_TO_URL_FILE, PATH_TO_OUTPUT_JSON_FILE)
client.extract_features_from_files_in_dir(PATH_TO_DIR',PATH_TO_OUTPUT_JSON_FILE)
client.delete_service()
