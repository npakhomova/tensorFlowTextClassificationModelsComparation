import cx_Oracle
import requests
import json
import tensorflow as tf

import re
import os
import os.path
import collections
import urllib
import numpy as np
import csv
import sys
import operator
import math


from tensorflow.contrib import learn
from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS

headers = {'Connection':'close','content-type': 'application/json'}


tf.app.flags.DEFINE_string(
    'model_dir', '/Users/npakhomova/modelTrainingSet/MenProdyctType_2016-11-26_17-57-08/',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")

tf.app.flags.DEFINE_integer('num_top_predictions', 20,
                            """Display this many predictions.""")

MAX_DOCUMENT_LENGTH = 70
NUMBER_OF_CATEGORIES = 29
EMBEDDING_SIZE = 50

### Training data
def loadData(filename):
    with gfile.Open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        data, target = [], []
        for row in data_file:
            target.append(row.pop(0))
            data.append(row.pop(0))

    target = np.array(target, dtype=np.int32)
    data = np.array(data, dtype=np.str)
    return data, target


def bag_of_words_model(x, y):
  """A bag-of-words model. Note it disregards the word order in the text."""
  target = tf.one_hot(y, NUMBER_OF_CATEGORIES, 1, 0)
  word_vectors = learn.ops.categorical_variable(x, n_classes=n_words,
      embedding_size=EMBEDDING_SIZE, name='words')
  features = tf.reduce_max(word_vectors, reduction_indices=1)
  prediction, loss = learn.models.logistic_regression(features, target)
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(),
      optimizer='Adam', learning_rate=0.01)
  return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op

MODELS ={
    'bagOfWords' : { 'trainingPath' : '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/data/MPTTraining.csv',
                     'testingPath' : '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/data/SMALL_MPTTesting.csv',
                     'smalltrainingPath': '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/data/SMALL_MPTTraining.csv',
                     'modelPath' : '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/models/modelBagOfWords_big',
                     'model': bag_of_words_model,
                     'name': 'Bag of Words based on w2vec',
                     'labels' : '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/data/ProductTypeToCodeMap.txt',
                     'vocabProcessorPath' : '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/models/vocabprocessor'},
}


def pretty_json(json_file):
    return json.dumps(json_file, sort_keys=True, indent=4)

def build_url(image_id, IMAGE_SERVICE="http://raymcompreviewprod", CPRI="214x261.jpg"):
    url = '/'
    iid = int(float(image_id))
    for i in reversed(range(1, 9)):
        url = '%s%d' % (url, int((iid % 10 ** i) / 10 ** (i - 1)))
        if i % 2 != 0:
            url = '%s/' % url
    url = '%s%sfinal/%s-%s' % (IMAGE_SERVICE, url, iid, CPRI)
    return url

def loadLabels(filename):
    with gfile.Open(filename) as csv_file:
        labelsFile = csv.reader(csv_file)
        return dict([(int(raw[0]), raw[1]) for raw in labelsFile])


def getCursorOverData():
    with open('/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/models/manProductTYpeQuery.sql','r') as myfile:
        sqlQuery = myfile.read().replace('\n', ' ')
        db = cx_Oracle.connect("macys", "macys", "dml1-scan.federated.fds:1521/dpmstg01")
        cursor = db.cursor()
        cursor.execute(sqlQuery)
        return cursor

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):

    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'output_labels.txt')
    self.node_lookup = self.read_labels(uid_lookup_path) #self.load(label_lookup_path, uid_lookup_path)


  def read_labels(self, label_lookup_path):

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(label_lookup_path).readlines()
    id_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for i,line in enumerate(proto_as_ascii_lines):
      id_to_human[i] = line[:-1]

    return id_to_human

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.
    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.
    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def run_inference_on_image(image):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()



  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    softmax_tensor = sess.graph.get_tensor_by_name('final_tensor_name:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)



    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    result =[]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]

      point = collections.namedtuple('Point', ['humanString', 'score'])
      point.humanString = human_string
      point.score = score
      result.append(point)
    return result

def down_or_load_image(url, path, filename):
    try:
        image_on_web = urllib.urlopen(url)
        if image_on_web.code != 200:
            return False
        buf = image_on_web.read()
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = '%s/%s' % (path, filename)
        downloaded_image = open(file_path, "wb")
        downloaded_image.write(buf)
        downloaded_image.close()
        image_on_web.close()
    except IOError as e:
        print(e)
        return False
    except AttributeError as e:
        print(e, sys.version)
        return False
    return file_path


def getReadableOutput(labels, scores, predictedClass ):
    result = list ([(labels[i], round(scores[i],2)) for i in range(1,len(scores))])
    return sorted(result, key=lambda (k,v) : (v,k), reverse=True)

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.

  # with tf.gfile.FastGFile(os.path.join(
  #     FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'output_graph.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def print_response_status(response):
    print("Request status:\n url: %s\n" % response.url,
          "request: %s\n" % str(response.request),
          "status code: %s\n" % str(response.status_code),
          "reason: %s\n" % str(response.reason),
          "json: %s\n"% str(response.json()))


def get_model_values(size):
    return {v.humanString.lower(): float(format(v.score, '.2f'))
            for i, v in enumerate(total_image_model_prediction) if i < size}

def getDistanceForParticularTypeText(textResult, product_type_original, modelDesigion):
    d = dict(textResult)
    original_type_score = d.get(product_type_original)
    if (original_type_score is None):
        original_type_score = 0.
    modelScore = dict(textResult)[modelDesigion]
    return math.sqrt(math.pow(original_type_score - modelScore, 2))


def getDistanceForParticularTypeImage(model_score_by_values, original_type):
    d = dict(model_score_by_values)
    original_type_score = d.get(original_type.lower())
    if (original_type_score is None):
        original_type_score = 0.
    model_value, model_score = model_score_by_values[0]
    if original_type_score is None:
        return model_score
    else:
        return math.sqrt(math.pow(original_type_score - model_score, 2))


# report_json = {
#     # this is fake attribute type
#     "attributeId": 1111,
#     "systemOfRecord": 'Facet',
#     "attributeName": 'Man product type',
#     "isPublished": False,
#     "isRepeatable": False,
#     "tenant": 'MCOM',
#     "attributeProductType": "",
#     "attributeDescription": "",
#     "isProcessed": False
# }

# reports_response = requests.post("http://mdc2vr6212:8080/api/reports/", data=pretty_json(report_json), headers=headers)
report_id = '583f8de809c19208d5f574b5'#str(reports_response.json()['_id'])

print("report_id = %s" % report_id)

products_put_url = '%s%s' % ('http://mdc2vr6212:8080', '/api/reports/%s/products/' % report_id)
current_report_url = '%s%s%s' % ('http://mdc2vr6212:8080', '/api/reports/', report_id)

global n_words
currectModelDescr=MODELS['bagOfWords']
X_train, target = loadData(currectModelDescr['smalltrainingPath'])
labels = loadLabels(currectModelDescr['labels'])

    ### Process vocabulary
vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH).restore(currectModelDescr['vocabProcessorPath'])
X_train = np.array(list(vocab_processor.transform(X_train)))
n_words = len(vocab_processor.vocabulary_)


classifier = learn.Estimator(model_fn=currectModelDescr['model'],model_dir=currectModelDescr['modelPath'])

    ## todo ugly way to restore model!!! no need to pass training set. need to pass empty
classifier.fit(X_train, target , steps=0)



cursor= getCursorOverData()
create_graph()

hierarchy_dict = []


def buildLabelScoreMap(labels, param):
    result = {}
    for i,item in enumerate(param):
        if labels.get(i) is not None:
            result[labels[i]] = param[i]
    return result



def calculatedistance(model_score_by_values, textResult, product_type_original, modelDesigion):
    imageDistance = getDistanceForParticularTypeImage(model_score_by_values, product_type_original)
    textDistance =  getDistanceForParticularTypeText(textResult, product_type_original,modelDesigion)
    sqrt = imageDistance + textDistance
    return float(round(sqrt,2))



for row in cursor:
    imageId = int(row[1])
    url = build_url(imageId)
    image = down_or_load_image(url, "images", imageId)
    total_image_model_prediction = run_inference_on_image(image)

    model_score_by_values = sorted(get_model_values(20).items(), key=operator.itemgetter(1),
                                   reverse=True)
    imageResult = total_image_model_prediction[0]
    product_type_original = str(row[2])
    if product_type_original == "T-SHIRT":
        product_type_original = "T_SHIRT"
    # np.array(data, dtype=np.str)
    textForClassification = np.array(list(vocab_processor.transform(np.array([str(row[9])], dtype=np.str))))
    textResult = [{ "label": labels[p['class']], "score":p['prob'][p['class']] , "map": buildLabelScoreMap(labels,p['prob'])  } for p in classifier.predict(textForClassification,  as_iterable=True)][0]


    if (imageResult.humanString == textResult["label"] and imageResult.humanString.lower() != product_type_original.lower()):
        print ("textScore %s, imageScore %s "%(textResult['score'],imageResult.score ))

        distance = calculatedistance(model_score_by_values,textResult['map'],product_type_original, textResult["label"])
        division = int(row[3])
        product_id = int(row[0])

        product_json = {
            "productId": int(row[0]),
            "productDescription": str(row[7].encode('utf-8')),
            "processedItems": [
                {

                    "imageUrl": url,
                    "originalAttributeNormalizedValue": [product_type_original],
                    "attributeSourceValue": "",
                    "attributeVerificationWarningLevel": 3,
                    "additionalInfo": {
                    },
                    "algorithm": "ML+texbase",
                    "division": 1,

                    "department": 123,
                    "buyer_approval": 'Approved',
                    "merch_approval": 'Approved',
                    "live": True,
                    "purchasable": True,
                    "attributeVerificationResult": [
                        {
                            "score": 0.88,
                            "value": textResult["label"]
                        }
                    ],
                    "distance": distance,
                    "availableValues": [
                        "novalues"
                    ],
                }
            ]
        }
        product_put_url = '%s%s' % (products_put_url, str(product_id))
        print pretty_json(product_json)
        product_put_resp = requests.put(product_put_url, data=json.dumps(product_json),headers=headers)


print(pretty_json(hierarchy_dict))

statistic_json = {
    "statistic": {
        "correct": {
            "amount": 10,
            "percent": 1
        },
        "incorrect": {
            "amount": 10,
            "percent": 1
        },
        "unassigned": {
            "amount": 10,
            "percent": 1
        },
        "suspicious": {
            "amount": 10,
            "percent": 1
        }
    },

    "hierarchy": [{"id":1, "departments" : [{"id":123}]}],
    "isProcessed": True,
}

update_report_resp = requests.put(current_report_url, data=pretty_json(statistic_json), headers=headers)

print_response_status(update_report_resp)










