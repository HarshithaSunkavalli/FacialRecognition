import numpy as np
import cv2
import os
import fnmatch
from pymongo import MongoClient
import random
import pickle
import sys

client = MongoClient("mongodb://152.46.18.168:27017")
db = client.alda_facenet
train_set = db.train_files



def update_db():
    i = 1
    image_dictionary = {}
    for root, dir, files in os.walk(os.path.join(".", "Datasets")):
        for file1 in fnmatch.filter(files, "*"):
            if(os.path.isdir(file1) == False):
                name = os.path.basename(os.path.normpath(root))
                number = file1.split('_')[-1].split('.')[0]
                record = {
                    "name" : os.path.basename(os.path.normpath(root)),
                    "number" : int(number),
                    "file" : file1
                }
                train_set.insert_one(record)

class DataReader:
    def __init__(self, batch_size = 100, batches = 3):
        self.dispatched = 0
        self.selected = []
        self.batch_size = batch_size
        self.batches = batches
        total = batch_size * batches
        records = train_set.aggregate([
            { "$group": {
                "_id": "$name",
                "count": { "$sum": 1 }
            }},
            { "$match": {
                "count": { "$gt": 1 }
            }}
        ])

        # count = train_set.count()
        self.non_distinct_records = []
        for record in records:
            self.non_distinct_records.append(record)


        self.final_positions = []
        count = 0
        while (count + len(self.non_distinct_records)) < total :
            for record in self.non_distinct_records:
                self.final_positions.append(record)
            count += len(self.non_distinct_records)
        
        total -= count
        count = 0
        for record in self.non_distinct_records:
            count += 1
            if(count <= total):
                self.final_positions.append(record)
            else:
                s = int(random.random() * count)
                if s < total:
                    self.final_positions[s] = record


        # print("Length => " + str(len(self.final_positions)))
        # print(str(self.final_positions))

    def getData(self):
        
        if self.dispatched > self.batch_size * self.batches:
            return "Cant retrieve data"
        i = 0
        data = []
        anchor_list = []
        positive_list = []
        negative_list = []

        while i < self.batch_size:

            base_set = train_set.aggregate([
                {"$match": {"name":self.final_positions[self.dispatched]['_id']}}, 
                {"$sample": {"size": 2}}
            ]);    
            # print("base set : ")
            key = "anchor"
            name = None
            record = {}

            anchor_flag = True
            for value in base_set:
                file2 = os.path.join(".","Datasets","lfw", value['name'], value['file'])
                # print(file)
                image = cv2.imread(file2)
                # print(image.shape)
                result = cv2.resize(image, (220,220), interpolation = cv2.INTER_AREA)
                # print(result.shape)

                if anchor_flag == True:
                    anchor_flag = False
                    anchor_list.append(result)
                    record['anchor_name'] = value['file']
                else:
                    positive_list.append(result)
                    
            found = False
            while found == False:
                negative_set = train_set.aggregate([
                    {"$sample": {"size": 1}}
                ])        
                for value in negative_set:
                    if value['name'] != self.final_positions[self.dispatched]['_id']:
                        duplicate = False
                        for x in self.selected:
                            if x['anchor'] == record['anchor_name'] and x['negative'] == value['file']:
                                duplicate = True
                                break
                        
                        if duplicate == False:
                            self.selected.append({"anchor": record['anchor_name'], "negative": value['file']})
                            file3 = os.path.join(".","Datasets","lfw", value['name'], value['file'])
                            # print(file)
                            image = cv2.imread(file3)
                            # print(image.shape)
                            result = cv2.resize(image, (220,220), interpolation = cv2.INTER_AREA)
                            # print(result.shape)
                            negative_list.append(result)
                            found = True
                            
            self.dispatched += 1
            i += 1

        data.append(anchor_list)
        data.append(positive_list)
        data.append(negative_list)
        return data

# d = DataReader(150,100)
#
# for i in range(0, 100):
#     data = d.getData()
#     print(i)