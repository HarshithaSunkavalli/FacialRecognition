import numpy as np
import cv2
import os
import fnmatch
from pymongo import MongoClient
import random

client = MongoClient("mongodb://152.46.19.17:27017")

db = client.alda_facenet
train_set = db.train_set_220

def update_db():
    i = 1
    for root, dir, files in os.walk(os.path.join("..", "Datasets")):
        for file in fnmatch.filter(files, "*"):
            # print(os.path.join(root, file) + " -> " + str(os.path.isdir(file)))
            if(os.path.isdir(file) == False):
                image = cv2.imread(os.path.join(root, file))
                result = cv2.resize(image, (220,220), interpolation = cv2.INTER_AREA)
                name = os.path.basename(os.path.normpath(root))
                # print(os.path.join(root, file) + "is in " + os.path.basename(os.path.normpath(root)))
                name = name.replace("_", " ")
                image = image.flatten().tolist()
                # print(name)
                record = {
                    "image" : image,
                    "name" : name
                }
                train_set.insert_one(record)
                print(str(i) + " : " + name)
                i = i + 1
    
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

        # print(self.final_positions)
        print("Length => " + str(len(self.final_positions)))


    def getData(self):
        
        # print(record)
        if self.dispatched > self.batch_size * self.batches:
            return "Cant retrieve data"
        i = 0
        data = []
        while i < self.batch_size:
            
            # for record in self.final_positions:
            base_set = train_set.aggregate([
                {"$match": {"name":self.final_positions[self.dispatched]['_id']}}, 
                {"$sample": {"size": 2}}
            ]);    
            # print("base set : ")
            key = "anchor"
            name = None
            record = {}
            for value in base_set:
                record[key] = value['image']
                key = "positive"
                name = value['name']
            record['anchor_name'] = name

            found = False
            while found == False:
                negative_set = train_set.aggregate([
                    {"$sample": {"size": 1}}
                ])        
                for value in negative_set:
                    if value['name'] != self.final_positions[self.dispatched]['_id']:
                        duplicate = False
                        for x in self.selected:
                            if x['anchor'] == record['anchor_name'] and x['negative'] == value['name']:
                                duplicate = True
                                break
                        
                        if duplicate == False:
                            self.selected.append({"anchor": record['anchor_name'], "negative": value['name']})
                            record['negative'] = value['image']
                            record['negative_name'] = value['name']
                            found = True
                            
                                
                # # print(negative_set)
                # print("Name of negative set : " + negative_set['name'])
                # if negative_set['name'] != self.final_positions[self.dispatched]['_id']:
                #     record['negative'] = negative_set['image']
                #     record['negative_name'] = negative_set['name']
                #     found = True
                    

            # print(record)
            data.append(record)
            self.dispatched += 1
            i += 1
        return data

    def printBatchDetails(self):
        print(str(self.batches) + " : " + str(self.batch_size))

    def getTrainingData(self):
        if(self.batches == 0):
            return None
        else:
            self.batches = self.batches - 1


            base_set = train_set.aggregate([
                {"$match": {"name":"Harsha"}}, 
                {"$sample": {"size": 2}}
            ]);

            return "Hello"
    
d = DataReader(3,3)
print("Data->")
print(d.getData())
print("<-End")
# d.printBatchDetails()
# print(d.getTrainingData())
