import datetime
import pandas as pd
import time
import matplotlib.pyplot as plt

#sets up the pages that are going to be used
gf = pd.read_excel('accomplishment_2.xlsx', 
                   usecols=['Date', 'Day', 'Shift', 'Total Boxes Kitted', 
                            'Total Pounds Gleaned', 
                            'Total Pounds of Reclamation', 
                            'Backpack program'])
gf_csv = pd.read_csv('Golden.csv', 
                     usecols=['Opportunity Name', 'Date','Hours', 'Time'])

gf_csv['Datetime'] = gf_csv['Date'] + " " + gf_csv['Time']
#gf_csv.to_excel("golden.xlsx", index = False)

#sets up node and linkedList class
class Node(object):
    def __init__(self, data = None):
        self.next = None
        data["# of volunteer"] = 0
        data["# of hours"] = 0
        data["# of mesh bags"] = 0
        data["Total Weight"] = 0
        data["Total kits and backpacks"] = 0
        self.data = data
        self.volunteer = 0
        self.hours = 0
        self.mesh = 0.0
        second_iter = iter(data.items())
        for j in data:
            key, value = next(second_iter)
            if key == "Date":
                self.datetime = value.date()
            if key == "Day":
                self.day = value.lower()[:3]
            if key == "Shift":
                self.shift = value
            if key == "Total Boxes Kitted":
                self.box = value
                data["Total kits and backpacks"] = value
            if key == "Total Pounds Gleaned":
                self.glean = value
                self.mesh += float(value/4)
                data["Total Weight"] += value
                data["# of mesh bags"] = self.mesh
            if key == "Total Pounds of Reclamation":
                self.reclaim = value
                data["Total Weight"] += value
            if key == "Backpack program":
                self.backpack = value
                self.mesh += value
                data["# of mesh bags"] = self.mesh
                data["Total kits and backpacks"] = value
class LinkedList(object):
    def __init__(self, head = None):
        self.head = head
        self.count = 0

    def insert(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        else:
            new_node.next = self.head
            self.head = new_node

    def updateNode(self, date, hour):
        current_node = self.head
        shift = ""
        time1 = datetime.time(8,30,0)
        time2 = datetime.time(12,30,0)
        time3 = datetime.time(16,30,0)
        time4 = datetime.time(20,30,0)
        if(time1 <= date.time() < time2):
            shift = "Morning"
        elif(time2 <= date.time() < time3):
            shift = "Afternoon"
        elif(time3 <= date.time() < time4):
            shift = "Evening"
        

        while(current_node):
            if(current_node.datetime == date.date()):
                if(time2 <= date.time() < time3):
                    current_node = current_node.next
                elif(time3 <= date.time() < time4):
                    current_node = current_node.next
                    current_node = current_node.next
                current_node.volunteer += 1
                current_node.hours += hour
                data_dict = current_node.data
                data_dict["# of volunteer"] += 1
                data_dict["# of hours"] += hour
                current_node.data = data_dict
                break
            current_node = current_node.next
    
    def data(self):
        current_node = self.head
        big_data = {}

        while(current_node):
            second_iter = iter(current_node.data.items())
            #checks if the area is empty
            if not bool(big_data):
                for j in current_node.data:
                    key, value = next(second_iter)
                    big_data[key] = [value]
            else:
                for j in current_node.data:
                    key, value = next(second_iter)
                    big_data[key].append(value)    
            current_node = current_node.next
        return big_data

#sets up dicitonary for golden data
golden_dict = gf_csv.to_dict()
time_dict = golden_dict["Datetime"]
hour_dict = golden_dict["Hours"]

#sets up dictionary for accomplishment data
accomplishment_dict = gf.to_dict()

#set up combined
first = LinkedList()

#fills up dicitonary with everything
for j in range(1, len(gf)):
    fod_dictionary = {}
    for key in accomplishment_dict.keys():
        value = accomplishment_dict[key][j]
        if key == "Date":
            i = 1
            while pd.isna(value) or type(value) != datetime.datetime:
                value = accomplishment_dict[key][j - i]
                i = i + 1
        elif key == "Shift" or key == "Day":
            i = 1
            while pd.isna(value):
                value = accomplishment_dict[key][j - i]
                i = i + 1
        else:
            if pd.isna(value):
                value = 0
            else:
                if type(value) != str:
                    value = int(value)
                else:
                    value = 0
        fod_dictionary[key] = value
    first.insert(fod_dictionary)

#updates volunteers with time
for i in range(len(gf_csv)):
    value = hour_dict[i]
    date = datetime.datetime.strptime(time_dict[i], '%m/%d/%Y %I:%M %p')
    if pd.isna(value):
        value = 0
    first.updateNode(date, value)

#turns it into an excel sheet
ref = pd.DataFrame(first.data())
#specifies datapoint
ref = ref.loc[171:753]
ref['Average weight'] = ref['Total Weight']/ref["# of hours"]
ref['Average kit/backpack'] = ref['Total kits and backpacks']/ref["# of hours"]
ref.to_excel('FullSheet.xlsx', index = False)

new = ref[(ref['Date'].dt.year >= 2023)]
boxes = new[(new['Total Boxes Kitted'] <= 2200)]


boxes.plot(kind = 'scatter', x = '# of hours', y = ["Total Boxes Kitted"])
plt.savefig("boxestrial.png")
new.plot(kind = 'scatter', x = '# of hours', y = ["Total Pounds Gleaned"])
plt.savefig("gleantrial.png")
new.plot(kind = 'scatter', x = '# of hours', y = ["Total Pounds of Reclamation"])
plt.savefig("reclaimtrial.png")
new.plot(kind = 'scatter', x = '# of hours', y = ["Backpack program"])
plt.savefig("backpacktrial.png")

#new.plot(x = 'Date', y = ["Total Boxes Kitted", "Total Pounds Gleaned", 'Total Pounds of Reclamation', 'Backpack program'])


#plt.show()
#

#of mesh bags per shift(for gleaned and backpack) 4 cent per 4 pounds
