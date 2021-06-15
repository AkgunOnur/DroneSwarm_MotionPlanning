import os
import json
import numpy as np
import shutil

class Json_Editor():
    def __init__(self, nagent):
        self.nagent = nagent

    def reset_json(self, filename='sample.json'):
        with open(filename,'r+') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
            file_data['Vehicles'] = {}
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(file_data, file, indent = 4)
            file.truncate()     # remove remaining part
            file.close()
                
    def modify(self):
        self.reset_json()
        if os.path.exists('./sample.json'):
            with open('./sample.json', 'r+') as file:

                #First we load existing data into a dict.
                file_data = json.load(file)
                drone = {}
                for n in range(self.nagent):
                    drone['Drone'+str(n+1)] = {"VehicleType": "SimpleFlight", "X": np.random.randint(-20,20), "Y": np.random.randint(-20,20), "Z": np.random.randint(-6,0)}

                # Join new_data with file_data
                file_data['Vehicles'].update(drone)

                #file_data["Vehicles"]["Drone1"].update(drone)

                # Sets file's current position at offset.
                # convert back to json.
                file.seek(0)
                json.dump(file_data, file, indent = 4)
                file.close()
        os.popen('cp ./sample.json ~/Documents/AirSim/settings.json') 




# if __name__ == "__main__":
#     nagent = 3
#     js_modifier = Airsim_Json(nagent)
#     js_modifier.modify()
#     