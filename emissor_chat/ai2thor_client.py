from ai2thor.controller import Controller
from leolani_client import Action
import numpy as np
from PIL import Image

ACTIONS = ["find", "describe", "move", "go", "turn", "forward", "back", "left", "right", "open", "close", "look"]


class Ai2ThorClient:

    def __init__(self):
        """
        returns: None
        """
        self._answers =[]
        self._actions=[]
        self._perceptions=[]
        self._controller = Controller()
        #self._controller.renderInstanceSegmentation = True
        self._controller.renderObjectImage = True
        self._controller.agentMode = "arm"
        self._event = None

    def getdistance(self, coord1, coord2):
        distance = np.sqrt((coord2['x'] - coord1['x'])**2
                        + (coord2['y'] - coord1['y'])**2
                        + (coord2['z'] - coord1['z'])**2)
        return distance

    def search_for_object_in_view_near(self, objectType):
        found = []
        for object1 in self._event.metadata['objects']:
            if object1['objectType'].lower()==objectType.lower():
                print('object1', object1)
                coord1 = object1['position']
                closest = 100
                closest_object = None
                for object2 in self._event.metadata['objects']:
                    if not object1['name']==object2['name']:
                        coord2 = object2['position']
                        distance = self.getdistance(coord1, coord2)
                        if distance>0 and distance<closest:
                            closest = distance
                            closest_object= object2
                found.append((object1, closest, closest_object))
        return found

    def search_for_object_in_view(self, objectType):
        found = []
        for obj in self._event.metadata['objects']:
            if obj['objectType'].lower()==objectType.lower():
                #coord = self._event.instance_detections2D.get(obj['name'])
                #print(coord)
                coord = obj['position']
                image = Image.fromarray(self._controller.last_event.frame)
                found.append((obj, objectType, coord, image))
        return found
        
    def search_for_object(self, objectType):
        answer = ""
        found = self.search_for_object_in_view(objectType)
        rotate =0
        while not found and rotate<4:
            self._event = self._controller.step(Action.RotateRight.name)
            found = self.search_for_object_in_view(objectType)
            rotate += 1
        if not found:
            answer = "I could not find it. Tell me to move?"
        else:
            answer = "I found %s instances of type %s in my view" % (len(found), objectType) 
            for f,objectType, coord, _ in found:
                answer += "\n"+f['name'] +" at " + str(coord)
                # affordances = self.get_true_properties(f)
                # answer += "These are its properties:"
                # for affordance in affordances:
                #     print(affordance)
        return answer, found

    def what_do_you_see(self, ):
        answer =  "I see %s things there.\n" % (len(self._event.metadata['objects']))
        for obj in self._event.metadata['objects']:
            answer += obj['objectType']+"\n"
            if obj['moveable']:
                answer += "\tI can move it.\n"
            if obj['openable']:
                answer += "\tI can open it.\n"
            if obj['breakable']:
                answer +="\tI can break it.\n"
        return answer

    def get_true_properties(self, object):
        affordances = []
        for key in object.items():
            if key[1]==True:
                affordances.append(key[0])
        return affordances


    def what_i_can_do(self):
        answer =  "I can do the following:", str(ACTIONS)
        return answer


    def do_action(self, w1, w2):
        answer = ""
        found_objects = []
        if w1.lower()=="find":
            answer, found_objects = self.search_for_object(w2)
            self._actions.append(Action.Look)

        elif w1.lower()=="describe":
            answer = self.what_do_you_see()
            
        elif w1.lower()=="look":
            if w2.lower()=="up":
                self._event =self._controller.step(Action.LookUp.name)
                self._actions.append(Action.LookUp)
            elif w2.lower()=="down":
                self._event = self._controller.step(Action.LookDown.name)
                self._actions.append(Action.LookDown)
                
        elif w1.lower()=="move" or w1.lower()=="go" or w1.lower()=="turn":
            if w2.lower()=="forward":
                self._event =self._controller.step(Action.MoveAhead.name)
                self._actions.append(Action.MoveAhead)
            elif w2.lower()=="back":
                self._event = self._controller.step(Action.MoveBack.name)
                self._actions.append(Action.MoveAhead)
            elif w2.lower()=="left":
                self._event = self._controller.step(Action.RotateLeft.name)
                self._actions.append(Action.RotateLeft)
            elif w2.lower()=="right":
                self._event = self._controller.step(Action.RotateRight.name)
                self._actions.append(Action.RotateRight)

        return answer, found_objects

    def process_instruction(self, prompt):
        self._answers =[]
        self._actions = []
        self._perceptions = []
        answer = ""
        words = prompt.split()
        if words[0].lower() in ACTIONS:
            self._event = self._controller.step(Action.MoveAhead.name)
            answer, found_objects = self.do_action(words[0].lower(), words[-1].lower())
            if answer:
                self._answers.append(answer)
            if found_objects:
                self._perceptions.extend(found_objects)
        else:
            answer = "Sorry I do not get that:"+words[0]
            self._answers.append(answer)
