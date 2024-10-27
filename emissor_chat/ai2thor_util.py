from ai2thor.controller import Controller
from leolani_client import Action
import numpy as np

ACTIONS = ["find", "describe", "move", "go", "forward", "back", "left", "right", "open", "close"]


class Ai2ThorClient:

    def __init__(self, controller: Controller):
        """
        returns: None
        """
        self._answers =[]
        self._actions=[]
        self._perceptions=[]
        self._controller = controller

    def getdistance(self, coord1, coord2):
        distance = np.sqrt((coord2['x'] - coord1['x'])**2
                        + (coord2['y'] - coord1['y'])**2
                        + (coord2['z'] - coord1['z'])**2)
        return distance

    def search_for_object_in_view(self, objectType, event):
        found = []
        for object1 in event.metadata['objects']:
            if object1['objectType'].lower()==objectType.lower():
                coord1 = object1['position']
                closest = 100
                closest_object = None
                for object2 in event.metadata['objects']:
                    if not object1['name']==object2['name']:
                        coord2 = object2['position']
                        distance = self.getdistance(coord1, coord2)
                        if distance>0 and distance<closest:
                            closest = distance
                            closest_object= object2
                found.append((object1, closest, closest_object))
        return found

    def search_for_object(self, objectType, event):
        answer = ""
        found = self.search_for_object_in_view(objectType, event)
        rotate =0
        while not found and rotate<4:
            event = self._controller.step(action.R)
            found = self.search_for_object_in_view(objectType, event)
            rotate += 1
        if not found:
            answer += "I could not find it. Shall I move?"
        else:
            answer +"I found ", len(found), " of those."
            for f in found:
                answer += "\n"+(f[0]['objectType'], f[1], ' pixels away from ', f[2]['objectType'])
                affordances = self.get_true_properties(f[0])
                answer += "These are its properties:", affordances
        return answer, found

    def what_do_you_see(self, event):
        answer =  "I see ", len(self, event.metadata['objects']), " things here."
        for object in event.metadata['objects']:
            answer += object['objectType']
            if object['moveable']:
                answer += "\tI can move it"
            if object['openable']:
                answer += "\tI can open it"
            if object['breakable']:
                answer +="\tI can break it"
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


    def do_action(self, event, w1, w2):
        answer = ""
        event = None
        found_objects = []
        try:
            if event:
                if w1.lower()=="find":
                    answer, found_objects = self.search_for_object(w2, event)
                elif w1.lower()=="describe":
                    answer = self.what_do_you_see(event)
                elif w1.lower()=="move" or w1.lower()=="go":
                    if w2.lower()=="forward":
                        event =self._controller.step(Action.MoveAhead.name)
                        self._actions.append(Action.MoveAhead)
                    elif w2.lower()=="back":
                        event = self._controller.step(Action.MoveBack.name)
                        self._actions.append(Action.MoveAhead)
                    elif w2.lower()=="left":
                        event = self._controller.step(Action.RotateLeft.name)
                        self._actions.append(Action.RotateLeft)
                    elif w2.lower()=="right":
                        event = self._controller.step(Action.RotateRight.name)
                        self._actions.append(Action.RotateRight)
        except:
            answer = "Could not move."

        if not event:
            answer = "No event!"
        return answer, found_objects

    def process_instruction(self, prompt):
        self._answers =[]
        self._actions = []
        self._perceptions = []
        answer = ""
        words = prompt.split()
        if words[0].lower() in ACTIONS:
            event = self._controller.step(Action.MoveAhead.name)
            answer, found_objects = self.do_action(event, words[0].lower(), words[-1].lower())
            self._answers.append(answer)
            self._perceptions.extend(found_objects)
        else:
            answer = "Sorry I do not get that:"+words[0]
            self._answers.append(answer)
