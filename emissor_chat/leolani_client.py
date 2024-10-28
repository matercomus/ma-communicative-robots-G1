import abc
import uuid
import numpy as np
from enum import Enum, auto
import time
import cltl.combot
from camera import Bounds, CameraResolution, Object, Image
from emissor.persistence import ScenarioStorage
from emissor.persistence.persistence import ScenarioController
from emissor.representation.container import MultiIndex
from emissor.representation.scenario import Modality, Signal, Scenario, ScenarioContext, Annotation, Mention
from emissor.representation.scenario import TextSignal, ImageSignal
from cltl.combot.event.emissor import AnnotationEvent, TextSignalEvent, ImageSignalEvent
from cltl.combot.infra.time_util import timestamp_now


class Action(Enum):
    MoveAhead = auto()
    MoveBack = auto()
    MoveLeft = auto()
    MoverRight = auto()
    RotateRight =auto()
    RotateLeft = auto()
    LookUp =auto()
    LookDown =auto()
    Crouch = auto()
    Stand = auto()
    Teleport = auto()
    TeleportFull = auto()
    Look = auto()

class LeolaniChatClient():

    def __init__(self, emissor_path:str, agent="Leolani", human="Alice"):
        """ Creates a scenario and adds signals
        params: emissor_path location on disk to store the scenarios
        returns: None
        """
        signals = {
            Modality.TEXT.name.lower(): "./text.json",
            Modality.IMAGE.name.lower(): "./image.json",
        }
        self._agent=agent
        self._human=human
        self._scenario_storage = ScenarioStorage(emissor_path)
        self._scenario_context =  ScenarioContext(agent)
        scenario_start = timestamp_now()
        self._scenario =self._scenario_storage.create_scenario(str(uuid.uuid4()), scenario_start, None, self._scenario_context, signals) 

    def _add_utterance(self, speaker_name, utterance):
        signal = TextSignal.for_scenario(self._scenario, timestamp_now(), timestamp_now(), None, utterance)
        TextSignalEvent.add_agent_annotation(signal, speaker_name)
        self._scenario.append_signal(signal)

    def _add_action(self, action:Action):
        signal = TextSignal.for_scenario(self._scenario, timestamp_now(), timestamp_now(), None, action.name)
        TextSignalEvent.add_agent_annotation(signal, "ACTION")
        self._scenario.append_signal(signal)

    def _add_image(self, objectType, bounds):
        # TODO SYSTEM_VIEW is the angle of the camera, this is for Leolain
        # TODO resolution of the camera needs to be chosen
        SYSTEM_VIEW = Bounds(-0.55, -0.41 + np.pi / 2, 0.55, 0.41 + np.pi / 2)
        resolution = CameraResolution.VGA

        # TODO image as numpy array is needed, depth array if available, file path
        file_path = ""
        image_array = np.zeros((resolution.height, resolution.width, 3), dtype=np.uint8)
        depth_array = np.zeros((resolution.height, resolution.width), dtype=np.uint8)
        image = Image(image_array, SYSTEM_VIEW.to_diagonal(), depth_array)
        signal = ImageSignal.for_scenario(self._scenario.id, timestamp_now(), timestamp_now(), file_path, image.bounds.to_diagonal())

        # TODO If annotation is needed: create annotation with type name, annotation data, annotation source name
        # Bounds() takes x_0, x_1, y_0, y_1 as arguments, to_diagonal converts it to x_0, y0, x_1, y_1 
        segment = MultiIndex(signal.ruler.container_id, Bounds(0, bounds['x'], bounds['y'],bounds['z']).to_diagonal())
        annotation_data = {}
        annotation_person = Annotation(objectType, annotation_data, "Ai2Thor", int(time.time()))
        mention = Mention(str(uuid.uuid4()), [segment], [annotation_person])
        signal.mentions.append(mention)
        self._scenario.append_signal(signal)

    def _save_scenario(self):
        self._scenario_storage.save_scenario(self._scenario)


if __name__ == "__main__":
    emissor_path = "./emissor"
    human = "Alice"
    agent="Leolani"
    leolaniClient = LeolaniChatClient(emissor_path=emissor_path, agent=agent, human=human)
    utterance = "Hello world"
    leolaniClient._add_utterance(agent, utterance)
    utterance = "Hello agent"
    leolaniClient._add_utterance(human, utterance)
    leolaniClient._save_scenario()    
