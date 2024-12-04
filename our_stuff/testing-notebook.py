#!/usr/bin/env python
# Matt
import sys
import os
from PIL import Image
from dotenv import load_dotenv
from ai2thor.controller import Controller
import prior
import random
from openaiapi import analyze_prompt, analyze_image
from utils import numpy_to_base64
import math
import re
sys.path.insert(0, os.path.abspath("../emissor_chat"))
from leolani_client import LeolaniChatClient, Action

# Get the current working directory
current_dir = os.getcwd()
# Construct the relative path to the .env file
env_path = os.path.join(current_dir, "../.env")
# Load the .env file
load_dotenv(env_path)


# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")


dataset = prior.load_dataset("procthor-10k")
house = dataset["train"][11]
controller = Controller(scene=house, visibilityDistance=10, width=750, height=750)
event = controller.step(action="GetReachablePositions")
reachable_positions = event.metadata["actionReturn"]

unique_object_list = [
    "AlarmClock",
    "AluminumFoil",
    "Apple",
    "ArmChair",
    "BaseballBat",
    "BasketBall",
    "Bathtub",
    "BathtubBasin",
    "Bed",
    "Blinds",
    "Book",
    "Boots",
    "Bottle",
    "Bowl",
    "Box",
    "Bread",
    "ButterKnife",
    "Cabinet",
    "Candle",
    "CD",
    "CellPhone",
    "Chair",
    "Cloth",
    "CoffeeMachine",
    "CoffeeTable",
    "CounterTop",
    "CreditCard",
    "Cup",
    "Curtains",
    "Desk",
    "DeskLamp",
    "Desktop",
    "DiningTable",
    "DishSponge",
    "DogBed",
    "Drawer",
    "Dresser",
    "Dumbbell",
    "Egg",
    "Faucet",
    "Floor",
    "FloorLamp",
    "Footstool",
    "Fork",
    "Fridge",
    "GarbageBag",
    "GarbageCan",
    "HandTowel",
    "HandTowelHolder",
    "HousePlant",
    "Kettle",
    "KeyChain",
    "Knife",
    "Ladle",
    "Laptop",
    "LaundryHamper",
    "Lettuce",
    "LightSwitch",
    "Microwave",
    "Mirror",
    "Mug",
    "Newspaper",
    "Ottoman",
    "Painting",
    "Pan",
    "PaperTowelRoll",
    "Pen",
    "Pencil",
    "PepperShaker",
    "Pillow",
    "Plate",
    "Plunger",
    "Poster",
    "Pot",
    "Potato",
    "RemoteControl",
    "RoomDecor",
    "Safe",
    "SaltShaker",
    "ScrubBrush",
    "Shelf",
    "ShelvingUnit",
    "ShowerCurtain",
    "ShowerDoor",
    "ShowerGlass",
    "ShowerHead",
    "SideTable",
    "Sink",
    "SinkBasin",
    "SoapBar",
    "SoapBottle",
    "Sofa",
    "Spatula",
    "Spoon",
    "SprayBottle",
    "Statue",
    "Stool",
    "StoveBurner",
    "StoveKnob",
    "TableTopDecor",
    "TargetCircle",
    "TeddyBear",
    "Television",
    "TennisRacket",
    "TissueBox",
    "Toaster",
    "Toilet",
    "ToiletPaper",
    "ToiletPaperHanger",
    "Tomato",
    "Towel",
    "TowelHolder",
    "TVStand",
    "VacuumCleaner",
    "Vase",
    "Watch",
    "WateringCan",
    "Window",
    "WineBottle",
]


position = random.choice(reachable_positions)
rotation = random.choice(range(360))
print("Teleporting the agent to", position, " with rotation", rotation)

event = controller.step(action="Teleport", position=position, rotation=rotation)

Image.fromarray(event.frame)


event = controller.step(
    action="Teleport",
    position={"x": 2.5, "y": 0.9009997844696045, "z": 4.5},
    rotation=80,
)

Image.fromarray(event.frame)


frame = controller.last_event.frame
base64_string = numpy_to_base64(frame)


frame = controller.last_event.frame
base64_string = numpy_to_base64(frame)


def teleport_in_front_of_object(
    controller, object_position, reachable_positions, distance=1.0
):
    """Teleports the agent to the closest reachable position in front of an object.

    Args:
      controller: The AI2Thor controller.
      object_position: The position of the target object.
      reachable_positions: A list of reachable positions in the scene.
      distance: The desired distance in front of the object.

    Returns:
      The event after teleporting.
    """

    print(f"DEBUG object pos: {object_position}")
    # Calculate the target position in front of the object
    target_position = {
        "x": object_position["x"] - distance,
        "y": object_position["y"],
        "z": object_position["z"],
    }

    # Find the closest reachable position
    closest_position = None
    min_distance = float("inf")

    for position in reachable_positions:
        dist = math.sqrt(
            (position["x"] - target_position["x"]) ** 2
            + (position["z"] - target_position["z"]) ** 2
        )
        if dist < min_distance:
            min_distance = dist
            closest_position = position

    # Calculate rotation towards the object
    dx = object_position["x"] - closest_position["x"]
    dz = object_position["z"] - closest_position["z"]
    rotation = math.degrees(math.atan2(dx, dz))

    # Teleport and rotate
    event = controller.step(
        action="Teleport", position=closest_position, rotation=rotation
    )

    return event  # Return the event after adjusting view angle


def get_object_positions(controller, matched_object):
    """
    Finds the positions of all visible objects of a specific type.

    Args:
      controller: The AI2Thor controller.
      matched_object: The type of object to find (e.g., "Painting", "Chair", "Table").

    Returns:
      A list of positions for the specified object type.
    """
    visible_objects = [
        obj for obj in controller.last_event.metadata["objects"] if obj["visible"]
    ]
    objects_of_interest = [
        obj for obj in visible_objects if obj["objectType"] == matched_object
    ]
    object_positions = []
    for obj in objects_of_interest:
        # print(obj["name"], obj["position"])
        object_positions.append(obj["position"])
    return object_positions


visible_objects = [
    obj for obj in controller.last_event.metadata["objects"] if obj["visible"]
]
paintings = [obj for obj in visible_objects if obj["objectType"] == "Painting"]
painting_positions = []
for painting in paintings:
    print(painting["name"], painting["position"])
    painting_positions.append(painting["position"])


visible_objects = [
    obj for obj in controller.last_event.metadata["objects"] if obj["visible"]
]
paintings = [obj for obj in visible_objects if obj["objectType"] == "Painting"]
all_painting_positions = []  # To store all painting positions

for _ in range(3):  # Rotate three times
    # Get visible paintings and their positions
    current_painting_positions = []
    for painting in paintings:
        print(painting["name"], painting["position"])
        current_painting_positions.append(painting["position"])

    # Add current painting positions to the overall list
    all_painting_positions.extend(current_painting_positions)

    # Rotate the agent
    controller.step("RotateRight")

    # Update visible objects and paintings for the next iteration
    visible_objects = [
        obj for obj in controller.last_event.metadata["objects"] if obj["visible"]
    ]
    paintings = [obj for obj in visible_objects if obj["objectType"] == "Painting"]

print("All painting positions in the room:", all_painting_positions)


# Teleport to the first painting
event = controller.step(action="GetReachablePositions")
reachable_positions = event.metadata["actionReturn"]

# painting_position = get_object_positions(controller, "Painting")
# print(f"Paiting pos: {painting_position}")

# event = teleport_in_front_of_object(controller, painting_position, reachable_positions)
# Image.fromarray(event.frame)


# We could probably do this automatically, but we could also ask user to ask robot to look up or down

# But think about if this would even be necessary? its just for the gpt4o system to describe it a tiny bit better which may be redundant

# after we take look up or down we should return to the original state, THIS IS IMPORTANT OTHERWISE IT WILL MESS WITH TELEPORTING


# ## Test Interaction


emissor_path = "./emissor"
AGENT = "Ai2Thor"
HUMAN = "Human"
leolaniClient = LeolaniChatClient(emissor_path=emissor_path, agent=AGENT, human=HUMAN)


# In[14]:


utterance = f"Hi {HUMAN}. What do you see in the room?"
print(AGENT + ">" + utterance)
leolaniClient._add_utterance(AGENT, utterance)


# In[15]:


# grab img of an item to look for

# example: look for a tv in a room
# event = controller.step(action="Teleport", position={'x': 2.75, 'y': 0.9009997844696045, 'z': 1.0}  with rotation 209)
# Image.fromarray(event.frame)


# In[16]:


human_room_description = (
    "there is a table. 5 chairs. wthere is a window. its probably a living room."
)


# In[17]:


print(HUMAN + ">" + human_room_description)
leolaniClient._add_utterance(HUMAN, human_room_description)


# In[45]:


# claryfying questions
claryfying_questions_response = analyze_prompt(
    api_key=api_key,
    model="gpt-4o-mini",
    prompt=f"Imagine you are a robot who needs to be on a exact location as the point of view that the human has. After a while, the human can no longer see this image. The human will most likely describe a room from memory. The human will most likely describe a few objects and maybe some other attributes, like colours of objects. Your task is to ask claryfing questions about the room and objects so that you (the robot) has the highest chance of finding where the human was standing. Remember, ask the questions as if you were directly talking to the human. Try not to ask for too much details and dont ask for too much; remember, the human has to describe the image from memory, so only ask what you deem most important. \n Human description: {human_room_description}",
)


# In[46]:


utterance = claryfying_questions_response[0]["choices"][0]["message"]["content"]
print(AGENT + ">" + utterance)
leolaniClient._add_utterance(AGENT, utterance)


# In[27]:


human_room_description_clarified = "The table is blue, chairs are all black. The window is on the left wall in the same corner as th balcony doors."
print(HUMAN + ">" + human_room_description_clarified)
leolaniClient._add_utterance(HUMAN, human_room_description)


# In[28]:


utterance = "Describe the object I should look for."
print(AGENT + ">" + utterance)
leolaniClient._add_utterance(AGENT, utterance)


# In[29]:


human_obj_description = "It's a dark painting with trees a moon. some clouds, a river."


# In[30]:


print(HUMAN + ">" + human_obj_description)
leolaniClient._add_utterance(HUMAN, human_room_description)


# In[ ]:


# MATTS FUNCTION  --- this is where the interactive object match comes :
# based on "dark painting with trees ...", did you mean "Painting"?


# In[39]:


# Teleport somewhere random

position = random.choice(reachable_positions)
rotation = random.choice(range(360))
print("Teleporting the agent to", position, " with rotation", rotation)

event = controller.step(action="Teleport", position=position, rotation=rotation)

# Image.fromarray(event.frame) # image for clearity


# In[41]:


# location classificaiton:

# chatgpt, do you think you are in the correct room based on human_room_description + human_room description_clarified? (maybe rotate 360 degrees, but also have to analyze 4 images then)

room_classifcation = analyze_image(
    base64_string,
    api_key=api_key,
    prompt=f"Imagine you are a robot looking for a certain room. Describe the room shortly. Do you think you are in the correct room based on the following description? Description: {human_room_description}, even further description: {human_room_description_clarified}, speak to me as if you have the robots point of view. Be consise in your answer.",
)
utterance = room_classifcation[0]["choices"][0]["message"]["content"]
print(AGENT + ">" + utterance)
leolaniClient._add_utterance(AGENT, utterance)


# In[ ]:


# Decide what to do:  teleport to another room, teleport to object instance (if there is one) or lookleft/lookright etc


# In[44]:


# Metadata

# i see x instances of matched_object
# teleport to matched_object[0]
# use chatgpt to describe image
# do so until image is found or no instances left

# matched_object = "Painting"  # placeholder

# object_positions = get_object_positions(controller, matched_object)

# for i in range(len(object_positions)):
#     position = object_positions[i]
#     event = teleport_in_front_of_object(controller, position, reachable_positions)

#     description = analyze_image(
#         base64_string,
#         api_key=api_key,
#         prompt=f"describe {matched_object} in great detail",
#     )
#     utterance = description[0]["choices"][0]["message"]["content"]

#     print(AGENT + ">" + utterance + "Was this the item you were looking for?")
#     leolaniClient._add_utterance(
#         AGENT, utterance + "Was this the item you were looking for?"
#     )
#     if input("Type yes if so...") == "yes":
#         print(HUMAN + ">" + "yes")
#         leolaniClient._add_utterance(HUMAN, "yes")
#         break
#     else:
#         print(HUMAN + ">" + "no")
#         leolaniClient._add_utterance(HUMAN, "no")
#         continue

# print("Teleport to new room if item isn't found")


# In[ ]:


# if no more instances, teleport to new room


# In[47]:


# leolaniClient._save_scenario()


# ###Room description

# ### Object description

# In[ ]:


def interactive_object_match(
    api_key: str, human_object_description: str, unique_object_list: list
):
    """
    Interactively matches a human description of an object to one from a given list using an LLM.
    The function continues to refine guesses based on user confirmation or denial.

    Args:
        api_key (str): The API key for accessing the LLM.
        human_object_description (str): A description of the object to match.
        unique_object_list (list): The list of unique objects to match against.

    Returns:
        str: The confirmed object from the user.
        list: The matched object(s) from the list based on the LLM's response.
    """

    def ask_llm(description: str, objects: list) -> str:
        """Helper function to query the LLM for matching the description."""
        object_list_str = ", ".join(objects)
        prompt = (
            f"Imagine you are tasked with identifying an object from a given list based on its description. "
            f"The list of objects is: {object_list_str}. "
            f"Your task is to match the following description to one or more objects from the list: \n"
            f"'{description}'\n\n"
            "If you have a single best guess, respond with: 'To be sure, would you describe your object as {object}?'\n"
            "If you are unsure and need clarification between a few options, respond with: "
            "'To be sure, would you describe your object as {object1} or {object2}?'"
            "Only use objects from the list."
        )
        # Make sure the response is extracted correctly from the LLM
        llm_response = analyze_prompt(api_key=api_key, prompt=prompt)
        if isinstance(llm_response, dict) and "choices" in llm_response:
            return llm_response[0]["choices"][0]["message"]["content"]
        return llm_response  # Return raw response if format is unexpected

    current_description = human_object_description

    while True:
        # Query the LLM for a guess
        response = ask_llm(current_description, unique_object_list)
        print(response)

        # Extract the matched object(s) from the response
        matched_objects = re.findall(
            r"\b(" + "|".join(map(re.escape, unique_object_list)) + r")\b", response
        )

        if matched_objects:
            print(f"Matched object(s): {matched_objects}")
        else:
            print("No matched objects found in the response.")

        # Ask the user for confirmation or denial
        user_input = input("Is this correct? (yes/no): ").strip().lower()

        if user_input == "yes":
            print("Great! Object successfully matched.")
            return matched_objects[0] if len(matched_objects) == 1 else matched_objects
        elif user_input == "no":
            print("Let's refine the search.")
            # Ask for more details to refine the description
            clarifying_question = input(
                "Can you provide more details or clarify the description?: "
            ).strip()
            current_description += " " + clarifying_question
        else:
            print("Please respond with 'yes' or 'no'.")


# Example usage
human_object_description = "a leafy green item used for decoration or air purification"
unique_object_list = [
    "AlarmClock",
    "AluminumFoil",
    "Apple",
    "ArmChair",
    "BaseballBat",
    "BasketBall",
    "Bathtub",
    "BathtubBasin",
    "Bed",
    "Blinds",
    "Book",
    "Boots",
    "Bottle",
    "Bowl",
    "Box",
    "Bread",
    "ButterKnife",
    "Cabinet",
    "Candle",
    "CD",
    "CellPhone",
    "Chair",
    "Cloth",
    "CoffeeMachine",
    "CoffeeTable",
    "CounterTop",
    "CreditCard",
    "Cup",
    "Curtains",
    "Desk",
    "DeskLamp",
    "Desktop",
    "DiningTable",
    "DishSponge",
    "DogBed",
    "Drawer",
    "Dresser",
    "Dumbbell",
    "Egg",
    "Faucet",
    "Floor",
    "FloorLamp",
    "Footstool",
    "Fork",
    "Fridge",
    "GarbageBag",
    "GarbageCan",
    "HandTowel",
    "HandTowelHolder",
    "HousePlant",
    "Kettle",
    "KeyChain",
    "Knife",
    "Ladle",
    "Laptop",
    "LaundryHamper",
    "Lettuce",
    "LightSwitch",
    "Microwave",
    "Mirror",
    "Mug",
    "Newspaper",
    "Ottoman",
    "Painting",
    "Pan",
    "PaperTowelRoll",
    "Pen",
    "Pencil",
    "PepperShaker",
    "Pillow",
    "Plate",
    "Plunger",
    "Poster",
    "Pot",
    "Potato",
    "RemoteControl",
    "RoomDecor",
    "Safe",
    "SaltShaker",
    "ScrubBrush",
    "Shelf",
    "ShelvingUnit",
    "ShowerCurtain",
    "ShowerDoor",
    "ShowerGlass",
    "ShowerHead",
    "SideTable",
    "Sink",
    "SinkBasin",
    "SoapBar",
    "SoapBottle",
    "Sofa",
    "Spatula",
    "Spoon",
    "SprayBottle",
    "Statue",
    "Stool",
    "StoveBurner",
    "StoveKnob",
    "TableTopDecor",
    "TargetCircle",
    "TeddyBear",
    "Television",
    "TennisRacket",
    "TissueBox",
    "Toaster",
    "Toilet",
    "ToiletPaper",
    "ToiletPaperHanger",
    "Tomato",
    "Towel",
    "TowelHolder",
    "TVStand",
    "VacuumCleaner",
    "Vase",
    "Watch",
    "WateringCan",
    "Window",
    "WineBottle",
]

response = interactive_object_match(
    api_key=api_key,
    human_object_description=human_object_description,
    unique_object_list=unique_object_list,
)
print(f"Final matched object(s): {response}")


Image.fromarray(controller.last_event.frame)

frame = controller.last_event.frame
base64_string = numpy_to_base64(frame)
analyze_image(
    base64_string,
    api_key=api_key,
    prompt="What's in this image? What kind of room are you in? Describe it as if you are in that room and you can directly see it.",
)

matched_object = "Vase"
frame = controller.last_event.frame
base64_string = numpy_to_base64(frame)
analyze_image(
    base64_string, api_key=api_key, prompt=f"describe {matched_object} in great detail"
)

matched_object = "Vase"
object_positions = get_object_positions(controller, matched_object)
for i in range(len(object_positions)):
    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]
    position = object_positions[i]
    event = teleport_in_front_of_object(controller, position, reachable_positions)

    frame = controller.last_event.frame
    base64_string = numpy_to_base64(frame)
    analyze_image(
        base64_string,
        api_key=api_key,
        prompt=f"describe {matched_object} in great detail",
    )

    print("Was this the item you were looking for?")
    if input("Type yes if so...") == "yes":
        break
