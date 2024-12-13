# utils.py
import sys
import numpy as np
from PIL import Image
import base64
import io
import copy
import os
import random
import csv
import matplotlib.pyplot as plt
import math
import re
from enum import Enum, auto

from openaiapi import analyze_image, analyze_prompt


# Define the Action enum
class Action(Enum):
    MoveAhead = auto()
    MoveBack = auto()
    MoveLeft = auto()
    MoveRight = auto()
    RotateRight = auto()
    RotateLeft = auto()
    LookUp = auto()
    LookDown = auto()
    Crouch = auto()
    Stand = auto()
    Teleport = auto()
    TeleportFull = auto()
    Look = auto()
    GetReachablePositions = auto()


def load_unique_object_list(csv_file_path):
    with open(csv_file_path, mode="r") as file:
        csv_reader = csv.reader(file)
        # Skip the header
        next(csv_reader)
        unique_object_list = [row[0] for row in csv_reader if row]
    return unique_object_list


def numpy_to_base64(image_array, image_format="PNG"):
    image_array = np.ascontiguousarray(image_array)
    buffer = io.BytesIO()
    Image.fromarray(image_array).save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def setup(env="colab"):
    from dotenv import load_dotenv

    if env == "local":
        current_dir = os.getcwd()
        env_path = os.path.join(current_dir, "../.env")
        load_dotenv(env_path)
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        import ai2thor_colab

        ai2thor_colab.start_xserver()
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

    return api_key


def load_dataset(house: int = 6666):
    import prior

    dataset = prior.load_dataset("procthor-10k")
    house = dataset["train"][house]
    return house


def record_action_and_image(
    leolaniClient, action: Action, event, object_name="scene_view", object_type="scene"
):
    leolaniClient._add_action(action)
    img_pil = Image.fromarray(event.frame)
    dummy_bounds = {"x": 0, "y": 0, "z": 0}
    leolaniClient._add_image(object_name, object_type, dummy_bounds, img_pil)


def step_and_record(controller, leolaniClient, action_name, **kwargs):
    event = controller.step(action=action_name.name, **kwargs)
    record_action_and_image(leolaniClient, action_name, event)
    return event


def teleport_in_front_of_object(
    controller,
    object_position,
    reachable_positions,
    visited_positions,
    leolaniClient,
    distance=1.0,
):
    target_position = {
        "x": object_position["x"] - distance,
        "y": object_position["y"],
        "z": object_position["z"] - distance,
    }

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

    event = step_and_record(
        controller,
        leolaniClient,
        Action.Teleport,
        position=closest_position,
        rotation={"x": 0, "y": 0, "z": 0},
    )
    agent_position = controller.last_event.metadata["agent"]["position"]
    visited_positions.append(agent_position)

    dx = object_position["x"] - closest_position["x"]
    dz = object_position["z"] - closest_position["z"]
    rotation = math.degrees(math.atan2(dx, dz))

    event = step_and_record(
        controller,
        leolaniClient,
        Action.Teleport,
        position=closest_position,
        rotation=rotation,
    )

    agent_position = controller.last_event.metadata["agent"]["position"]
    visited_positions.append(agent_position)

    return event


def get_object_positions(controller, matched_object):
    visible_objects = [
        obj for obj in controller.last_event.metadata["objects"] if obj["visible"]
    ]
    objects_of_interest = [
        obj for obj in visible_objects if obj["objectType"] == matched_object
    ]
    object_positions = [obj["position"] for obj in objects_of_interest]
    return object_positions


def safe_input(prompt):
    """Prompt until user enters something or 'bye'."""
    while True:
        inp = input(prompt).strip()
        if inp == "":
            print("Please type something or 'bye'.")
            continue
        return inp


def user_input_with_bye(prompt, leolaniClient, HUMAN, AGENT, scenario_saver):
    """Get user input and if 'bye', stop and save scenario."""
    while True:
        inp = input(prompt).strip().lower()
        if inp == "":
            print("Please type something or 'bye'.")
            continue
        leolaniClient._add_utterance(HUMAN, inp)
        print(f"{HUMAN}>{inp}")
        if inp == "bye":
            leolaniClient._add_utterance(AGENT, "Understood, stopping now. Goodbye!")
            print(f"{AGENT}>Understood, stopping now. Goodbye!")
            scenario_saver()
            sys.exit(0)  # Exit program immediately
        return inp


def add_image_from_event(leolaniClient, event, object_name, object_type):
    img_pil = Image.fromarray(event.frame)
    dummy_bounds = {"x": 0, "y": 0, "z": 0}
    leolaniClient._add_image(object_name, object_type, dummy_bounds, img_pil)


def interactive_object_match(
    api_key: str,
    human_object_description: str,
    unique_object_list: list,
    HUMAN: str,
    AGENT: str,
    leolaniClient,
):
    def ask_llm(description: str, objects: list) -> str:
        object_list_str = ", ".join(objects)
        prompt = (
            f"Imagine you are tasked with identifying an object from a given list based on its description. "
            f"The list of objects is: {object_list_str}. "
            f"Match the description:\n'{description}'"
        )
        llm_response = analyze_prompt(api_key=api_key, prompt=prompt)
        if isinstance(llm_response, tuple):
            llm_response = llm_response[0]
        if isinstance(llm_response, list) and llm_response:
            llm_response = llm_response[0]
        if (
            isinstance(llm_response, dict)
            and "choices" in llm_response
            and llm_response["choices"]
        ):
            return llm_response["choices"][0]["message"]["content"]
        return llm_response

    current_description = human_object_description

    while True:
        response = ask_llm(current_description, unique_object_list)
        leolaniClient._add_utterance(AGENT, response)
        print(f"{AGENT}>{response}")
        matched_objects = re.findall(
            r"\b(" + "|".join(map(re.escape, unique_object_list)) + r")\b", response
        )

        if matched_objects:
            if len(matched_objects) == 1:
                confirmation_prompt = "Is this correct? (yes/no or bye to stop): "
                leolaniClient._add_utterance(AGENT, confirmation_prompt)
                print(f"{AGENT}>{confirmation_prompt}")
                user_input = safe_input("")
                leolaniClient._add_utterance("Human", user_input)
                print(f"Human>{user_input}")
                if user_input.lower() == "bye":
                    return matched_objects[0]
                if user_input.lower() == "yes":
                    success_message = "Great! Object successfully matched."
                    leolaniClient._add_utterance(AGENT, success_message)
                    print(f"{AGENT}>{success_message}")
                    return matched_objects[0]
                elif user_input.lower() == "no":
                    refine_message = (
                        "Let's refine the search. Provide more details or clarify."
                    )
                    leolaniClient._add_utterance(AGENT, refine_message)
                    print(f"{AGENT}>{refine_message}")
                    clarifying_question = safe_input("")
                    leolaniClient._add_utterance("Human", clarifying_question)
                    print(f"Human>{clarifying_question}")
                    if clarifying_question.lower() == "bye":
                        return matched_objects[0]
                    current_description = clarifying_question
                else:
                    error_message = "Please respond with 'yes', 'no', or 'bye'."
                    leolaniClient._add_utterance(AGENT, error_message)
                    print(f"{AGENT}>{error_message}")
            else:
                objects_str = " or ".join(matched_objects)
                selection_prompt = (
                    f"I've found multiple matches: {objects_str}. Which one?"
                )
                leolaniClient._add_utterance(AGENT, selection_prompt)
                print(f"{AGENT}>{selection_prompt}")
                user_input = safe_input("")
                leolaniClient._add_utterance("Human", user_input)
                print(f"Human>{user_input}")
                if user_input.lower() == "bye":
                    return matched_objects[0]
                user_input_lower = user_input.lower()
                matched_objects_lower = [obj.lower() for obj in matched_objects]

                if user_input_lower in matched_objects_lower:
                    selected_object = matched_objects[
                        matched_objects_lower.index(user_input_lower)
                    ]
                    success_message = f"Great! '{selected_object}' matched."
                    leolaniClient._add_utterance(AGENT, success_message)
                    print(f"{AGENT}>{success_message}")
                    return selected_object
                else:
                    refine_message = "Refine the description."
                    leolaniClient._add_utterance(AGENT, refine_message)
                    print(f"{AGENT}>{refine_message}")
                    clarifying_question = safe_input("")
                    leolaniClient._add_utterance("Human", clarifying_question)
                    print(f"Human>{clarifying_question}")
                    if clarifying_question.lower() == "bye":
                        return matched_objects[0]
                    current_description = clarifying_question
        else:
            error_message = "No matches. Provide more details?"
            leolaniClient._add_utterance(AGENT, error_message)
            print(f"{AGENT}>{error_message}")
            clarifying_question = safe_input("")
            leolaniClient._add_utterance("Human", clarifying_question)
            print(f"Human>{clarifying_question}")
            if clarifying_question.lower() == "bye":
                return unique_object_list[0]
            current_description += " " + clarifying_question


def find_all_object_positions(controller, object_type, num_rotations=3):
    all_object_positions = []
    for _ in range(num_rotations):
        visible_objects = [
            obj for obj in controller.last_event.metadata["objects"] if obj["visible"]
        ]
        objects_of_interest = [
            obj for obj in visible_objects if obj["objectType"] == object_type
        ]
        current_object_positions = [obj["position"] for obj in objects_of_interest]
        all_object_positions.extend(current_object_positions)
        controller.step("RotateRight")
    return all_object_positions


def teleport_to_pos(pos, visited_positions, controller, leolaniClient):
    event = step_and_record(
        controller,
        leolaniClient,
        Action.Teleport,
        position=pos,
        rotation={"x": 0, "y": 0, "z": 0},
    )
    agent_position = controller.last_event.metadata["agent"]["position"]
    visited_positions.append(agent_position)


def euclidean_distance_2d(pos1, pos2):
    return math.sqrt((pos1["x"] - pos2["x"]) ** 2 + (pos1["z"] - pos2["z"]) ** 2)


def get_farthest_position(reachable_positions, visited_positions):
    max_min_distance = -1
    farthest_position = None
    for position in reachable_positions:
        if not visited_positions:
            # If no visited positions, just pick a random or first
            return position
        distances = [euclidean_distance_2d(position, vp) for vp in visited_positions]
        min_distance = min(distances) if distances else 0
        if min_distance > max_min_distance:
            max_min_distance = min_distance
            farthest_position = position
    return farthest_position


def init_chat_client(emissor_path="./emissor", AGENT="Ai2Thor", HUMAN="Human"):
    sys.path.insert(0, os.path.abspath("../emissor_chat"))
    from leolani_client import LeolaniChatClient

    leolaniClient = LeolaniChatClient(
        emissor_path=emissor_path, agent=AGENT, human=HUMAN
    )
    return leolaniClient


def add_utterance(WHO, utterance, leolaniClient):
    print(f"{WHO}>{utterance}")
    leolaniClient._add_utterance(WHO, utterance)


def random_teleport(controller, leolaniClient):
    event = step_and_record(controller, leolaniClient, Action.GetReachablePositions)
    reachable_positions = event.metadata["actionReturn"]
    visited_positions = []
    position = random.choice(reachable_positions)
    event = step_and_record(
        controller,
        leolaniClient,
        Action.Teleport,
        position=position,
        rotation={"x": 0, "y": 0, "z": 0},
    )
    agent_position = controller.last_event.metadata["agent"]["position"]
    visited_positions.append(agent_position)
    return visited_positions
