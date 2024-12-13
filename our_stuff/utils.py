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


def load_unique_object_list(csv_file_path):
    with open(csv_file_path, mode="r") as file:
        csv_reader = csv.reader(file)
        # Skip the header
        next(csv_reader)
        # Read the rest of the rows into a list
        unique_object_list = [row[0] for row in csv_reader if row]
    return unique_object_list


def numpy_to_base64(image_array, image_format="PNG"):
    """
    Convert a numpy array to a Base64-encoded string.
    """
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


def get_top_down_frame(controller):
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]
    return Image.fromarray(top_down_frame)


def record_action_and_image(
    leolaniClient, action: Action, event, object_name="scene_view", object_type="scene"
):
    # Record the action
    leolaniClient._add_action(action)

    # Record the image
    img_pil = Image.fromarray(event.frame)
    dummy_bounds = {"x": 0, "y": 0, "z": 0}
    leolaniClient._add_image(object_name, object_type, dummy_bounds, img_pil)


def step_and_record(controller, leolaniClient, action_name, **kwargs):
    """
    Execute controller.step(...) and record the action and resulting image.
    action_name should be one of the Action enum values.
    """
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

    # First teleport to position
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
    print(f"closest_position: {closest_position} Rotation: {rotation}")

    # Adjust rotation
    event = step_and_record(
        controller,
        leolaniClient,
        Action.Teleport,
        position=closest_position,
        rotation=rotation,
    )

    print(f"Teleporting to position: {closest_position}, after rotation {rotation}")
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
            f"Your task is to match the following description to one or more objects from the list: \n"
            f"'{description}'\n\n"
            "If you have a single best guess, respond with: 'To be sure, would you describe your object as {object}?'\n"
            "If you are unsure and need clarification between a few options, respond with: "
            "'To be sure, would you describe your object as {object1} or {object2}?'"
            "Only use objects from the list."
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
                confirmation_prompt = "Is this correct? (yes/no): "
                leolaniClient._add_utterance(AGENT, confirmation_prompt)
                print(f"{AGENT}>{confirmation_prompt}")
                user_input = input().strip().lower()
                leolaniClient._add_utterance(HUMAN, user_input)
                print(f"{HUMAN}>{user_input}")

                if user_input == "yes":
                    success_message = "Great! Object successfully matched."
                    leolaniClient._add_utterance(AGENT, success_message)
                    print(f"{AGENT}>{success_message}")
                    return matched_objects[0]
                elif user_input == "no":
                    refine_message = "Let's refine the search. Can you provide more details or clarify the description?"
                    leolaniClient._add_utterance(AGENT, refine_message)
                    print(f"{AGENT}>{refine_message}")
                    clarifying_question = input().strip()
                    leolaniClient._add_utterance(HUMAN, clarifying_question)
                    print(f"{HUMAN}>{clarifying_question}")
                    current_description = clarifying_question
                else:
                    error_message = "Please respond with 'yes' or 'no'."
                    leolaniClient._add_utterance(AGENT, error_message)
                    print(f"{AGENT}>{error_message}")

            else:
                objects_str = " or ".join(matched_objects)
                selection_prompt = f"I've found multiple possible matches: {objects_str}. Which one best matches your object?"
                leolaniClient._add_utterance(AGENT, selection_prompt)
                print(f"{AGENT}>{selection_prompt}")
                user_input = input().strip()
                leolaniClient._add_utterance(HUMAN, user_input)
                print(f"{HUMAN}>{user_input}")

                user_input_lower = user_input.lower()
                matched_objects_lower = [obj.lower() for obj in matched_objects]

                if user_input_lower in matched_objects_lower:
                    selected_object = matched_objects[
                        matched_objects_lower.index(user_input_lower)
                    ]
                    success_message = (
                        f"Great! '{selected_object}' successfully matched."
                    )
                    leolaniClient._add_utterance(AGENT, success_message)
                    print(f"{AGENT}>{success_message}")
                    return selected_object
                else:
                    refine_message = "Let's refine the search. Can you provide more details or clarify the description?"
                    leolaniClient._add_utterance(AGENT, refine_message)
                    print(f"{AGENT}>{refine_message}")
                    clarifying_question = input().strip()
                    leolaniClient._add_utterance(HUMAN, clarifying_question)
                    print(f"{HUMAN}>{clarifying_question}")
                    current_description = clarifying_question
        else:
            error_message = (
                "I couldn't find a matching object. Can you provide more details?"
            )
            leolaniClient._add_utterance(AGENT, error_message)
            print(f"{AGENT}>{error_message}")
            clarifying_question = input().strip()
            leolaniClient._add_utterance(HUMAN, clarifying_question)
            print(f"{HUMAN}>{clarifying_question}")
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
        current_object_positions = []
        for obj in objects_of_interest:
            print(obj["name"], obj["position"])
            current_object_positions.append(obj["position"])
        all_object_positions.extend(current_object_positions)
        controller.step("RotateRight")
    return all_object_positions


def teleport_to_pos(pos, visited_positions, controller, leolaniClient):
    print(f"Teleporting to position: {pos}")
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
        distances = [euclidean_distance_2d(position, vp) for vp in visited_positions]
        min_distance = min(distances)
        if min_distance > max_min_distance:
            max_min_distance = min_distance
            farthest_position = position
    return farthest_position


def plot_trajectory(reachable_positions, visited_positions, farthest_position):
    visited_x = [pos["x"] for pos in visited_positions]
    visited_z = [pos["z"] for pos in visited_positions]
    reachable_x = [pos["x"] for pos in reachable_positions]
    reachable_z = [pos["z"] for pos in reachable_positions]

    plt.figure(figsize=(10, 8))
    plt.scatter(reachable_x, reachable_z, c="blue", label="Reachable Positions")
    plt.scatter(visited_x, visited_z, c="red", label="Visited Positions")
    if farthest_position:
        plt.scatter(
            farthest_position["x"],
            farthest_position["z"],
            c="green",
            label="Farthest Position",
            s=100,
        )
    plt.xlabel("X coordinate")
    plt.ylabel("Z coordinate")
    plt.title("Positions in the Environment")
    plt.legend()
    plt.grid(True)
    plt.show()


def find_object_and_confirm(
    controller,
    matched_object,
    reachable_positions,
    api_key,
    AGENT,
    HUMAN,
    leolaniClient,
    visited_positions,
    human_room_descriptions,
    max_rotations=3,
    max_teleports=25,
):
    searched_positions = []
    teleport_count = 0

    while True:
        object_positions = find_all_object_positions(
            controller, matched_object, num_rotations=max_rotations
        )
        if not object_positions:
            if len(visited_positions) == len(reachable_positions):
                print(
                    f"{AGENT}>I have searched all locations but couldn't find any {matched_object}."
                )
                leolaniClient._add_utterance(
                    AGENT,
                    f"I have searched all locations but couldn't find any {matched_object}.",
                )
                return False
            farthest_position = get_farthest_position(
                reachable_positions, visited_positions
            )
            if farthest_position is None:
                print(
                    f"{AGENT}>I have searched all locations but couldn't find any {matched_object}."
                )
                leolaniClient._add_utterance(
                    AGENT,
                    f"I have searched all locations but couldn't find any {matched_object}.",
                )
                return False
            else:
                print("Teleporting to a new location to continue the search.")
                teleport_to_pos(
                    farthest_position, visited_positions, controller, leolaniClient
                )
                teleport_count += 1
                if teleport_count >= max_teleports:
                    print(
                        f"{AGENT}>I have reached the maximum number of teleports ({max_teleports}) but couldn't find any {matched_object}."
                    )
                    leolaniClient._add_utterance(
                        AGENT,
                        f"I have reached the maximum number of teleports ({max_teleports}) but couldn't find any {matched_object}.",
                    )
                    return False
                continue
        else:
            for position in object_positions:
                if position in searched_positions:
                    continue

                event_far = teleport_in_front_of_object(
                    controller,
                    position,
                    reachable_positions,
                    visited_positions,
                    leolaniClient,
                    distance=2.0,
                )
                base64_string_far = numpy_to_base64(event_far.frame)
                human_room_description, human_room_description_clarified = (
                    human_room_descriptions
                )
                description_far = analyze_image(
                    base64_string_far,
                    api_key=api_key,
                    prompt=f"""
                            "First, determine if the image (partly) matches the
                            descriptions and explain your reasoning. It might
                            be the same room but from a different point of
                            view.Then, describe distinguishable objects around
                            {matched_object} as bullet points. For each object,
                            include:

                            - What it is and VERY briefly what it looks like.
                            - Its position relative to {matched_object}

                            Human’s initial description:
                            {human_room_description}

                            Clarified details:
                            {human_room_description_clarified}"
                            """,
                )
                utterance = description_far[0]["choices"][0]["message"]["content"]
                agent_message = f"{utterance} Should I get a closer look at the object surrounded by these objects?"
                print(f"{AGENT}>{agent_message}")
                leolaniClient._add_utterance(AGENT, agent_message)

                while True:
                    user_input = (
                        input("Type 'yes' if so, or 'no' to continue: ").strip().lower()
                    )
                    leolaniClient._add_utterance(HUMAN, user_input)
                    print(f"{HUMAN}>{user_input}")

                    if user_input == "no":
                        searched_positions.append(position)
                        break

                    if user_input == "yes":
                        print(
                            f"{AGENT}>Great! Let me have a closer look at the {matched_object}."
                        )
                        leolaniClient._add_utterance(
                            AGENT,
                            f"Great! Let me have a closer look at the {matched_object}.",
                        )

                        event = teleport_in_front_of_object(
                            controller,
                            position,
                            reachable_positions,
                            visited_positions,
                            leolaniClient,
                        )
                        base64_string = numpy_to_base64(event.frame)

                        input("GPT close Press Enter to continue...")

                        description = analyze_image(
                            base64_string,
                            api_key=api_key,
                            prompt=f"""Describe the {matched_object}, be concise,
                            focus on the characteristics that make it easily
                            identifiable and distinguishable from similar objects
                            (shape, color).""",
                        )
                        utterance = description[0]["choices"][0]["message"]["content"]
                        agent_message = (
                            f"{utterance} Was this the item you were looking for?"
                        )
                        print(f"{AGENT}>{agent_message}")
                        leolaniClient._add_utterance(AGENT, agent_message)

                        while True:
                            user_input = (
                                input("Type 'yes' if so, or 'no' to continue: ")
                                .strip()
                                .lower()
                            )
                            leolaniClient._add_utterance(HUMAN, user_input)
                            print(f"{HUMAN}>{user_input}")

                            if user_input == "yes":
                                print(
                                    f"{AGENT}>Great! I've found the {matched_object}."
                                )
                                leolaniClient._add_utterance(
                                    AGENT, f"Great! I've found the {matched_object}."
                                )
                                return True
                            elif user_input == "no":
                                searched_positions.append(position)
                                break
                            else:
                                error_message = "Please respond with 'yes' or 'no'."
                                print(f"{AGENT}>{error_message}")
                                leolaniClient._add_utterance(AGENT, error_message)
                        break

                    else:
                        error_message = "Please respond with 'yes' or 'no'."
                        print(f"{AGENT}>{error_message}")
                        leolaniClient._add_utterance(AGENT, error_message)

            if len(visited_positions) == len(reachable_positions):
                print(
                    f"{AGENT}>I have searched all locations but couldn't find any {matched_object}."
                )
                leolaniClient._add_utterance(
                    AGENT,
                    f"I have searched all locations but couldn't find any {matched_object}.",
                )
                return False
            else:
                print("Teleporting to a new location to continue the search.")
                farthest_position = get_farthest_position(
                    reachable_positions, visited_positions
                )
                if farthest_position is None:
                    print(
                        f"{AGENT}>I have searched all locations but couldn't find any {matched_object}."
                    )
                    leolaniClient._add_utterance(
                        AGENT,
                        f"I have searched all locations but couldn't find any {matched_object}.",
                    )
                    return False
                teleport_to_pos(
                    farthest_position, visited_positions, controller, leolaniClient
                )
                teleport_count += 1
                if teleport_count >= max_teleports:
                    print(
                        f"{AGENT}>I have reached the maximum number of teleports ({max_teleports}) but couldn't find any {matched_object}."
                    )
                    leolaniClient._add_utterance(
                        AGENT,
                        f"I have reached the maximum number of teleports ({max_teleports}) but couldn't find any {matched_object}.",
                    )
                    return False
                continue


def init_chat_client(emissor_path="./emissor", AGENT="Ai2Thor", HUMAN="Human"):
    sys.path.insert(0, os.path.abspath("../emissor_chat"))
    from leolani_client import LeolaniChatClient

    leolaniClient = LeolaniChatClient(
        emissor_path=emissor_path, agent=AGENT, human=HUMAN
    )
    return leolaniClient


def add_utterance(WHO, utterance, leolaniClient):
    print(WHO + ">" + utterance)
    leolaniClient._add_utterance(WHO, utterance)


def random_teleport(controller, leolaniClient):
    event = step_and_record(controller, leolaniClient, Action.GetReachablePositions)
    reachable_positions = event.metadata["actionReturn"]
    visited_positions = []
    position = random.choice(reachable_positions)
    print("Teleporting the agent to", position)

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
