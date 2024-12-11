import sys
import numpy as np
from PIL import Image
import base64
import io
import copy
import os
import random
import csv


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

    Parameters:
        image_array (numpy.ndarray): The image array to convert.
        image_format (str): The desired image format (e.g., "PNG", "JPEG").

    Returns:
        str: Base64-encoded string of the image.
    """
    # Ensure the array is C-contiguous
    image_array = np.ascontiguousarray(image_array)

    # Save to in-memory buffer
    buffer = io.BytesIO()
    Image.fromarray(image_array).save(buffer, format=image_format)

    # Encode to Base64 and return as string
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def setup(env="colab"):
    from dotenv import load_dotenv

    if env == "colab":
        os.system("pip install --upgrade ai2thor --quiet")
        os.system("pip install ai2thor-colab prior --upgrade &> /dev/null")
        os.system("pip install python-dotenv")
        os.system("pip install cltl.combot --break-system-packages")

        os.system("apt-get install xvfb")
        import ai2thor_colab

        ai2thor_colab.start_xserver()
        # Load the .env file
        load_dotenv()
        # OpenAI API Key
        api_key = os.getenv("OPENAI_API_KEY")

    if env == "local":
        # Get the current working directory
        current_dir = os.getcwd()
        # Construct the relative path to the .env file
        env_path = os.path.join(current_dir, "../.env")
        # Load the .env file
        load_dotenv(env_path)

        # OpenAI API Key
        api_key = os.getenv("OPENAI_API_KEY")

    else:
        raise ValueError("Invalid environment. Use 'colab' or 'local'.")

    return api_key


def load_dataset(house: int = 11):
    import prior

    dataset = prior.load_dataset("procthor-10k")
    house = dataset["train"][house]  # CHOOSE HOUSE
    return house


def get_top_down_frame():
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]
    return Image.fromarray(top_down_frame)


from openaiapi import analyze_image, analyze_prompt

import math


def teleport_in_front_of_object(
    controller, object_position, reachable_positions, visited_positions, distance=1.0
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
    agent_position = controller.last_event.metadata["agent"]["position"]
    visited_positions.append(agent_position)

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


import re


def interactive_object_match(
    api_key: str,
    human_object_description: str,
    unique_object_list: list,
    HUMAN: str,
    AGENT: str,
    leolaniClient,
):
    """
    Interactively matches a human description of an object to one from a given list using an LLM.
    If a single object is suggested, the user is asked for a yes/no confirmation.
    If multiple objects are suggested, the user is asked to select one of the suggested options.
    Case-insensitive matching is used when the user selects from multiple matches.
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
        llm_response = analyze_prompt(api_key=api_key, prompt=prompt)

        # Extract the response content if needed
        if isinstance(llm_response, tuple):
            llm_response = llm_response[0]
        if isinstance(llm_response, list) and llm_response:  # Non-empty list
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
        # Query the LLM for a guess
        response = ask_llm(current_description, unique_object_list)
        leolaniClient._add_utterance(AGENT, response)
        print(f"{AGENT}>{response}")

        # Extract the matched object(s) from the response
        matched_objects = re.findall(
            r"\b(" + "|".join(map(re.escape, unique_object_list)) + r")\b", response
        )

        if matched_objects:
            if len(matched_objects) == 1:
                # Only one object suggested
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
                # Multiple objects suggested
                objects_str = " or ".join(matched_objects)
                selection_prompt = f"I've found multiple possible matches: {objects_str}. Which one best matches your object?"
                leolaniClient._add_utterance(AGENT, selection_prompt)
                print(f"{AGENT}>{selection_prompt}")
                user_input = input().strip()
                leolaniClient._add_utterance(HUMAN, user_input)
                print(f"{HUMAN}>{user_input}")

                # Compare in a case-insensitive manner
                user_input_lower = user_input.lower()
                matched_objects_lower = [obj.lower() for obj in matched_objects]

                if user_input_lower in matched_objects_lower:
                    # Find the original matched object that corresponds to the user's selection
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
                    # User didn't pick one of the offered objects
                    refine_message = "Let's refine the search. Can you provide more details or clarify the description?"
                    leolaniClient._add_utterance(AGENT, refine_message)
                    print(f"{AGENT}>{refine_message}")
                    clarifying_question = input().strip()
                    leolaniClient._add_utterance(HUMAN, clarifying_question)
                    print(f"{HUMAN}>{clarifying_question}")
                    current_description = clarifying_question
        else:
            # No matched objects found
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
    """
    Finds the positions of all objects of a specific type by rotating the agent.

    Args:
      controller: The AI2Thor controller.
      object_type: The type of object to find (e.g., "Painting", "Chair", "Table").
      num_rotations: The number of times to rotate the agent to scan for objects.

    Returns:
      A list of positions for the specified object type.
    """
    all_object_positions = []  # To store all object positions

    for _ in range(num_rotations):  # Rotate specified number of times
        # Get visible objects and their positions
        visible_objects = [
            obj for obj in controller.last_event.metadata["objects"] if obj["visible"]
        ]
        objects_of_interest = [
            obj for obj in visible_objects if obj["objectType"] == object_type
        ]
        current_object_positions = []
        for obj in objects_of_interest:
            print(obj["name"], obj["position"])  # Optional: Print object details
            current_object_positions.append(obj["position"])

        # Add current object positions to the overall list
        all_object_positions.extend(current_object_positions)

        # Rotate the agent
        controller.step("RotateRight")

    return all_object_positions


def teleport_to_pos(pos, visited_positions, controller):
    """
    Teleports the agent to a given position and updates the visited_positions list.

    Args:
        pos (dict): The position to teleport to.
    """
    print(f"Teleporting to position: {pos}")
    # rotation = random.choice(range(0, 360, 90))  # Optional: choose a rotation
    event = controller.step(
        action="Teleport", position=pos, rotation={"x": 0, "y": 0, "z": 0}
    )
    agent_position = controller.last_event.metadata["agent"]["position"]
    visited_positions.append(agent_position)


def euclidean_distance_2d(pos1, pos2):
    """
    Calculate the Euclidean distance between two positions in 2D space (x and z axes).

    Args:
        pos1 (dict): The first position with 'x' and 'z' coordinates.
        pos2 (dict): The second position with 'x' and 'z' coordinates.

    Returns:
        float: The Euclidean distance between the two positions.
    """
    return math.sqrt((pos1["x"] - pos2["x"]) ** 2 + (pos1["z"] - pos2["z"]) ** 2)


def get_farthest_position(reachable_positions, visited_positions):
    """
    Find the reachable position that is farthest from any of the visited positions.

    Args:
        reachable_positions (list): A list of reachable positions (each position is a dict with 'x', 'y', 'z').
        visited_positions (list): A list of positions already visited.

    Returns:
        dict: The position in reachable_positions that is farthest from visited_positions.
    """
    max_min_distance = -1
    farthest_position = None
    for position in reachable_positions:
        # Compute distances to all visited positions
        distances = [
            euclidean_distance_2d(position, visited_pos)
            for visited_pos in visited_positions
        ]
        # Find the minimum distance to the visited positions
        min_distance = min(distances)
        # Update if this is the farthest minimum distance found so far
        if min_distance > max_min_distance:
            max_min_distance = min_distance
            farthest_position = position
    return farthest_position


import matplotlib.pyplot as plt


def plot_trajectory(reachable_positions, visited_positions, farthest_position):
    # Extract x and z coordinates
    visited_x = [pos["x"] for pos in visited_positions]
    visited_z = [pos["z"] for pos in visited_positions]
    reachable_x = [pos["x"] for pos in reachable_positions]
    reachable_z = [pos["z"] for pos in reachable_positions]

    # Plot the positions
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
    max_rotations=3,
):
    """
    Searches for the matched_object in the environment and asks the user for confirmation.
    If the object is not found, it teleports to the farthest away position and continues the search.

    Args:
        controller: The AI2Thor controller instance.
        matched_object (str): The type of object to search for (e.g., "Painting").
        reachable_positions (list): List of reachable positions in the environment.
        api_key (str): API key for the OpenAI API.
        AGENT (str): Name of the agent (e.g., "Ai2Thor").
        HUMAN (str): Name of the human user.
        leolaniClient: Instance of LeolaniChatClient for communication.
        visited_positions (list): List of positions the agent has visited.
        max_rotations (int): Number of rotations to perform when searching for objects.

    Returns:
        bool: True if the object was found and confirmed by the user, False otherwise.
    """

    # Keep track of positions where the object has been searched for
    searched_positions = []

    while True:
        # Find all positions of the matched object in the current view
        object_positions = find_all_object_positions(
            controller, matched_object, num_rotations=max_rotations
        )

        # If no objects are found in the current location
        if not object_positions:
            # Check if all reachable positions have been visited
            if len(visited_positions) == len(reachable_positions):
                print(
                    f"{AGENT}>I have searched all locations but couldn't find any {matched_object}."
                )
                leolaniClient._add_utterance(
                    AGENT,
                    f"I have searched all locations but couldn't find any {matched_object}.",
                )
                return False

            # Teleport to the farthest unvisited position
            farthest_position = get_farthest_position(
                reachable_positions, visited_positions
            )
            if farthest_position is None:
                # All positions have been visited
                print(
                    f"{AGENT}>I have searched all locations but couldn't find any {matched_object}."
                )
                leolaniClient._add_utterance(
                    AGENT,
                    f"I have searched all locations but couldn't find any {matched_object}.",
                )
                return False
            else:
                # **Plot the trajectory before teleporting**
                plot_trajectory(
                    reachable_positions, visited_positions, farthest_position
                )
                print("Teleporting to a new location to continue the search.")
                teleport_to_pos(farthest_position, visited_positions, controller)
                continue  # Continue the while loop
        else:
            # Iterate over the found object positions
            for position in object_positions:
                if position in searched_positions:
                    continue  # Skip if we've already checked this object

                # FAR
                # Teleport in front of the object
                event_far = teleport_in_front_of_object(
                    controller,
                    position,
                    reachable_positions,
                    visited_positions,
                    distance=2.0,
                )

                base64_string_far = numpy_to_base64(event_far.frame)
                Image.fromarray(event_far.frame)  # image for clearity

                # Analyze the image using the OpenAI API
                description_far = analyze_image(
                    base64_string_far,
                    api_key=api_key,
                    prompt=f"Describe in detail the objects that {matched_object} is sourrounded by with spatial relations.",
                )
                utterance = description_far[0]["choices"][0]["message"]["content"]

                # Communicate with the user
                agent_message = f"{utterance} Should I get a closer look at the object sorrounded by these objects?"
                print(f"{AGENT}>{agent_message}")
                leolaniClient._add_utterance(AGENT, agent_message)

                # Get user input
                user_input = (
                    input("Type 'yes' if so, or 'no' to continue: ").strip().lower()
                )
                leolaniClient._add_utterance(HUMAN, user_input)
                print(f"{HUMAN}>{user_input}")

                if user_input == "no":
                    searched_positions.append(position)
                    continue

                if user_input == "yes":
                    print(
                        f"{AGENT}>Great! Let me have a coser look at the {matched_object}."
                    )
                    leolaniClient._add_utterance(
                        AGENT,
                        f"Great! Let me have a coser look at the {matched_object}.",
                    )
                    # return True

                    # -----------------------------------------------------------------------------------------------

                    # NORMAL
                    # Teleport in front of the object
                    event = teleport_in_front_of_object(
                        controller, position, reachable_positions, visited_positions
                    )

                    base64_string = numpy_to_base64(event.frame)
                    Image.fromarray(event.frame)  # image for clearity

                    # Analyze the image using the OpenAI API
                    description = analyze_image(
                        base64_string,
                        api_key=api_key,
                        prompt=f"Describe the {matched_object} in great detail.",
                    )
                    utterance = description[0]["choices"][0]["message"]["content"]

                    # Communicate with the user
                    agent_message = (
                        f"{utterance} Was this the item you were looking for?"
                    )
                    print(f"{AGENT}>{agent_message}")
                    leolaniClient._add_utterance(AGENT, agent_message)

                    # Get user input
                    user_input = (
                        input("Type 'yes' if so, or 'no' to continue: ").strip().lower()
                    )
                    leolaniClient._add_utterance(HUMAN, user_input)
                    print(f"{HUMAN}>{user_input}")

                    if user_input == "yes":
                        print(f"{AGENT}>Great! I've found the {matched_object}.")
                        leolaniClient._add_utterance(
                            AGENT, f"Great! I've found the {matched_object}."
                        )
                        return True

                    elif user_input == "no":
                        searched_positions.append(position)
                        continue
                    else:
                        # Handle unexpected input
                        error_message = "Please respond with 'yes' or 'no'."
                        print(f"{AGENT}>{error_message}")
                        leolaniClient._add_utterance(AGENT, error_message)

            # After checking all objects in current location, move to a new location
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
                # **Plot the trajectory before teleporting**

                print("Teleporting to a new location to continue the search.")
                farthest_position = get_farthest_position(
                    reachable_positions, visited_positions
                )
                plot_trajectory(
                    reachable_positions, visited_positions, farthest_position
                )
                if farthest_position is None:
                    # All positions have been visited
                    print(
                        f"{AGENT}>I have searched all locations but couldn't find any {matched_object}."
                    )
                    leolaniClient._add_utterance(
                        AGENT,
                        f"I have searched all locations but couldn't find any {matched_object}.",
                    )
                    return False
                teleport_to_pos(farthest_position, visited_positions, controller)
                continue  # Continue the while loop


def init_chat_client(emissor_path="./emissor", AGENT="Ai2Thor", HUMAN="Human"):
    # adding to the system path
    sys.path.insert(0, os.path.abspath("../emissor_chat"))
    from leolani_client import LeolaniChatClient

    leolaniClient = LeolaniChatClient(
        emissor_path=emissor_path, agent=AGENT, human=HUMAN
    )
    return leolaniClient


def add_utterance(WHO, utterance, leolaniClient):
    print(WHO + ">" + utterance)
    leolaniClient._add_utterance(WHO, utterance)


def random_teleport(controller):
    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]
    # Teleport somewhere random (Only happens once at the start)
    visited_positions = []
    position = random.choice(reachable_positions)
    # rotation = random.choice(range(360))
    print("Teleporting the agent to", position)
    event = controller.step(
        action="Teleport", position=position, rotation={"x": 0, "y": 0, "z": 0}
    )
    agent_position = controller.last_event.metadata["agent"]["position"]
    visited_positions.append(agent_position)
    Image.fromarray(event.frame)  # image for clearity

    return visited_positions
