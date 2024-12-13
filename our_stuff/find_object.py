# find_object.py
import os
from PIL import Image
from utils import (
    setup,
    load_dataset,
    init_chat_client,
    add_utterance,
    load_unique_object_list,
    analyze_prompt,
    interactive_object_match,
    random_teleport,
    find_all_object_positions,
    get_farthest_position,
    teleport_to_pos,
    user_input_with_bye,
    euclidean_distance_2d,
    # Note: find_object_and_confirm not included as user asked about main scenario changes
)

from ai2thor.controller import Controller


def main(env="colab"):
    csv_file_path = os.path.abspath("unique_object_list.csv")
    unique_object_list = load_unique_object_list(csv_file_path)
    api_key = setup(env=env)
    house = load_dataset()
    controller = Controller(scene=house, visibilityDistance=10, width=750, height=750)
    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]

    AGENT = input("Type the agent name...")
    HUMAN = input("Type the human name...")
    leolaniClient = init_chat_client(AGENT=AGENT, HUMAN=HUMAN)

    greeting = f"Hi {HUMAN}. What do you see in the room? Try to describe the spatial relationships between objects."
    add_utterance(AGENT, greeting, leolaniClient)

    # If user types nothing or presses enter, safe_input or user_input_with_bye would handle it
    room_description = input("Type the room description (or 'bye' to stop): ").strip()
    if room_description.lower() == "bye":
        leolaniClient._save_scenario()
        return
    if room_description == "":
        print("Please provide some description or 'bye' to stop.")
        room_description = input("Type the room description (or 'bye') again: ").strip()
        if room_description.lower() == "bye":
            leolaniClient._save_scenario()
            return
    add_utterance(HUMAN, room_description, leolaniClient)

    # Ask clarifying questions using GPT
    claryfying_questions_response = analyze_prompt(
        api_key=api_key,
        prompt=f"Imagine you are a robot who needs to be at the location the human was describing. The human describes: {room_description}\nAsk clarifying questions.",
    )
    clarification_utterance = claryfying_questions_response[0]["choices"][0]["message"][
        "content"
    ]
    add_utterance(AGENT, clarification_utterance, leolaniClient)

    clarified_description = input(
        "Type the clarified room description (or 'bye' to stop): "
    ).strip()
    if clarified_description.lower() == "bye":
        leolaniClient._save_scenario()
        return
    if clarified_description == "":
        print("Please provide some clarification or type 'bye' to stop.")
        clarified_description = input(
            "Type the clarified room description again: "
        ).strip()
        if clarified_description.lower() == "bye":
            leolaniClient._save_scenario()
            return
    add_utterance(HUMAN, clarified_description, leolaniClient)

    human_room_descriptions = [room_description, clarified_description]

    ask_object = "Describe the object I should look for."
    add_utterance(AGENT, ask_object, leolaniClient)

    object_description = input(
        "Type the object description (or 'bye' to stop): "
    ).strip()
    if object_description.lower() == "bye":
        leolaniClient._save_scenario()
        return
    if object_description == "":
        print("Please provide the object description or 'bye' to stop.")
        object_description = input("Type the object description again: ").strip()
        if object_description.lower() == "bye":
            leolaniClient._save_scenario()
            return
    add_utterance(HUMAN, object_description, leolaniClient)

    matched_object = interactive_object_match(
        api_key=api_key,
        human_object_description=object_description,
        unique_object_list=unique_object_list,
        HUMAN=HUMAN,
        AGENT=AGENT,
        leolaniClient=leolaniClient,
    )

    visited_positions = random_teleport(controller, leolaniClient)

    # Here you would call find_object_and_confirm or your updated logic
    # For brevity, not shown. Just a placeholder:
    add_utterance(
        AGENT,
        f"Now I would search for {matched_object}, but that logic isn't shown here.",
        leolaniClient,
    )

    # Save scenario at the end
    leolaniClient._save_scenario()


if __name__ == "__main__":
    main(env="local")
