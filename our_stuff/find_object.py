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
    find_object_and_confirm,
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

    AGENT = input("Type the agent name...").strip()
    HUMAN = input("Type the human name...").strip()
    leolaniClient = init_chat_client(AGENT=AGENT, HUMAN=HUMAN)

    greeting = f"Hi {HUMAN}. What do you see in the room? Try to describe the spatial relationships between objects."
    add_utterance(AGENT, greeting, leolaniClient)

    # Ensure user provides a non-empty description
    while True:
        room_description = input(
            "Type the room description (or 'bye' to stop): "
        ).strip()
        if room_description.lower() == "bye":
            leolaniClient._save_scenario()
            print(f"{AGENT}>Understood, stopping now. Goodbye!")
            return
        if room_description == "":
            print("Please provide a description or type 'bye' to stop.")
            continue
        break
    add_utterance(HUMAN, room_description, leolaniClient)

    # Ask clarifying questions using GPT
    clarification_prompt = (
        f"Imagine you are a robot who needs to be at the location the human was describing. "
        f"The human describes: {room_description}\n"
        f"Ask clarifying questions to better understand the room layout and object placements."
    )
    claryfying_questions_response = analyze_prompt(
        api_key=api_key,
        prompt=clarification_prompt,
    )
    clarification_utterance = claryfying_questions_response[0]["choices"][0]["message"][
        "content"
    ]
    add_utterance(AGENT, clarification_utterance, leolaniClient)

    # Ensure user provides a non-empty clarified description
    while True:
        clarified_description = input(
            "Type the clarified room description (or 'bye' to stop): "
        ).strip()
        if clarified_description.lower() == "bye":
            leolaniClient._save_scenario()
            print(f"{AGENT}>Understood, stopping now. Goodbye!")
            return
        if clarified_description == "":
            print("Please provide a clarification or type 'bye' to stop.")
            continue
        break
    add_utterance(HUMAN, clarified_description, leolaniClient)

    human_room_descriptions = [room_description, clarified_description]

    ask_object = "Describe the object I should look for."
    add_utterance(AGENT, ask_object, leolaniClient)

    # Ensure user provides a non-empty object description
    while True:
        object_description = input(
            "Type the object description (or 'bye' to stop): "
        ).strip()
        if object_description.lower() == "bye":
            leolaniClient._save_scenario()
            print(f"{AGENT}>Understood, stopping now. Goodbye!")
            return
        if object_description == "":
            print("Please provide the object description or type 'bye' to stop.")
            continue
        break
    add_utterance(HUMAN, object_description, leolaniClient)

    matched_object = interactive_object_match(
        api_key=api_key,
        human_object_description=object_description,
        unique_object_list=unique_object_list,
        HUMAN=HUMAN,
        AGENT=AGENT,
        leolaniClient=leolaniClient,
    )

    # Start searching for the matched object
    visited_positions = random_teleport(controller, leolaniClient)

    found = find_object_and_confirm(
        controller=controller,
        matched_object=matched_object,
        reachable_positions=reachable_positions,
        api_key=api_key,
        AGENT=AGENT,
        HUMAN=HUMAN,
        leolaniClient=leolaniClient,
        visited_positions=visited_positions,
        human_room_descriptions=human_room_descriptions,
        unique_object_list=unique_object_list,
    )

    if found:
        add_utterance(AGENT, "Successfully found the object.", leolaniClient)
        print(f"{AGENT}>Successfully found the object.")
    else:
        add_utterance(
            AGENT, "Object not found after searching all locations.", leolaniClient
        )
        print(f"{AGENT}>Object not found after searching all locations.")

    leolaniClient._save_scenario()


if __name__ == "__main__":
    main(env="local")
