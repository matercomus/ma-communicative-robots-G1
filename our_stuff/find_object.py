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

    AGENT = input("Type the agent name...")
    HUMAN = input("Type the human name...")
    leolaniClient = init_chat_client(AGENT=AGENT, HUMAN=HUMAN)

    add_utterance(
        AGENT,
        f"""Hi {HUMAN}. What do you see in the room? Try to
                  describe the spatial relationships between objects.""",
        leolaniClient,
    )

    human_room_description = input("Type the room description...")
    add_utterance(HUMAN, human_room_description, leolaniClient)

    # Ask clarifying questions using GPT
    claryfying_questions_response = analyze_prompt(
        api_key=api_key,
        model="gpt-4o-mini",
        prompt=f"Imagine you are a robot who needs to be on an exact location as the point of view that the human has. After a while, the human can no longer see this image. The human will most likely describe a room from memory. The human will most likely describe a few objects and maybe some other attributes, like colours of objects. Your task is to ask clarifying questions about the room and objects so that you (the robot) has the highest chance of finding where the human was standing. Remember, ask the questions as if you were directly talking to the human. Try not to ask for too much detail and don't ask for too much; remember, the human has to describe the image from memory, so only ask what you deem most important. \nHuman description: {human_room_description}",
    )
    utterance = claryfying_questions_response[0]["choices"][0]["message"]["content"]
    add_utterance(AGENT, utterance, leolaniClient)

    human_room_description_clarified = input("Type the clarified room description...")
    add_utterance(HUMAN, human_room_description_clarified, leolaniClient)

    human_room_descriptions = [human_room_description, human_room_description_clarified]

    utterance = "Describe the object I should look for."
    add_utterance(AGENT, utterance, leolaniClient)

    human_obj_description = input("Type the object description...")
    add_utterance(HUMAN, human_obj_description, leolaniClient)

    matched_object = interactive_object_match(
        api_key=api_key,
        human_object_description=human_obj_description,
        unique_object_list=unique_object_list,
        HUMAN=HUMAN,
        AGENT=AGENT,
        leolaniClient=leolaniClient,
    )

    # Pass both controller and leolaniClient now
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
    )

    if found:
        print("Successfully found the object.")
    else:
        print("Object not found after searching all locations.")

    leolaniClient._save_scenario()


if __name__ == "__main__":
    main(env="local")
