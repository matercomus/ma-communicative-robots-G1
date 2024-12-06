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


def main():
    csv_file_path = os.path.abspath("unique_object_list.csv")
    unique_object_list = load_unique_object_list(csv_file_path)
    api_key = setup(env="local")
    house = load_dataset()
    controller = Controller(scene=house, visibilityDistance=10, width=750, height=750)
    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]

    AGENT = input("Type the agent name...")
    HUMAN = input("Type the human name...")
    leolaniClient = init_chat_client(AGENT, HUMAN)

    add_utterance(AGENT, f"Hi {HUMAN}. What do you see in the room?", leolaniClient)

    human_room_description = input("Type the room description...")
    # human_room_description = (
    #     "there is a table. 5 chairs. there is a window. its probably a living room."
    # )
    add_utterance(HUMAN, human_room_description, leolaniClient)

    # claryfying questions
    claryfying_questions_response = analyze_prompt(
        api_key=api_key,
        model="gpt-4o-mini",
        prompt=f"Imagine you are a robot who needs to be on a exact location as the point of view that the human has. After a while, the human can no longer see this image. The human will most likely describe a room from memory. The human will most likely describe a few objects and maybe some other attributes, like colours of objects. Your task is to ask claryfing questions about the room and objects so that you (the robot) has the highest chance of finding where the human was standing. Remember, ask the questions as if you were directly talking to the human. Try not to ask for too much details and dont ask for too much; remember, the human has to describe the image from memory, so only ask what you deem most important. \n Human description: {human_room_description}",
    )
    utterance = claryfying_questions_response[0]["choices"][0]["message"]["content"]
    add_utterance(AGENT, utterance, leolaniClient)

    # human_room_description_clarified = "The table is blue, chairs are all black. The window is on the left wall in the same corner as th balcony doors."
    human_room_description_clarified = input("Type the clarified room description...")
    add_utterance(HUMAN, human_room_description_clarified, leolaniClient)

    utterance = "Describe the object I should look for."
    add_utterance(AGENT, utterance, leolaniClient)

    # human_obj_description = (
    #     "It's a dark painting with trees a moon. some clouds, a river."
    # )
    human_obj_description = input("Type the object description...")
    add_utterance(HUMAN, human_obj_description, leolaniClient)

    # this is where the interactive object match comes :
    # based on "dark painting with trees ...", did you mean "Painting"?

    matched_object = interactive_object_match(
        api_key=api_key,
        human_object_description=human_obj_description,
        unique_object_list=unique_object_list,
        HUMAN=HUMAN,
        AGENT=AGENT,
        leolaniClient=leolaniClient,
    )

    visited_positions = random_teleport(controller)

    found = find_object_and_confirm(
        controller=controller,
        matched_object=matched_object,
        reachable_positions=reachable_positions,
        api_key=api_key,
        AGENT=AGENT,
        HUMAN=HUMAN,
        leolaniClient=leolaniClient,
        visited_positions=visited_positions,
    )

    if found:
        print("Successfully found the object.")
    else:
        print("Object not found after searching all locations.")

    leolaniClient._save_scenario()


if __name__ == "__main__":
    main()
