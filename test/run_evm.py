from ptcr import EventModel


with open("../samples/oulu/event_model.json", "r") as f:
    e = EventModel.load_model(f.read())
    for _ in range(50):
        current_state, event = e.step()
        print(f"Current state: {current_state}, event: {event}")
