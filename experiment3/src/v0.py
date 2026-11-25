def decide(observation, tools):
    scores = tools.match()
    frame = int(scores[1] > scores[0])
    heading = observation["views"][frame]["heading"]
    return frame, "right" if heading > 0 else "left", {"heading": heading}
