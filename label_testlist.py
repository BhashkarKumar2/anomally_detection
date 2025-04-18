import os

# Mapping class names to IDs
class_mapping = {
    "ApplyEyeMakeup": 1,
    "ApplyLipstick": 2,
    "Archery": 3,
    "BabyCrawling": 4,
    "BalanceBeam": 5,
    "BandMarching": 6,
    "BaseballPitch": 7,
    "Basketball": 8,
    "BasketballDunk": 9,
    "BenchPress": 10,
    "Biking": 11,
    "Billiards": 12,
    "BlowDryHair": 13,
    "BlowingCandles": 14,
    "BodyWeightSquats": 15,
    "Bowling": 16,
    "BoxingPunchingBag": 17,
    "BoxingSpeedBag": 18,
    "BreastStroke": 19,
    "BrushingTeeth": 20,
    "CleanAndJerk": 21,
    "CliffDiving": 22,
    "CricketBowling": 23,
    "CricketShot": 24,
    "CuttingInKitchen": 25,
    "Diving": 26,
    "Drumming": 27,
    "Fencing": 28,
    "FieldHockeyPenalty": 29,
    "FloorGymnastics": 30,
    "FrisbeeCatch": 31,
    "FrontCrawl": 32,
    "GolfSwing": 33,
    "Haircut": 34,
    "Hammering": 35,
    "HammerThrow": 36,
    "HandstandPushups": 37,
    "HandstandWalking": 38,
    "HeadMassage": 39,
    "HighJump": 40,
    "HorseRace": 41,
    "HorseRiding": 42,
    "HulaHoop": 43,
    "IceDancing": 44,
    "JavelinThrow": 45,
    "JugglingBalls": 46,
    "JumpingJack": 47,
    "JumpRope": 48,
    "Kayaking": 49,
    "Knitting": 50,
    "LongJump": 51,
    "Lunges": 52,
    "MilitaryParade": 53,
    "Mixing": 54,
    "MoppingFloor": 55,
    "Nunchucks": 56,
    "ParallelBars": 57,
    "PizzaTossing": 58,
    "PlayingCello": 59,
    "PlayingDaf": 60,
    "PlayingDhol": 61,
    "PlayingFlute": 62,
    "PlayingGuitar": 63,
    "PlayingPiano": 64,
    "PlayingSitar": 65,
    "PlayingTabla": 66,
    "PlayingViolin": 67,
    "PoleVault": 68,
    "PommelHorse": 69,
    "PullUps": 70,
    "Punch": 71,
    "PushUps": 72,
    "Rafting": 73,
    "RockClimbingIndoor": 74,
    "RopeClimbing": 75,
    "Rowing": 76,
    "SalsaSpin": 77,
    "ShavingBeard": 78,
    "Shotput": 79,
    "SkateBoarding": 80,
    "Skiing": 81,
    "Skijet": 82,
    "SkyDiving": 83,
    "SoccerJuggling": 84,
    "SoccerPenalty": 85,
    "StillRings": 86,
    "SumoWrestling": 87,
    "Surfing": 88,
    "Swing": 89,
    "TableTennisShot": 90,
    "TaiChi": 91,
    "TennisSwing": 92,
    "ThrowDiscus": 93,
    "TrampolineJumping": 94,
    "Typing": 95,
    "UnevenBars": 96,
    "VolleyballSpiking": 97,
    "WalkingWithDog": 98,
    "WallPushups": 99,
    "WritingOnBoard": 100,
    "YoYo": 101
}

# Define class mapping



def write_labels(input_file):
    outdir=os.path.join("testlabels",input_file.split('/')[-1])
    with open(input_file, "r") as infile, open(outdir, "w") as outfile:
        for line in infile:
            line = line.strip()  # Remove any leading/trailing whitespace
            if line:  # Ensure the line is not empty
                # Extract class name from the file path
                class_name = line.split('/')[0]
                # Get class ID from the mapping
                class_id = class_mapping.get(class_name)
                if class_id:
                    # Write the labeled line
                    outfile.write(f"{line} {class_id}\n")
                else:
                    print(f"Warning: Class '{class_name}' not found in class_mapping.")

if __name__ == "__main__":
    dataset_dir = "UCF101TrainTestSplits-RecognitionTask/"
    video_list_file = "testlist0"
    for i in [1, 2, 3]:
        file_path = f'{dataset_dir}/{video_list_file}{i}.txt'
        write_labels(file_path)
