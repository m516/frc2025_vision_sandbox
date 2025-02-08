# frc2025_vision_sandbox
Reef detectors with OpenCV

## Goal: 
get `detect_reef` to be a standalone script that:
- [X] uses April Tags to detect the position and orientation of reefs and important game elements
- [ ] adaptive reef/coral color search based on the white value from april tag.
- [ ] search the area around the predicted positions of the game elements
- [ ] stick it all on a Limelight

## Scripts:
### detect_reef_position
checks april tag position for the location of a reef post
  * includes sliders to find the coordinates of interesting features in April Tag space

### detect_reef
currently a rewrite of detect_reef_position
  * includes 3 reef posts instead of just one
  * includes common positions for algae
  * ISSUE: top reef post isn't in the right place because that post is squiggly. The coral sits further up.
  * TODO: simulation with coral game elements
