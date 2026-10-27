"""Verbatim source labels and a PROJECT display palette, never severity targets."""

# IDs are assigned by this project, not published dataset class IDs. RGB values
# are display choices, not a verified reconstruction of publisher mask colors.
CLASS_NAMES = (
    "Background",
    "Microcrack",
    "Structural crack",
    "Minor spalling",
    "Moderate spalling",
    "Delamination",
    "Rebar corrosion",
    "Concrete crushing",
    "Structural deformation",
)
LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES) if index}
DISPLAY_COLORS = (
    (128, 128, 128), (230, 40, 40), (40, 180, 60),
    (40, 210, 210), (130, 70, 170), (245, 140, 30),
    (240, 215, 50), (50, 90, 220), (230, 60, 200),
)
DATASET_URL = "https://data.mendeley.com/datasets/yfhwfcrfmk/1"
SOURCE_NOTE = (
    "Source labels verified by JSON archive inspection. Microcrack and Delamination "
    "differ from the landing-page terminology; no semantic aliases or severity "
    "labels are inferred. IDs and RGB display colors are project-defined."
)
