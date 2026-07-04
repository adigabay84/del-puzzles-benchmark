"""
Narrative configurations for the seven puzzle stories.

PUZZLE_CONFIGS maps each puzzle type to the phrasing dictionary used by the
text generators to render the same underlying epistemic scenario in a
different story: three classic narratives (Muddy Children, Blue-Eyed
Islanders, Wise Men) and four new variations (Olympic Games, Singing
Contest, Health Screening, Safety Inspection).

Each config supplies the agent nouns (singular/plural) and the announcing
authority, the setup introduction, the observation text for both setups
(classic_observation for the everyone-sees-everyone-else arrangement,
matrix_prefix for the random observation matrix), and the positive/negative
status vocabulary — state names, singular/plural action phrases, group
descriptions, the question posed to the queried agent, and the knew /
didn't-know phrasings used in previous-rounds summaries.

MATRIX_CORE_TEXT is the setup-agnostic explanation of the observation
matrix semantics, formatted with each config's agent and status terms.

The keys form the template contract consumed by dataset_generator,
random_obs_text_generator, and standard_observation_text_generator.
A new narrative is added by providing a complete entry here.
"""


from src.constants import MUDDY_CHILDREN_PUZZLE, BLUE_EYED_ISLANDERS_PUZZLE, WISE_MEN_PUZZLE, \
    OLYMPIC_GAMES_PUZZLE, SINGING_CONTEST_PUZZLE, HEALTH_SCREENING_PUZZLE, SAFETY_INSPECTION_PUZZLE


#prefix text for the random observation matrices
MATRIX_CORE_TEXT = (
    "Each entry (i,k) in the observation matrix indicates whether {agent} i knows the {status} of {agent} k. A value of 1 "
    "means {agent} i knows {agent} k’s {status}, while a value of 0 means they do not know."
)


PUZZLE_CONFIGS = {
    MUDDY_CHILDREN_PUZZLE: {
        "agent_role": "children",
        "agent_single": "child",
        "authority": "father",
        "setup_intro": "There are n children playing in the mud.",
        "classic_observation": "Their father calls them to the house and arranges them in a semicircle, so that each "
                               "child can clearly see every other child, but not themselves.",
        "matrix_prefix": "Their father calls them to the house and arranges them in a certain order that creates an observation matrix.",
        "status": "status",
        "pos_state": "muddy",
        "neg_state": "clean",
        "pos_action": "have muddy foreheads",
        "neg_action": "have clean foreheads",
        "pos_action_singular": "has a muddy forehead",
        "neg_action_singular": "has a clean forehead",
        "pos_prop_desc": "child with a muddy forehead",
        "pos_prop_desc_plural": "children with muddy foreheads",
        "question": "is your forehead muddy?",
        "whether_knowledge": "they know if their own forehead is muddy",
        "observed_status": "muddy/clean status",
        "the_rest_phrase": "are clean",
        "unk_txt": "did not know what was on their forehead",
        "known_txt": "knew if they had mud on their forehead",
        "pos_group": "the muddy children",
        "neg_group": "the clean children",
    },
    BLUE_EYED_ISLANDERS_PUZZLE: {
        "agent_role": "islanders",
        "agent_single": "islander",
        "authority": "foreigner",
        "setup_intro": "There are n islanders living on a remote island, where talking about eye color is forbidden. "
                       "A foreigner arrives on the island, captivated by the islanders' eye colors and eager to discuss "
                       "the topic with them.",
        "classic_observation": "He then gathers the islanders in a circle, so that each islander can clearly see every "
                               "other islander's eye color, but not their own.",
        "matrix_prefix": "He then arranges the islanders in a certain order that creates an observation matrix.",
        "status": "eye color",
        "pos_state": "blue-eyed",
        "neg_state": "brown-eyed",
        "pos_action": "have blue eyes",
        "neg_action": "have brown eyes",
        "pos_action_singular": "has blue eyes",
        "neg_action_singular": "has brown eyes",
        "pos_prop_desc": "islander with blue eyes",
        "pos_prop_desc_plural": "islanders with blue eyes",
        "question": "do you have blue eyes?",
        "whether_knowledge": "they know if their own eyes are blue",
        "observed_status": "eye colors",
        "the_rest_phrase": "have brown eyes",
        "unk_txt": "did not know if their own eyes were blue",
        "known_txt": "knew if their own eyes were blue",
        "pos_group": "the blue-eyed islanders",
        "neg_group": "the brown-eyed islanders",
    },
    WISE_MEN_PUZZLE: {
        "agent_role": "wise men",
        "agent_single": "wise man",
        "authority": "king",
        "setup_intro": "A king in a distant country searches for a new advisor, and calls n wise men to his court for a final test. "
                       "The king places a hat - either white or blue - on each wise man’s head,",
        "classic_observation": "and arranges the wise men in a circle so that each wise man can clearly see every other wise man, but not themselves.",
        "matrix_prefix": "and arranges the wise men in a certain order that creates an observation matrix.",
        "status": "hat color",
        "pos_state": "blue",
        "neg_state": "white",
        "pos_action": "are wearing blue hats",
        "neg_action": "are wearing white hats",
        "pos_action_singular": "is wearing a blue hat",
        "neg_action_singular": "is wearing a white hat",
        "pos_prop_desc": "wise man wearing a blue hat",
        "pos_prop_desc_plural": "wise men wearing blue hats",
        "question": "are you wearing a blue hat?",
        "whether_knowledge": "they know if their own hat is blue",
        "observed_status": "hat colors",
        "the_rest_phrase": "are wearing white hats",
        "unk_txt": "did not know if their own hat was blue",
        "known_txt": "knew if their own hat was blue",
        "pos_group": "the wise men wearing blue hats",
        "neg_group": "the wise men wearing white hats",
    },
    OLYMPIC_GAMES_PUZZLE: {
        "agent_role": "gymnasts",
        "agent_single": "gymnast",
        "authority": "coach",
        "setup_intro": "There are n gymnasts competing in an Olympic qualification round. At the end of the round, the coach gives each gymnast a sheet",
        "classic_observation": "listing every other gymnast’s name and status - advanced to the final or not - excluding their own status. "
                               "The gymnasts can’t share the given information with others.",
        "matrix_prefix": "listing some gymnast's name and status - advanced to the final or not - that creates an observation "
                         "matrix. The gymnasts can’t share the given information with others.",
        "status": "status",
        "pos_state": "advanced",
        "neg_state": "not advanced",
        "pos_action": "made it into the final",
        "neg_action": "did not make it into the final",
        "pos_prop_desc": "gymnast that made it into the final",
        "pos_prop_desc_plural": "gymnasts that made it into the final",
        "pos_action_singular": "made it into the final",
        "neg_action_singular": "did not make it into the final",
        "question": "did you make it into the final?",
        "whether_knowledge": "they know if they made it into the final",
        "observed_status": "status",
        "the_rest_phrase": "didn’t",
        "unk_txt": "did not know if they made it into the final",
        "known_txt": "knew if they made it into the final",
        "pos_group": "the gymnasts who made it into the final",
        "neg_group": "the gymnasts who did not make it",
    },
    SINGING_CONTEST_PUZZLE: {
        "agent_role": "singers",
        "agent_single": "singer",
        "authority": "host",
        "setup_intro": "There are n singers competing in a singing contest. Each singer receives a score from the judges, "
                       "which determines whether they are included on this year’s list of rising singing talents. "
                       "At the end of the contest, the host sticks a sticker on each of the singers' backs; X if "
                       "the singer didn’t get into the list and V if they did.",
        "classic_observation": "the host then arranges the singers so that each singer can clearly see every other "
                               "singer's sticker, but not their own.",
        "matrix_prefix": "the host then arranges the singers in a certain order that creates an observation matrix.",
        "status": "status",
        "pos_state": "V",
        "neg_state": "X",
        "pos_action": "have V stickers",
        "neg_action": "have X stickers",
        "pos_prop_desc": "singer with a V sticker",
        "pos_prop_desc_plural": "singers with V stickers",
        "pos_action_singular": "has a V sticker",
        "neg_action_singular": "has an X sticker",
        "question": "did you make it into the list of rising singing talents?",
        "whether_knowledge": "they know if they made it into the list",
        "observed_status": "stickers",
        "the_rest_phrase": "have X stickers",
        "unk_txt": "did not know if they made it into the list",
        "known_txt": "knew if they made it into the list",
        "pos_group": "the singers with V stickers",
        "neg_group": "the singers with X stickers",
    },
    HEALTH_SCREENING_PUZZLE: {
        "agent_role": "patients",
        "agent_single": "patient",
        "authority": "nurse",
        "setup_intro": "There are n patients attending a routine health screening at a clinic. After reviewing "
                       "the preliminary test results, the nurse gives each patient a sheet",
        "classic_observation": "listing every other patient's name and test result - clear or needs "
                               "follow-up - excluding their own result. The patients can’t share the given information with others.",
        "matrix_prefix": "listing some patients' names and test result - clear or needs follow-up - "
                         "that creates an observation matrix. The patients can’t share the given information with others.",
        "status": "status",
        "pos_state": "clear",
        "neg_state": "needs follow-up",
        "pos_action": "received a clear result",
        "neg_action": "need a follow-up",
        "pos_action_singular": "received a clear result",
        "neg_action_singular": "needs a follow-up",
        "pos_prop_desc": "patient who received a clear result",
        "pos_prop_desc_plural": "patients who received a clear result",
        "question": "did you receive a clear result?",
        "whether_knowledge": "they know if their own result is clear",
        "observed_status": "test results",
        "the_rest_phrase": "need a follow-up",
        "unk_txt": "did not know if their result was clear",
        "known_txt": "knew if their result was clear",
        "pos_group": "the patients who received a clear result",
        "neg_group": "the patients who need a follow-up",
    },
    SAFETY_INSPECTION_PUZZLE: {
        "agent_role": "tenants",
        "agent_single": "tenant",
        "authority": "safety warden",
        "setup_intro": "There are n tenants living in an apartment building undergoing a safety inspection. "
                        "After inspecting all units, the safety warden clips a colored lanyard around each "
                        "tenant's neck before they gather in the lobby: green if their unit passed, red if "
                        "their unit requires repairs.",
        "classic_observation": "The warden arranges the tenants in a circle in the lobby so that each tenant "
                                "can clearly see every other tenant's lanyard, but not their own.",
        "matrix_prefix": "The warden arranges the tenants in a certain order in the lobby that creates an observation matrix.",
        "status": "status",
        "pos_state": "passed",
        "neg_state": "requires repairs",
        "pos_action": "have green lanyards",
        "neg_action": "have red lanyards",
        "pos_action_singular": "has a green lanyard",
        "neg_action_singular": "has a red lanyard",
        "pos_prop_desc": "tenant with a green lanyard",
        "pos_prop_desc_plural": "tenants with green lanyards",
        "question": "did your unit pass the inspection?",
        "whether_knowledge": "they know if their own unit passed",
        "observed_status": "lanyards",
        "the_rest_phrase": "have red lanyards",
        "unk_txt": "did not know if their unit passed the inspection",
        "known_txt": "knew if their unit passed the inspection",
        "pos_group": "the tenants with green lanyards",
        "neg_group": "the tenants with red lanyards",
    },
}
