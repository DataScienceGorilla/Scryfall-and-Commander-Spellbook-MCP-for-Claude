# Rules Gotchas - candidates for review

*Extracted from 10 rules-explainer videos by `distill_theory.py --rules`. NOT loaded by the advisor: check each point (especially VERIFY BEFORE USE) and merge the good ones into advisor_app.py's RULES INTERACTIONS section by hand.*

# RULES GOTCHAS FOR COMMANDER DECKBUILDING

## COLOR IDENTITY & DECK LEGALITY

### Color vs. Color Identity
- **Misconception:** A card's color (for effects like board wipes) is the same as its color identity (for deckbuilding legality), and lands/colorless-framed cards are colorless for both purposes.
- **Rule:** A card's *color* is determined only by mana cost (plus color indicators/characteristic-defining abilities like devoid, which override the mana cost). *Color identity* is cumulative: mana cost + color indicators + characteristic-defining abilities + color of mana symbols in rules text. Example: Pact of Negation has no mana cost but is blue due to its color indicator; a devoid card with green mana symbols has green color identity but is colorless in color.
- **For the advisor:** When checking legality for a commander's color identity, don't just look at mana cost—check color indicators and abilities like devoid/hybrid symbols in rules text. Don't assume a colorless-framed or cost-free card is actually colorless.
- **Source:** THAT'S How It Works?!! | The Command Zone 768

### Colorless Is Not a Color (Wastes & "Any Color" Fixing)
- **Misconception:** Wastes function like other basic lands, and colorless mana rocks (Arcane Signet, Command Tower) work in colorless decks because they tap for "any color."
- **Rule:** Colorless is the absence of color, not a color. Wastes have no basic land type. Effects that produce "mana of any color" (Arcane Signet, Command Tower, Star Compass) cannot produce colorless mana and are dead in mono-colorless decks. Wastes don't satisfy basic-land-type requirements (Myriad Landscape, domain).
- **For the advisor:** Do not recommend "any color" mana rocks or fixing lands (Command Tower, Arcane Signet, Star Compass, Myriad Landscape) for colorless/Wastes-based decks; recommend dedicated colorless ramp instead.
- **Source:** THAT'S How It Works?!! | The Command Zone 768

---

## COMMANDER & COMMAND ZONE

### Meld Commanders — Commander Damage, Identity, and Tax
- **Rule:** A melded creature is still the same physical commander card for commander-damage tracking (cumulative across forms) and for effects referencing "your commander" (Deflecting Swat, Fierce Guardianship). Commander tax only applies to casting the commander piece from the command zone, not the non-commander meld partner, and is unaffected by melding itself.
- **For the advisor:** Meld commanders (e.g., Mishra pairs) retain full commander-related support (protection spells, tax-reduction tools) even after melding—don't treat the melded form as a "new" permanent for identity purposes.
- **Source:** How Good Are the Meld Commanders? | The Command Zone 500

### Meld + Unearth Timing
- **Rule:** If an unearthed creature melds with your commander while unearthed, the resulting melded creature loses the unearth exile clause (it won't be exiled if it dies). If the unearthed creature leaves play *without* melding first, it gets exiled per unearth and cannot be used for melding later.
- **For the advisor:** In a meld-commander deck using unearth creatures (e.g., Phyrexian Dragon Engine), only unearth when melding can happen immediately (commander in play, haste available) to avoid losing the piece to exile.
- **Source:** How Good Are the Meld Commanders? | The Command Zone 500

### Planeswalkers Are Legendary Permanents
- **Rule:** All planeswalkers are legendary permanents (Oracle update), so "legendary"/"historic" tutors and synergy cards apply to them.
- **For the advisor:** Recommend legendary-tutors like Captain Sisay in Superfriends/planeswalker decks, since they can fetch any planeswalker, not just legendary creatures.
- **Source:** Ultimate Guide to Planeswalkers in Commander | The Command Zone 265

### Older Board Wipes Don't Hit Planeswalkers
- **Rule:** Cards templated before planeswalkers existed (Nevinyrral's Disk, Pernicious Deed: "destroy all artifacts, enchantments, and creatures") do not destroy planeswalkers, since planeswalkers aren't named in the text.
- **For the advisor:** Recommend these older wipes specifically in planeswalker-heavy ("Superfriends") decks as effectively one-sided board wipes; pair with recursion for repeatable value.
- **Source:** Ultimate Guide to Planeswalkers in Commander | The Command Zone 265

---

## TOKENS & COPIES

### Tokens Are Not Cards
- **Misconception:** Tokens function identically to cards for any graveyard/recursion/"card" effect.
- **Rule:** Tokens are not cards; effects that reference "card" specifically do not interact with tokens (e.g., Dauthi Voidwalker's exile-replacement only affects cards, not tokens — tokens still die normally and trigger death-synergy like Blood Artist).
- **For the advisor:** When building around graveyard hate or death-synergy, check whether the payoff cares about "cards" (excludes tokens) vs. any creature dying (includes tokens) — this changes whether token-heavy decks get full value.
- **Source:** THAT'S How It Works?!! | The Command Zone 768

### "Copy and Cast" vs. "Copy Target Spell"
- **Misconception:** All "copy" effects are functionally identical for triggering cast-matters synergies.
- **Rule:** Effects that create a copy in a zone which you then cast (God-Eternal Kefnet) count as "casting" and trigger cast-matters cards (Gutter Snipe). Effects that put a copy directly on the stack (Fork) never "cast" it, so cast-triggers don't fire (though "copy" triggers like Magecraft still do). Prepared/exiled copies (Reality Fracture) aren't cards, so "played from exile" triggers (Prosper) won't fire.
- **For the advisor:** When building spells-matter synergies (Gutter Snipe, Prosper), verify whether your copy-effects say "cast," "copy," or require a "card," since this determines which payoffs actually trigger.
- **Source:** THAT'S How It Works?!! | The Command Zone 768

---

## MANA, COSTS & CASTING

### Mana Cost vs. Mana Value for Alternate Casting
- **Misconception:** Any "cast for X" or flashback-granting effect works on spells with no mana cost (suspend cards like Living End).
- **Rule:** Spells with no printed mana cost have mana value 0 but no mana cost. Effects referencing "mana value" (cascade, Jace, Reawakened's plot) work on them; effects referencing "mana cost" (Snapcaster Mage flashback, Fblthp's plot) do not function, since there's no mana cost to copy.
- **For the advisor:** When including Snapcaster Mage-style effects or plot/suspend synergies, check whether the enabling text says "mana cost" (fails on no-cost spells) or "mana value" (works) before recommending the pairing.
- **Source:** THAT'S How It Works?!! | The Command Zone 768

### Alternate Costs Are Mutually Exclusive; Alternate Currencies Are Not
- **Misconception:** Delve/convoke behave like flashback/overload and can't stack with other cost effects.
- **Rule:** Alternate casting costs (flashback, overload, Force of Will's alternate cost) are mutually exclusive — only one applies per cast. Delve and convoke are alternate *currencies* (different ways to pay the same cost), and can be combined with an alternate cost (e.g., Snapcaster-granted flashback on Treasure Cruise can still be paid down with delve).
- **For the advisor:** Don't assume delve/convoke are blocked by other cost-reduction/alternate-cost effects on the same spell—recommend stacking delve with flashback-granting effects for extra value; warn against expecting two alternate costs (e.g., flashback + overload) to combine.
- **Source:** THAT'S How It Works?!! | The Command Zone 768

### Affinity Counts Treasures Twice; Improvise Does Not
- **Rule:** Total cost (including affinity) is calculated before mana abilities are activated, so a treasure can count toward affinity's cost reduction and then be sacrificed for mana. Improvise is checked after mana abilities are activated (during cost payment), so you cannot sacrifice an artifact for mana and also tap it for improvise on the same spell.
- **For the advisor:** In artifact/treasure-affinity decks, affinity gets extra value from treasures that regular improvise-based artifact decks won't get—factor this into evaluating affinity vs. improvise payoffs when choosing a build-around.
- **Source:** THAT'S How It Works?!! | The Command Zone 768

---

## COMBAT

### Lifelink Is Not a Trigger
- **Misconception:** Lifelink works like old "whenever this deals damage, gain that much life" trigger cards (Spirit Link).
- **Rule:** Lifelink causes life gain simultaneously with damage, before state-based actions are checked; a player who would die to lethal damage survives if they lifelink-gain life in that same event. Older triggered "gain life equal to damage dealt" cards resolve as separate triggers and won't save a player from simultaneous lethal damage the way Lifelink does.
- **For the advisor:** Recommend true Lifelink keyword sources (and granters like Loxodon Warhammer) over older trigger-based life-gain-on-damage cards for defensive/race value.
- **Source:** Level Up Your Combat Phase | The Command Zone 607

### Double Strike Doubles Combat-Damage Triggers
- **Rule:** Double strike causes "whenever this creature deals combat damage" triggers to fire twice (once in first strike step, once in regular damage step). This also doubles Lifelink's effective gain when paired with Lifelink.
- **For the advisor:** Recommend giving double strike to creatures with damage-trigger payoffs (e.g., card-draw-on-damage effects) or Lifelink creatures, as this effectively doubles their value; prioritize double-strike-granting equipment/auras in such decks.
- **Source:** Level Up Your Combat Phase | The Command Zone 607

### Creatures Entering "Tapped and Attacking" Skip Attack Triggers
- **Rule:** "Whenever this creature attacks" triggers only from being *declared* an attacker. Creatures put onto the battlefield already attacking (Ilharg, the Raze-Boar) never trigger attack-triggers, and also don't need to pay attack-taxes like Ghostly Prison/Propaganda.
- **For the advisor:** Don't expect attack-trigger payoffs (combat-matters cards) to work with "enters attacking" effects like Ilharg; note these effects do bypass attack-tax cards, which can be a deliberate upside to build around.
- **Source:** Level Up Your Combat Phase | The Command Zone 607; THAT'S How It Works?!! | The Command Zone 768

### Creatures Stay "Attacking" Through End of Combat
- **Rule:** Attacking creatures remain attacking through the end of combat step, even after damage is dealt. This lets end-of-combat cards (Aetherize, Desert, Reconnaissance, Mandate of Peace) still interact with them post-damage.
- **For the advisor:** Recommend end-of-combat utility (Reconnaissance for pseudo-vigilance, Desert for cleanup, Mandate of Peace as flexible fog/spell-lock) as higher-value, flexible includes rather than narrow pre-combat tricks.
- **Source:** Level Up Your Combat Phase | The Command Zone 607

### Ninjutsu Requires "Unblocked" Status (Declare Blockers Step Only)
- **Rule:** A creature only gains "unblocked" status during/after the declare blockers step; Ninjutsu cannot be activated during declare attackers even on an obviously-unblocked creature. Ninjutsu also remains usable in the end of combat step on an unblocked attacker.
- **For the advisor:** Recommend pairing Ninjutsu with evasive creatures (flying, menace, unblockable enablers like Rogue's Passage) to reliably reach "unblocked" status; Ninjutsu decks benefit from end-of-combat-step activations to dodge removal and reset for future turns.
- **Source:** Level Up Your Combat Phase | The Command Zone 607

### Damage Assignment Order (Post-Foundations Change)
- **Rule:** VERIFY BEFORE USE — As of the Foundations (2024) rules update, there is no declared blocker-ordering step; the attacking player freely assigns combat damage among blockers in any distribution (including overassigning lethal to one blocker) during the combat damage step itself, rather than needing to assign lethal to an ordered first blocker before moving to the next.
- **For the advisor:** This is a meaningful change from older combat-math assumptions (CR previously required ordering blockers and assigning lethal in order); double check against current Comprehensive Rules before advising based on pre-2024 blocker-ordering logic. Note: this appears to contradict older CR 509/510 blocker-ordering rules as commonly known, so confirm the actual current wording before using this to justify deck choices.
- **Source:** THAT'S How It Works?!! | The Command Zone 768

---

## GRAVEYARD, COUNTERS & SELF-MILL MECHANICS

### Descend Counts Permanents Only (Not Like Delve)
- **Misconception:** Descend works like Delve, counting all cards in the graveyard.
- **Rule:** Descend only counts permanent cards in the graveyard (not instants/sorceries). This intentionally makes it weaker/slower than Delve.
- **For the advisor:** Don't treat Descend payoffs as having the same explosive graveyard-fill requirements as Delve cards; they need permanent-heavy graveyard filling (creatures, lands, artifacts) rather than any card type.
- **Source:** Every Mechanic in Lost Caverns of Ixalan Explained

### Discover Is Not Cascade
- **Misconception:** Discover works exactly like Cascade (must cast the found card).
- **Rule:** Discover lets you choose to put the found card into hand instead of casting it, and the number is a fixed value independent of the card's own mana cost (unlike Cascade, which can be exploited via cost-reducers).
- **For the advisor:** Recommend Discover sources as flexible value/dig tools (can grab reactive cards like counterspells) rather than pure cascade-style free-spell generators; the discover land cycle is a safe, strong include as fixing + late-game value in most casual Commander decks.
- **Source:** Every Mechanic in Lost Caverns of Ixalan Explained

### Surveil Is Stronger Than Scry for Graveyard Decks
- **Rule:** Surveil lets you put looked-at cards into the graveyard instead of only bottom-of-library, enabling self-mill while filtering.
- **For the advisor:** In graveyard/reanimator decks (Muldrotha-style), prioritize Surveil-bearing cards over plain Scry effects, since they double as self-mill enablers.
- **Source:** Guilds of Ravnica Set Review | The Command Zone 234

### Jump-Start Costs a Discard (Unlike Flashback)
- **Rule:** Jump-Start lets you cast a card from the graveyard by paying its mana cost and discarding a card (then it's exiled); unlike Flashback, it requires an additional discard cost.
- **For the advisor:** Still recommend as card advantage in singleton decks (effectively "replaces" an unwanted hand card with a graveyard spell), but don't treat it as free value equivalent to Flashback.
- **Source:** Guilds of Ravnica Set Review | The Command Zone 234

---

## EQUIPMENT, AURAS & STATIC EFFECTS

### Koan's Aid Grants Flash to Auras/Equipment and Free Attachment
- **Rule:** Koan's Aid lets you cast Auras and Equipment as though they had flash, and automatically offers to attach an Equipment to a creature you control when it enters, without paying the equip cost.
- **For the advisor:** Strong recommend for Voltron strategies — removes the equip-cost tax that normally limits equipment playability and enables holding up interaction while still deploying gear.
- **Source:** Eldritch Moon Set Review | The Command Zone 113

### Selfless Spirit as Budget Board-Wipe Protection
- **Rule:** Selfless Spirit sacrifices for two mana to give all your creatures indestructible until end of turn.
- **For the advisor:** Recommend as a cheap, efficient board-wipe insurance piece in any token-heavy or creature-based deck, generally superior to pricier legacy options like Faith's Reward.
- **Source:** Eldritch Moon Set Review | The Command Zone 113

---

## REMOVAL & TARGETING

### Assassin's Trophy Destroys Lands Too
- **Misconception:** Compared to Path to Exile, it seems like a strict downgrade due to the opponent's ramp.
- **Rule:** Assassin's Trophy destroys ANY permanent an opponent controls, including lands (it doesn't say "nonland"), though the opponent may search for a basic land to replace it.
- **For the advisor:** Recommend as a near-auto-include two-mana answer in any B/G deck — the land-destroy downside is minor compared to its unconditional permanent removal (works on planeswalkers, enchantments, etc.) and unlike Path, it destroys (not exiles).
- **Source:** Guilds of Ravnica Set Review | The Command Zone 234
