"""
Ingest Rebel Lily's Commander Template Academy articles into the theory RAG corpus.
================================================================================
Clean written deckbuilding theory (Ramp / Card Draw / Removal / Engines / Multipliers /
Operational Threshold / How to Build) by Lily (Rebel Lily), founder of Commander Template.
The site is a Cloudflare-protected Next.js SPA, so the article bodies were captured via the
in-app browser and are embedded here. Re-run to upsert (idempotent) into the same
`mtg_deckbuilding_theory` Chroma collection the video transcripts live in.

    python ingest_academy.py
"""
from theory_ingestion import get_collection, chunk_text

AUTHOR = "Rebel Lily"
SOURCE = "Rebel Lily Academy (Commander Template)"
BASE = "https://commandertemplate.com/academy/"

# slug -> (title, body). Bodies are the article prose (nav/footer stripped).
ARTICLES = {
    "how-to-build-commander": ("How to Build Commander", """
When we build a Commander deck, it always starts with the card in the name of the format, your legendary creature Commander. Picking your Commander does a lot of the work for you, because focusing on your Commander can guide you on the strategy and approach you should take with your deck to win with your Commander.

Some Commanders win by getting really powerful through being enchanted by Auras or wielding Equipment, while others may empower a field of creatures or strengthen the spells you cast to make them more deadly. Commanders and decks can take a lot of forms in terms of strategy, and the easiest way to start to get the most out of your Commander is by building with Themes.

Themes are like packages of cards that all do similar things, or related to one idea. A Kaalia of the Vast deck would want a theme of angels, demons, or dragons; a Lathril, Blade of the Elves deck would want a theme of Elf cards. The reason we want a package of multiple cards within a single theme is because a Commander deck has 99 unique cards and having multiples of the same type of effect helps give our deck more consistency in seeing the cards it needs.

Commander Template uses a classic model known as '8 by 8 theory' which advises players to build each thematic package using 8 cards. Statistically, having 8 cards within a theme means you have a 45.6% chance to have one in your opening hand, and 62.5% chance to see one by turn 4. The more important a theme is, the more you want to adjust; you can double a package to 16 cards to raise the chance to 87.2% by turn 4.

Some themes can also be 'engines', which synergize with other themes within the deck. For example, a theme that gives you life when you take action synergizes well with cards that do things when you gain life, creating a synergistic engine of cause and effect.

Normally, a Commander deck will have about 31 theme cards to support your Commander's strategy. To make it clean, use 32 cards so that's 4 theme packages of 8 cards each.

You might notice we haven't talked about cards like 'ramp' or 'card draw' or 'removal' yet. Those are just as important, but we start with theme cards because those make our deck unique and are the true heart of the strategy. The rest of the utility packages make sure the deck can perform its strategy.

In our base template, we recommend 12 ramp, 10 card draw, and 10 interaction, with the remainder of the deck being 36 lands. Card draw, ramp and lands are intertwined because the goal of these cards is to help you cast your spells. Adding 36 lands and 12 ramp together is about 48 cards dedicated to mana, which gives about a 75% chance to have 3 sources in your opening 7 and a 94.4% chance to have 3 by turn 3.

We could replace every ramp card with a land for the same mana effect, but lands and ramp are divided this way to leverage the benefit of playing ahead - being able to play ahead of one-land-a-turn is even better.

This leaves 10 card draw and 10 removal. Card draw smooths out clunkiness in what we draw; more looks to find a card is the same effect as taking more turns to see it. The real work comes later when we goldfish our deck.
"""),
    "ramp": ("Ramp", """
Named after the spell 'Rampant Growth', there are two main goals of ramping: acceleration and mana fixing.

Cards like Llanowar Elves or Sol Ring let a player effectively be turns ahead by giving them more mana to cast spells with - playing spells ahead of schedule (a 4-mana spell on turn 2) or a higher quantity of spells. Ramping is highly valued in Commander because the early turns have less pressure and games focus on playing more expensive threats or engines to snowball into victory.

The problem with ramp is the 'hot garbage' concept: if you play too few lands and too much ramp, you might spend turn 2 finding your third land, and that Rampant Growth would have been better as just a land. This creates an interconnected relationship between your land and ramp spells.

The other important facet of ramp is type, dictated by your mana curve and strategy. A big-mana deck wants more 3-mana ramp like Cultivate or Kodama's Reach; a leaner aggressive deck might not need much ramp if most spells are 1-3 mana.

1-Mana Ramp (Birds of Paradise, Avacyn's Pilgrim, Noble Hierarch): mostly creatures known as 'dorks'. They can't tap immediately, but are some of the best ramp for lower-mana decks - a 3-mana commander on turn 2.

2-Mana Ramp (Talisman, Arcane Signet, Nature's Lore): the workhorse ramp. Artifact 'rocks' here are great because they immediately refund 1 mana (no summoning sickness), enabling follow-up plays - two rocks on turn 3 starts turn 4 with 6 mana. The land-fetchers are some of the best color fixing.

3-Mana Ramp (Chromatic Lantern, Relic of Legends, Kodama's Reach): the top end of playable mana ramp; beyond this, opening with too many lands isn't desirable. Still, some have great utility - Chromatic Lantern fixes colors, Relic of Legends is strong in legend-heavy decks, and Cultivate/Kodama's Reach find two lands.

Rituals and Cost Reducers (Ruby Medallion, Dark Ritual, cost reducers): also ramp, with specialized needs. Medallions are excellent in low-color decks casting multiple spells a turn. Rituals are powerful one-off mana generators, more mana-efficient than rocks but temporary and higher-risk.

In Commander Template, the ramp and land count is dynamically controlled via the 'operational threshold'; for simplicity it's controlled by the mana value of your commander.
"""),
    "card-draw": ("Card Draw", """
Card draw is one of the best effects in Magic, because more cards in hand means more options and more things to do. In its simplest form, card draw has three primary functions: dig for better options, exhaust opponents through gaining resources, and refill your hand to maintain momentum. This creates three categories:

Dig (Impulse, Dig Through Time, Serum Visions): better in a deck that cares about specific cards, like a combo deck. You aren't getting an extra card, you're exchanging mana and time to get more options. Great to find a land, removal, protection, or a threat. Its flexibility is what makes it great.

Card Advantage (Harmonize, Esper Sentinel, Curiosity): keeps your hand flush with options. A steady stream of cards negates the variance of one draw a turn and lets you be more liberal with resources - trading creatures or casting removal knowing you have backup. Highly valuable in average, longer games; loses value as the game compresses (except high-volume engines like Esper Sentinel, Rhystic Study, Mystic Remora).

Refill (Windfall, Rishkar's Expertise, Mass Appeal): best to recover your hand after spending cards early, to maintain momentum or catch up after a board wipe. Aggressive and combo decks benefit most. Generally play 1-2 for mid-game momentum or late-game recovery.

While it's good to evenly distribute your ~12 card-draw slots across these three, each type matters differently depending on the deck and the role it plays.
"""),
    "removal": ("Removal", """
Interaction spells are spells you use to interact with your opponent's board. Different colors have access to different types: blue has counterspells, black removes creatures, white and green target enchantments, red removes artifacts.

Every deck needs removal because all four players are moving toward their finish line and will play threats that get too strong or outright win (an Overrun, a giant Torment of Hailfire). Two important things about removal in Commander:

Remember the primary goal of being proactive and playing to win - playing too much removal makes your deck too reactive and wastes mana/time developing your board. And spells that remove one thing are poor value, because you and one opponent are down a card while the other two are unaffected.

So be mindful of how much removal you play - not so little you can't interact, not so much you become reactive. Levers: specificity (Swords to Plowshares vs. flexible Beast Within/Generous Gift), range (single-target Doom Blade, scaled Vona's Hunger, mass Damnation), and timing (instant Abrade vs. sorcery Brotherhood's End).

When designing interaction: want enough flexibility to answer a wide variety of threats; want it lean enough to play a spell and remove something the same turn (instant speed to act on other turns); prioritize threats/card types you know are problematic for your deck; and want 1-2 board wipes for when there's simply too much to answer.

The template recommends 10 removal spells - enough to draw 1-2 per match, not so much you prolong the game. Prioritizing flexibility and speed is a safe baseline.
"""),
    "engines": ("Engines", """
An engine is a series of cards that work together to convert one resource into another, or take each other's effects and reward you with additional value. For example, Ajani's Mantra generates 1 life each upkeep; with Dawn of Hope you convert that into card draw, with Enduring Tenacity into opponent life loss, with Nykthos Paragon into +1/+1 counters.

Resources you can generate: draw cards, create tokens, gain life, deal damage, cast spells, play lands. Each can be a driver for more value.

In every engine there are two types of cards: ENABLERS (cards that generate a resource) and PAYOFFS (cards that turn that resource into value).

Choosing Enablers: the most important cards, the backbone of the engine. You want them (1) cheap in mana value - you don't want to wait late to get going; (2) self-sufficient - they work on their own rather than waiting on another card; (3) consistent - repeatable in how they generate the resource (upkeep trigger, activated ability, or a common triggered ability). Not every enabler hits all three; card design creates trade-offs.

Choosing Payoffs: the reverse of enablers - they take the resource and convert it to creatures, card draw, damage, or mana. But the emphasis is that payoffs should actually pay off toward WINNING. Prefer threats or game-enders (Archangel of Thune, Vito, Angelic Accord, Sanguine Bond) over cards that generate value but don't close the game (Dawn of Hope, Well of Lost Dreams). Play FEWER payoffs than enablers, since payoffs are useless until an enabler supports them - you're happier to draw them later.

How to build: start with at least 8 enablers and 8 payoffs (easy to remember), but really it should be at least 12 enablers and 6 payoffs. With 12 enablers you have a 60.7% chance to open one and 77.8% to see one by turn 4. With 6 payoffs you have a 51.6% chance to see one by turn 4. The last ~3 cards in the package should be synergy cards that scale or double your effects (Parallel Lives, Torbran) - multipliers.
"""),
    "multipliers": ("Multipliers", """
Multipliers are cards that double a form of effect you have on the board, or double an effect for your spells. Doubling Season doubles counters and tokens; Twinflame Tyrant doubles your damage. Multipliers don't only double - they can be additive: Guttersnipe adds 2 damage per noncreature spell, Glorious Anthem adds +1/+1 to each creature. Extra combat spells and extra turn spells are also multipliers.

Multipliers matter because Commander is a game of value and economy - grand battles where the battalion matters more than the unit, so cards that scale your existing board or spells are very useful.

You can scale two ways: make your board WIDER per spell (Lys Alana Huntmaster makes an extra elf per elf spell) or TALLER (Elvish Champion gives each elf +power). But the common problem: if your deck is nothing but multipliers, they're sad on their own (Elvish Champion with no elves does nothing). This is why multipliers are a SECONDARY type of card that boosts your themes - like payoffs to enablers, multipliers give more value to your enablers and spells.

How to choose: think about the most common resource/path in your deck, then find effects that are additive or multiplicative to that path. The advanced form is thinking in reverse about how your deck falls short - a burn deck that runs out of spells can play Mizzix's Mastery to recast them, or Young Pyromancer so spells come with bodies.

Multipliers are extremely powerful with the right conditions but often useless alone. Play around four per theme, and goldfish to reliably have a multiplier by turn 5-6 to escalate when needed. If a multiplier isn't doing enough, you need more enablers; if you have a big board that doesn't move, you need more ways to scale it.
"""),
    "operational-threshold": ("Operational Threshold", """
The operational threshold (or 'the flip turn') is the amount of mana required for your deck to consistently operate and execute its gameplan - the turn the deck flips from setting up to executing.

It's based on the 'fundamental turn' concept (Zvi Mowshowitz, 2000): the turn an archetype starts to win. An aggro deck's fundamental turn might be 4; a control deck's might be 12. Commander doesn't work exactly this way (multiplayer, more dynamics), but the lesson holds: every deck has a threshold of mana it needs to 'start doing the thing'. Most Commander decks need 4-5 mana to be ready to cast most of their spells and play interactively - but some need less (Kaalia needs 4; a Yoshimaru deck 3-4).

The operational threshold dictates the type and quantity of ramp you need. The second factor is PRESSURE: the amount of pressure to do something by a certain turn. Hitting 5 mana looks very different across brackets - a Bracket 2 deck hitting 5 is much more lax than a Bracket 5 deck needing 5 for Ad Nauseam. The reminder: hitting your operational threshold as fast as possible is always good; race past your set-up turns and flip into executing.

The operational threshold isn't always your Commander's mana value - it's the mana value at which your deck begins turning set-up into victory. Think of your multipliers or wincons (Doubling Season, Triumph of the Hordes) which might sit in a low-cost commander's deck (Rhys, the Redeemed at 1). If your wincon is a 6-mana Rampaging Baloths but your token doublers/generators cost 3-4, your flip turn is when you play the enablers (Retreat to Emeria), and the Baloths is something you draw into.

Decide your threshold by looking at the mana curve and finding where the most crucial spells need to be cast for your deck to execute. On Commander Template, land count, ramp, and draw numbers are all tied to this number (default: commander's mana value, adjustable in Intelligence Mode).
"""),
}


def main():
    col = get_collection()
    total_chunks = 0
    for slug, (title, body) in ARTICLES.items():
        chunks = chunk_text(body.strip())
        ids = [f"academy:{slug}:{i}" for i in range(len(chunks))]
        metas = [{
            "source": SOURCE, "author": AUTHOR, "title": title,
            "url": BASE + slug, "video_id": f"academy:{slug}", "type": "article",
        } for _ in chunks]
        col.upsert(ids=ids, documents=chunks, metadatas=metas)
        total_chunks += len(chunks)
        print(f"  [{len(chunks):>2} chunks] {title}")
    print(f"Ingested {len(ARTICLES)} academy articles, {total_chunks} chunks -> mtg_deckbuilding_theory")


if __name__ == "__main__":
    main()
