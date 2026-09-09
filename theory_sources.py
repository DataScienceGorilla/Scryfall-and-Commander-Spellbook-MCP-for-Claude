"""
Sources for the Commander deckbuilding THEORY corpus (see theory_ingestion.py).

Each source is a YouTube channel; the ingester enumerates its uploads and pulls
the caption transcript for every video. Add or remove channels here, then re-run
theory_ingestion.py (it's incremental - only new videos get fetched).
"""

SOURCES = [
    {
        "name": "Rebel Lily",
        "author": "Rebel Lily",
        "channel_url": "https://www.youtube.com/@RebellLily/videos",
    },
    {
        # The Magic Mirror Podcast is published on this channel too, so this one
        # source covers both The Trinket Mage's essays and the podcast.
        "name": "The Trinket Mage",
        "author": "The Trinket Mage",
        "channel_url": "https://www.youtube.com/@thetrinketmage/videos",
    },
    {
        "name": "3/3 Elk",
        "author": "3/3 Elk",
        "channel_url": "https://www.youtube.com/@33elk/videos",
    },
    {
        "name": "Salubrious Snail",
        "author": "Salubrious Snail",
        "channel_url": "https://www.youtube.com/@salubrioussnail/videos",
    },
    {
        "name": "Commander Challenge",
        "author": "Commander Challenge",
        "channel_url": "https://www.youtube.com/@TheCommanderChallenge/videos",
    },
    {
        "name": "Wise Hoopoe",
        "author": "Wise Hoopoe",
        "channel_url": "https://www.youtube.com/@WiseHoopoeMTG/videos",
    },
    {
        # This channel hosts several shows (the podcast, Game Knights, Extra Turns,
        # Turn Talk...). We want ONLY the podcast, whose episodes are titled
        # "... | The Command Zone <number> | ...", so include-filter on that.
        "name": "The Command Zone",
        "author": "The Command Zone",
        "channel_url": "https://www.youtube.com/@commandcast/videos",
        # "Command Zone <n>" (older eps drop the "The"); does not match Game
        # Knights / Extra Turns / "Commander", so it's the podcast only.
        "include_title": ["Command Zone"],
    },
]
