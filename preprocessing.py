import re
import pandas as pd
import numpy as np

def clean_description(desc, stock_code):
    """
    Clean and standardize product descriptions.
    This function normalizes text, handles special cases, and categorizes descriptions based on patterns.
    """
    if pd.isna(desc) or desc is None or str(desc).strip() == "" or desc == "Unknown":
        return "Unknown"

    desc = str(desc).strip()
    stock_code = str(stock_code).strip()
    desc = ' '.join(desc.split())

    if re.match(r"^\?{1,4}$", desc):
        return "Unknown"

    if stock_code in ["BANK CHARGES", "CRUK", "B", "D", "M"]:
        return desc.upper()

    if re.search(r"\bNo$", desc):
        return desc.upper()

    if desc.startswith("*"):
        return desc.replace("*", "").strip().upper()

    if desc == "FOLK ART GREETING CARD,pack/12":
        return "FOLK ART GREETING CARD, PACK/12"
    elif "TRADITIONAl" in desc:
        return desc.upper()
    elif "Voucher" in desc:
        return desc
    elif "CHECK" in desc:
        return "Inventory Check"
    elif "WET/MOULDY" in desc:
        return "Damaged"
    elif "CREDIT ERROR" in desc:
        return "Credit Issue"
    elif "MIA" in desc:
        return "Missing"
    elif "POSSIBLE DAMAGES OR LOST?" in desc:
        return "Damaged"

    if desc.isupper():
        return desc

    if desc.lower() in [
        "dr. jam's arouzer stress ball",
        "dad's cab electronic meter",
        "flowers handbag blue and orange"
    ]:
        return desc.upper()

    regex_combined = r"\b\d+\s*(g|cm|x\s*\d+)\b|\b\d+x\d+cm\b|\b\d+cmx\d+cm\b|\b\d+x\d+x\d+cm\b"
    if re.search(regex_combined, desc.lower()):
        return desc

    desc_lower = desc.lower()

    # Category mapping
    category_map = {
        "Unknown": [
            r"\bmystery\b",
            r"\bmichel oops\b"
        ],
        "Missing": [
            r"\bmissing\b",
            r"\bcan't find\b",
            r"\bcan't find\b",
            r"\bnot rcvd\b"
        ],
        "Damaged": [
            r"\bdamage\b",
            r"\bdamages\b",
            r"\bdamaged\b",
            r"\bbroken\b",
            r"\bcrushed\b",
            r"\bdagamed\b",
            r"\bwet\b",
            r"\bmouldy\b",
            r"\bfaulty\b"
        ],
        "Lost": [r"\blost\b"],
        "Found": [r"\bfound\b"],
        "Display Item": [r"\bdisplay\b", r"\bshowroom\b"],
        "Stock Adjustment": [
            r"\badjustment\b",
            r"\bre-adjustment\b",
            r"\badjust\b",
            r"\bbad debt\b",
            r"\bstock allocate\b",
            r"\bsale error\b",
            r"\badd stock to allocate online orders\b",
            r"\bto push order througha s stock was\b"
        ],
        "Barcode Issue": [r"\bbarcode problem\b", r"\bwrong barcode\b"],
        "Broken Item": [r"\bbreakages\b", r"\bsmashed\b", r"\bcracked\b"],
        "Sample Item": [r"\bsample\b", r"\bshow samples\b", r"\bsamples\b"],
        "Discarded Item": [r"\bthrown away\b", r"\bunsaleable\b", r"\bthrow away\b"],
        "Wrongly Sold as Sets": [
            r"\bdotcom set\b",
            r"\bsold as sets\b",
            r"\bsold sets\b",
            r"\bsold in set\b",
            r"\bsold as set\b",
            r"\bsold as 1\b",
            r"\bsold as 6\b"
        ],
        "Online Marketplace": [
            r"\bamazon\b",
            r"\bamazon sales\b",
            r"\bebay\b",
            r"\bdotcom\b",
            r"\bonline retail orders\b",
            r"\bfba\b",
            r"\bwebsite fixed\b",
            r"\bdotcomstock\b"
        ],
        "Inventory Check": [r"\bcheck\b", r"\bcounted\b"],
        "Returned Item": [r"\breturned\b"],
        "Shipping & Logistics": [
            r"\bcargo order\b",
            r"\bmailout\b",
            r"\bnext day carriage\b"
        ],
        "Incorrect Stock Entry": [
            r"\bmixed up\b",
            r"\blabel mix up\b",
            r"\bincorrect\b",
            r"\bcredited\b",
            r"\bincorrectly\b",
            r"\bmix up with c\b",
            r"\bstock creditted wrongly\b"
        ],
        "Miscellaneous": [
            r"\bjohn lewis\b",
            r"\bhistoric computer\b",
            r"\blighthouse trading\b",
            r"\bhigh resolution image\b",
            r"\bhad been put aside\b",
            r"\balan hodge cant mamage this section\b"
        ],
        "Promotional Item": [r"\bgiven away\b"],
        "Testing Item": [r"\btest\b"],
        "Credit Issue": [r"\bdid\s+a\s+credit\s+and\s+did\s+not\s+tick\s+ret\b"],
        "Wrongly Coded": [r"\bwrong code\b", r"\bwrongly marked\b"],
        }

    for category, patterns in category_map.items():
        if any(re.search(pattern, desc_lower) for pattern in patterns):
            return category

    return "Uncategorized"

def update_stockcode_and_description(df):
    """
    Update stock codes and descriptions for consistency.
    Fixes common issues with stock codes and standardizes descriptions.
    """
    df = df.copy()
    
    # Numbers to check in descriptions
    stockcode_numbers = ["20713", "84930", "23343", "22467", "22804", "85123A", "22719"]

    for index, row in df.iterrows():
        desc = str(row["Description"])
        stock_code = str(row["StockCode"])

        # Fix stock codes with lowercase "m" by converting to uppercase "M"
        df.at[index, "StockCode"] = re.sub(r"m", "M", stock_code, flags=re.IGNORECASE)

        # Check if any stockcode number is inside the Description
        matched_code = next((code for code in stockcode_numbers if code in desc), None)

        if matched_code:
            # Replace StockCode with the matched number
            df.at[index, "StockCode"] = matched_code

            # Find mode (most common) description for this StockCode
            mode_description = df[df["StockCode"] == matched_code]["Description"].mode()

            if not mode_description.empty and mode_description[0] != df.at[index, "Description"]:
                df.at[index, "Description"] = mode_description[0]
            else:
                df.at[index, "Description"] = "Unknown"

        # Special case: If '85123a' is found in the Description, replace StockCode with '85123A'
        if "85123a" in desc.lower():
            df.at[index, "StockCode"] = "85123A"

            # Find mode (most common) description for "85123A"
            mode_description = df[df["StockCode"] == "85123A"]["Description"].mode()

            if not mode_description.empty and mode_description[0] != df.at[index, "Description"]:
                df.at[index, "Description"] = mode_description[0]
            else:
                df.at[index, "Description"] = "Unknown"

    return df

# Dictionary mapping products to categories based on keywords in descriptions
category_mapping = {
    # Specific functional mini-mappings
    'light_holder': ['light holder'],
    'bag': ['bag'],
    'chalkboard': ['chalkboard'],
    'peg': ['peg'],
    'bottle': ['water bottle'],
    'wicker': ['wicker'],
    'paper_towel': ['paper towel'],
    'furniture': ['stool'],
    'pet_accessories': ['dogs collar', 'dog collar'],

    # Merged wall decor related
    "wall_hanging_decor": [
        "bunting", "garland", "metal sign", "wood letters", "wood block letters",
        "hanging heart mirror decoration", "wreath", "wooden advent calendar", "tile hook", "wall plaque",
        "french wc sign blue metal", "bathroom metal sign", "red charlie+lolapersonal doorsign",
        "french blue metal door sign", "sign", "wood black board", "wooden letters", "hanging", 'photo clip', 'wall clock',
        'hook', 'shells print', 'chalice', 'metal swinging bunny', 'standing fairy', 'clock', 'blackboard',
        'key holder', 'letter holder', 'wall thermometer', 'wall mirror'
    ],

    # Decorative combined: decorative_items + home_decor + decorative_accessories
    "decorative_items": [
        'glass cloche', 'lantern', 'star decoration', 'birdcage', 'ceramic cherry cake money bank',
        'antique glass dressing table pot', 'vase', 'tile', 'gnome', 'heart', 'heart ivory trellis large',
        'painted metal star', 'magnets', 'magnet', 'decor', 'retrospot', 'keepsake box', 'photo album',
        'platter', 'bell', 'frame', 'mirror', 'wicker star', 'doorstop', 'chandelier',
        'art', 'ornament', 'ceramic', 'flying ducks', 'block letters', 'enamel', 'photo cube', 'geisha girl', 'clam shell',
        'board cover', 'memoboard', 'shells print', 'table clock', 'paperweight', 'feather tree', 'lace c/cover', 'love seat',
        'floral pink monster', 'floral blue monster', 'magic sheep wool', 'acrylic jewel','canvas', 'doorknob', 'acrylic jewel',
        'porcelain', 'circular mobile', 'chime', 'buddha head', 'white base', 'metal cat', 'laurel star', 'screen', 'colour glass',
    ],

    # Artificial plants and flowers
    "artificial_plants_and_flowers": [
        "artificial flower", "flower jug", "flower garland", "sprig lavender", "potted plant"
    ],

    # Merge kitchen-related categories
    "kitchenware_and_tableware": [
        'tray', 'teaspoons', 'tumbler', 'milk', 'tea set', 'cake cases', 'set/5 red retrospot lid glass bowls',
        'set of 36 paisley flower doilies', 'set of 72 retrospot paper doilies', 'coaster', 'cake tin',
        'pizza plate in box', 'popcorn holder', 'saucer', 'paper plates', 'paper cups', 'cutlery',
        'snack boxes', 'food container', 'black kitchen scales', 'ivory kitchen scales', 'organiser',
        "plate", "bowl", "cup", "wine glass", "champagne glass", "sundae dish", "beurre dish", "teapot",
        "butter dish", "mug", "cake stand", "cake mould", "dish", 'beakers', 'cakestand', "kitchen", "breakfast",
        'food cover', 'tea', 'place setting', 'biscuit cutters', 'toastrack', 'lolly holders', 'coffee set', 'spoon',
        'aperitif', 'straws', 'straw', 'cookie cutter', 'pantry', 'wine', 'sugar', 'cake', 'beaker', 'jug', 'cocktail',
        'baking', 'jampot', 'grinder', 'chopping board', 'spoons', 'spoon', 'coffee', 'd.o.f glass', 'd.o.f. glass', 'toast its', 'four cases',
    ],

    # Food storage merged
    "food_storage": [
        "storage jar", "glass jar", "bread bin", "biscuit tin", "coffee container", 'cannister'
    ],

    # Cooking tools merged
    "cooking_tools": [
        "squeezer", "colander", "frying pan", "chopsticks", "measuring tape", "tea cosy"
    ],

    # Soft furnishing + textile overlap removed (combined)
    "soft_furnishing_and_textile": [
        "cushion", "quilt", "throw", "rug", "curtain", "pouffe", "blanket", "mat", "doormat",
        "set of 4 modern vintage cotton napkins", "napkin", "tea towels", "placemats", "coasters",
        "flannel", 'table cloth', 'polyester'
    ],

    # Merged storage items
    "storage_items": [
        'tissue box', 'cabinet', 'coat rack', 'basket', 'set 3 wicker oval baskets', 'dog bowl chasing ball design',
        'trinket', 'magazine rack', 'jar', 'wastepaper bin', 'shelf', 'mini cases', 'crates', 'hook', 'oval boxes',
        'tins', 'oval box', 'box', 'boxes', 'folding shoe', 'metal box', 'storage tin', 'storage box', 'storage cubes',
        'bin', 'tin'
    ],

    # Stationery + Craft merged
    "stationery_craft_and_tools": [
        'recipe box blue sketchbook design', 'make your own flowers', 'herb marker chives', 'herb marker mint',
        'herb', 'paint', "pen", "writing set", "scissor", "pencil", "eraser", "ruler", "office", "gift tape",
        "stationery", "sketchbook", "notebook", "diary", "journal", "planner", "set of 6 vintage notelets kit",
        "book", "deluxe sewing kit", "travel sewing kit", "s/6 sew on crochet flowers", "victorian sewing kit",
        "sewing susan 21 needle set", "danish rose round sewing box", "alphabet stencil craft",
        "wood stamp set thank you", "wood stamp set best wishes", "wood stamp set happy birthday",
        "paint your own canvas set", "happy stencil craft", "freestyle canvas art picture",
        "calendar", "canvas set", "wood block letters", "chalk sticks", "chalkboard", "beads", 'make your own', 'making',
        'tape measure', 'tape', 'screwdriver', 'spirit level', 'paper fan', 'paper ball', 'paper', 'sewing', 'chalk', 'tissue ream',
        'stamp', 'hole punch', 'page photo'
    ],

    # Seasonal and festive merged
    "seasonal_and_festive": [
        'wooden tree christmas scandinavian', 'wooden star christmas scandinavian', 'rocking horse christmas',
        'painted sea shell metal windchime', 'metal rabbit ladder easter', 'christmas musical zinc star',
        'christmas musical zinc tree', 'zinc folkart sleigh bells', 'wood stocking christmas scandispot',
        'christmas tree', 'white christmas flock droplet', 'black christmas flock droplet',
        'christmas retrospot angel wood', 'christmas retrospot star wood', 'christmas star wish list chalkboard',
        "feltcraft christmas fairy", "christmas craft tree top angel", "christmas craft white fairy",
        "advent calendar gingham sack", "christmas gingham tree", "christmas gingham star",
        "christmas metal tags assorted", "christmas toilet roll", "set of 6 kashmir folkart baubles",
        "rocking horse red christmas", "smallfolkart bauble christmas dec", "pink christmas flock droplet",
        "folkart zinc star christmas dec", "glitter christmas star", "hen party cordon barrier tape", 'christmas ribbons',
        'romantic pink ribbons', 'pinks ribbons', 'blues ribbons', 'chocolate box ribbons', 'wrap cowboys', 'christmas', 'wrap apples',
        'scandinavian red ribbons', 'ribbons rustic', 'wrap red apples', 'ribbons,', 'ribbon', 'gift wrap', 'wrap', 'crackers',
        'santa', 'honeycomb fan', 'sandalwood fan'
    ],

    # Lighting and candles preserved
    "lighting_and_candles": [
        "t-light", "t-lights", "jam scented candles", "candle stick", "candle", "candle holder",
        "candleholder pink hanging heart", "star decoration painted zinc", "wooden owl light garland",
        "lights", "nightlight", "light", "cinnamon set of 9 t-lights", "lamp", "porcelain t-light holders assorted", 'bulb', 'silicone cube'
    ],

    # Accessories and fashion merged
    "accessories_and_fashion": [
        "necklace", "earring", "bracelet", "ring", "silver m.o.p orbit drop earrings", "edwardian drop earrings jet black",
        "necklace+bracelet set blue hibiscus", "letter 'd' bling key ring", "scarf", "hair", "bangle", "hat", "glove", "bag",
        "belt", "rain hat", "recycling bag", "parasol", "purse", "umbrella", "pink butterfly handbag w bobbles",
        "tote bag", "pink flower fabric pony", "green murano twist bracelet", "small yellow babushka notebook",
        "bobbles", "luggage tag", "shirt", "dress", "pants", "jacket", "socks", "cushion",
        "vintage union jack cushion cover", "french paisley cushion cover", "quilt", "doilies", "fairy cakes notebook a6 size",
        "cupcake", "slipper shoes", "jewellery box", "brooch", "badge", "flower garland", 'shopper', 'apron', 'passport cover',
        'neckl','neckl.', 'choker', 'lariat', 'bead charm', 'poncho', 'slipper', 'sock', 'sunglasses', 'bead', 'jewellery', 'backpack',
        'silk fan', 'sombrero', 'tiara', 'shoes', 'shoe', 'mobile', 'skirt', 'rucksack', 'clips', 'glasses case', 'diamante chain', 'key-chains',
        'flag'
    ],

    # Kids and toys as-is
    "kids_and_toys": [
        "toy", "harmonica", "doll", "action figure", "blocks", "building block word", "soldier skittles",
        "wooden skipping rope", "dominoes", "mini jigsaw dinosaur", "mini jigsaw bake a cake",
        "set of 6 wooden skittles in cotton bag", "paint set", "tattoos", "tattoo", "crayons",
        "magic drawing slate go to the fair", "retrospot party bag + sticker set", "spaceboy childrens egg cup",
        "mini cake stand with hanging cakes", "sleeping cat erasers", "set/12 taper candles", "puzzle", "board game",
        "jigsaw", "paper chain", "card party games", "game", "recipe box", "party cones", "lunch box", "picnic basket",
        "dolly", "sticker", "inflatable political globe", "booze & women greeting card", "200 red + white bendy straws",
        "set/4 fairy cake placemats", "boys vintage tin seaside bucket", "girls vintage tin seaside bucket",
        "retrospot childrens apron", "kings choice giant tube matches", "12 daisy pegs in wood box",
        "retrospot small tube matches", "set/3 rose candle in jewelled box", "retrospot red washing up gloves",
        "small red babushka notebook", "babushka", "snakes & ladders", "spinning tops", "dinosaur", "circus",
        "sock puppet", "bingo set", "holiday fun ludo", "feltcraft", "puppet", "easter", "skipping rope",
        "canvas art", "stencil", "windmill", "space toy", 'playhouse', 'tool set', "sporting fun", 'carriage',
        'beach spade', 'garden set', 'helicopter', 'teddy bear', 'swinging bunny', 'baby bib', 'crawlies', 'space', 'animals in bucket',
        'crawlie', 'ninja', 'dolphins', 'flying disc', 'toadstool', 'fluffy chicks', 'skittles', 'inflatable', 'baby gift', 'singing canary', 'lolly moulds',
        'bubbles', 'rocket', 'clay', 'kids rain', 'stress ball', 'bunnies', 'bunny', 'knitting', 'rubber', 'fly swat', 'rabbit', 'frog', 'knitted hen', 'felt farm animal',
        'sandcastles', 'sandcastle', 'snake', 'naughts and crosses'
    ],

    # Bath and beauty as-is
    "bath_and_beauty": [
        "makeup", "lipstick", "perfume", "shampoo", "conditioner", "hair comb", "cosmetic", "toothbrush", "shower",
        "bath", "tissues", "soap holder", "soap dish", "incense", "fragrance", "lavender", "oil burner", "towel",
        "essential balm", 'lip gloss',
    ],

    # Electronics and media
    "electronics_and_media": [
        "phone", "tablet", "laptop", "charger", "headphones", "boom box speaker", "radio", "tv tray", "lamp", 'alarm', 'ipod', 'electronic',
        'key fob'
    ],

    # Bottle and warmer
    "bottle_and_warmer": [
        "english rose hot water bottle", "water bottle", "bottle", "hot water bottle", "hand warmer",
        "red polka dot hand warmer", "union jack hand warmer"
    ],

    # Gardening
    "gardening": [
        "garden", 'gardening', 'plant', 'potting', 'garden thermometer', 'hen house', 'wheelbarrow', 'hammock', 'bird feeder', 'bird table',
        'dovecote', 'feeding station'
    ],

    # Miscellaneous preserved
    "miscellaneous": [
        "chocolate calculator", "victorian sewing box", "tomato charlie+lolacoaster set", "jam making set printed",
        "discount", "love heart napkin box", "woodland charlotte bag", "black record cover frame",
        "cosmetic bag vintage rose paisley", "kneeling pad", "plasters", "mirror", "drawer", "hanger", "torch",
         "egg", "slate", "sideboard", "red retrospot luggage tag",
        "christmas craft little friends", "3 wicker oval baskets", "lavender incense in tin", "piggy bank retrospot",
        "money box", "treasure chest", "gecko", "patches", "flower", "mushroom", "gnome", "space toy", "magnet",
        "daisy", "flower garland", "asparagus", "newspaper stand", "ashtray", 'postage', 'pannetone', 'card', 'gift tag',
        'birthday wrap', 'first aid', 'gift boxes', 'gift box', 'envelope', 'bank charges', 'gift voucher', 'packing charge',
        'manual', 'bicycle', 'missing', 'inventory check', 'online marketplace', 'damaged', 'samples', 'amazon fee', 'found', 'stock adjustment',
        'display item', 'wrongly sold as sets', 'barcode issue', 'lost', 'broken item', 'credit issue', 'returned item', 'wrongly coded', 'sample item',
        'shipping & logistics', 'fba', 'adjust bad debt', 'cruk commission', 'testing item', 'amazon', 'promotional item', 'incorrect stock entry',
        'miscellaneous'
    ],

    # Party preserved
    "party_items": [
        'fairy lights', 'party decorations', 'balloon', 'balloon art', 'disco ball', 'confetti', 'picnic hamper'
    ]
}

def categorize_item(description):
    """
    Categorize product based on its description.
    Uses a predefined mapping of keywords to categories.
    """
    description = str(description).lower()
    for category, keywords in category_mapping.items():
        if any(keyword in description for keyword in keywords):
            return category
    return "uncategorized"