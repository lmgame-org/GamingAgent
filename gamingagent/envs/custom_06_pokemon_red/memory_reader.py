from dataclasses import dataclass
from enum import IntEnum, IntFlag


class StatusCondition(IntFlag):
    NONE = 0
    SLEEP_MASK = 0b111  # Bits 0-2
    SLEEP = 0b001  # For name display purposes
    POISON = 0b1000  # Bit 3
    BURN = 0b10000  # Bit 4
    FREEZE = 0b100000  # Bit 5
    PARALYSIS = 0b1000000  # Bit 6
    
    @property
    def is_asleep(self) -> bool:
        """Check if the Pokémon is asleep (any value in bits 0-2)"""
        # For sleep, we directly check if any bits in positions 0-2 are set (values 1-7)
        return bool(int(self) & 0b111)
    
    def get_status_name(self) -> str:
        """Get a human-readable status name"""
        if self.is_asleep:
            return "SLEEP"
        elif self & StatusCondition.PARALYSIS:
            return "PARALYSIS"
        elif self & StatusCondition.FREEZE:
            return "FREEZE"
        elif self & StatusCondition.BURN:
            return "BURN"
        elif self & StatusCondition.POISON:
            return "POISON"
        return "OK"


class Tileset(IntEnum):
    """Maps tileset IDs to their names"""

    OVERWORLD = 0x00
    REDS_HOUSE_1 = 0x01
    MART = 0x02
    FOREST = 0x03
    REDS_HOUSE_2 = 0x04
    DOJO = 0x05
    POKECENTER = 0x06
    GYM = 0x07
    HOUSE = 0x08
    FOREST_GATE = 0x09
    MUSEUM = 0x0A
    UNDERGROUND = 0x0B
    GATE = 0x0C
    SHIP = 0x0D
    SHIP_PORT = 0x0E
    CEMETERY = 0x0F
    INTERIOR = 0x10
    CAVERN = 0x11
    LOBBY = 0x12
    MANSION = 0x13
    LAB = 0x14
    CLUB = 0x15
    FACILITY = 0x16
    PLATEAU = 0x17


from enum import IntEnum


class PokemonType(IntEnum):
    NORMAL = 0x00
    FIGHTING = 0x01
    FLYING = 0x02
    POISON = 0x03
    GROUND = 0x04
    ROCK = 0x05
    BUG = 0x07
    GHOST = 0x08
    FIRE = 0x14
    WATER = 0x15
    GRASS = 0x16
    ELECTRIC = 0x17
    PSYCHIC = 0x18
    ICE = 0x19
    DRAGON = 0x1A


class Pokemon(IntEnum):
    """Maps Pokemon species IDs to their names"""

    RHYDON = 0x01
    KANGASKHAN = 0x02
    NIDORAN_M = 0x03
    CLEFAIRY = 0x04
    SPEAROW = 0x05
    VOLTORB = 0x06
    NIDOKING = 0x07
    SLOWBRO = 0x08
    IVYSAUR = 0x09
    EXEGGUTOR = 0x0A
    LICKITUNG = 0x0B
    EXEGGCUTE = 0x0C
    GRIMER = 0x0D
    GENGAR = 0x0E
    NIDORAN_F = 0x0F
    NIDOQUEEN = 0x10
    CUBONE = 0x11
    RHYHORN = 0x12
    LAPRAS = 0x13
    ARCANINE = 0x14
    MEW = 0x15
    GYARADOS = 0x16
    SHELLDER = 0x17
    TENTACOOL = 0x18
    GASTLY = 0x19
    SCYTHER = 0x1A
    STARYU = 0x1B
    BLASTOISE = 0x1C
    PINSIR = 0x1D
    TANGELA = 0x1E
    MISSINGNO_1F = 0x1F
    MISSINGNO_20 = 0x20
    GROWLITHE = 0x21
    ONIX = 0x22
    FEAROW = 0x23
    PIDGEY = 0x24
    SLOWPOKE = 0x25
    KADABRA = 0x26
    GRAVELER = 0x27
    CHANSEY = 0x28
    MACHOKE = 0x29
    MR_MIME = 0x2A
    HITMONLEE = 0x2B
    HITMONCHAN = 0x2C
    ARBOK = 0x2D
    PARASECT = 0x2E
    PSYDUCK = 0x2F
    DROWZEE = 0x30
    GOLEM = 0x31
    MISSINGNO_32 = 0x32
    MAGMAR = 0x33
    MISSINGNO_34 = 0x34
    ELECTABUZZ = 0x35
    MAGNETON = 0x36
    KOFFING = 0x37
    MISSINGNO_38 = 0x38
    MANKEY = 0x39
    SEEL = 0x3A
    DIGLETT = 0x3B
    TAUROS = 0x3C
    MISSINGNO_3D = 0x3D
    MISSINGNO_3E = 0x3E
    MISSINGNO_3F = 0x3F
    FARFETCHD = 0x40
    VENONAT = 0x41
    DRAGONITE = 0x42
    MISSINGNO_43 = 0x43
    MISSINGNO_44 = 0x44
    MISSINGNO_45 = 0x45
    DODUO = 0x46
    POLIWAG = 0x47
    JYNX = 0x48
    MOLTRES = 0x49
    ARTICUNO = 0x4A
    ZAPDOS = 0x4B
    DITTO = 0x4C
    MEOWTH = 0x4D
    KRABBY = 0x4E
    MISSINGNO_4F = 0x4F
    MISSINGNO_50 = 0x50
    MISSINGNO_51 = 0x51
    VULPIX = 0x52
    NINETALES = 0x53
    PIKACHU = 0x54
    RAICHU = 0x55
    MISSINGNO_56 = 0x56
    MISSINGNO_57 = 0x57
    DRATINI = 0x58
    DRAGONAIR = 0x59
    KABUTO = 0x5A
    KABUTOPS = 0x5B
    HORSEA = 0x5C
    SEADRA = 0x5D
    MISSINGNO_5E = 0x5E
    MISSINGNO_5F = 0x5F
    SANDSHREW = 0x60
    SANDSLASH = 0x61
    OMANYTE = 0x62
    OMASTAR = 0x63
    JIGGLYPUFF = 0x64
    WIGGLYTUFF = 0x65
    EEVEE = 0x66
    FLAREON = 0x67
    JOLTEON = 0x68
    VAPOREON = 0x69
    MACHOP = 0x6A
    ZUBAT = 0x6B
    EKANS = 0x6C
    PARAS = 0x6D
    POLIWHIRL = 0x6E
    POLIWRATH = 0x6F
    WEEDLE = 0x70
    KAKUNA = 0x71
    BEEDRILL = 0x72
    MISSINGNO_73 = 0x73
    DODRIO = 0x74
    PRIMEAPE = 0x75
    DUGTRIO = 0x76
    VENOMOTH = 0x77
    DEWGONG = 0x78
    MISSINGNO_79 = 0x79
    MISSINGNO_7A = 0x7A
    CATERPIE = 0x7B
    METAPOD = 0x7C
    BUTTERFREE = 0x7D
    MACHAMP = 0x7E
    MISSINGNO_7F = 0x7F
    GOLDUCK = 0x80
    HYPNO = 0x81
    GOLBAT = 0x82
    MEWTWO = 0x83
    SNORLAX = 0x84
    MAGIKARP = 0x85
    MISSINGNO_86 = 0x86
    MISSINGNO_87 = 0x87
    MUK = 0x88
    MISSINGNO_89 = 0x89
    KINGLER = 0x8A
    CLOYSTER = 0x8B
    MISSINGNO_8C = 0x8C
    ELECTRODE = 0x8D
    CLEFABLE = 0x8E
    WEEZING = 0x8F
    PERSIAN = 0x90
    MAROWAK = 0x91
    MISSINGNO_92 = 0x92
    HAUNTER = 0x93
    ABRA = 0x94
    ALAKAZAM = 0x95
    PIDGEOTTO = 0x96
    PIDGEOT = 0x97
    STARMIE = 0x98
    BULBASAUR = 0x99
    VENUSAUR = 0x9A
    TENTACRUEL = 0x9B
    MISSINGNO_9C = 0x9C
    GOLDEEN = 0x9D
    SEAKING = 0x9E
    MISSINGNO_9F = 0x9F
    MISSINGNO_A0 = 0xA0
    MISSINGNO_A1 = 0xA1
    MISSINGNO_A2 = 0xA2
    PONYTA = 0xA3
    RAPIDASH = 0xA4
    RATTATA = 0xA5
    RATICATE = 0xA6
    NIDORINO = 0xA7
    NIDORINA = 0xA8
    GEODUDE = 0xA9
    PORYGON = 0xAA
    AERODACTYL = 0xABAA
    MISSINGNO_AC = 0xAC
    MAGNEMITE = 0xAD
    MISSINGNO_AE = 0xAE
    MISSINGNO_AF = 0xAF
    CHARMANDER = 0xB0
    SQUIRTLE = 0xB1
    CHARMELEON = 0xB2
    WARTORTLE = 0xB3
    CHARIZARD = 0xB4
    MISSINGNO_B5 = 0xB5
    FOSSIL_KABUTOPS = 0xB6
    FOSSIL_AERODACTYL = 0xB7
    MON_GHOST = 0xB8
    ODDISH = 0xB9
    GLOOM = 0xBA
    VILEPLUME = 0xBB
    BELLSPROUT = 0xBC
    WEEPINBELL = 0xBD
    VICTREEBEL = 0xBE


class Move(IntEnum):
    """Maps move IDs to their names"""

    POUND = 0x01
    KARATE_CHOP = 0x02
    DOUBLESLAP = 0x03
    COMET_PUNCH = 0x04
    MEGA_PUNCH = 0x05
    PAY_DAY = 0x06
    FIRE_PUNCH = 0x07
    ICE_PUNCH = 0x08
    THUNDERPUNCH = 0x09
    SCRATCH = 0x0A
    VICEGRIP = 0x0B
    GUILLOTINE = 0x0C
    RAZOR_WIND = 0x0D
    SWORDS_DANCE = 0x0E
    CUT = 0x0F
    GUST = 0x10
    WING_ATTACK = 0x11
    WHIRLWIND = 0x12
    FLY = 0x13
    BIND = 0x14
    SLAM = 0x15
    VINE_WHIP = 0x16
    STOMP = 0x17
    DOUBLE_KICK = 0x18
    MEGA_KICK = 0x19
    JUMP_KICK = 0x1A
    ROLLING_KICK = 0x1B
    SAND_ATTACK = 0x1C
    HEADBUTT = 0x1D
    HORN_ATTACK = 0x1E
    FURY_ATTACK = 0x1F
    HORN_DRILL = 0x20
    TACKLE = 0x21
    BODY_SLAM = 0x22
    WRAP = 0x23
    TAKE_DOWN = 0x24
    THRASH = 0x25
    DOUBLE_EDGE = 0x26
    TAIL_WHIP = 0x27
    POISON_STING = 0x28
    TWINEEDLE = 0x29
    PIN_MISSILE = 0x2A
    LEER = 0x2B
    BITE = 0x2C
    GROWL = 0x2D
    ROAR = 0x2E
    SING = 0x2F
    SUPERSONIC = 0x30
    SONICBOOM = 0x31
    DISABLE = 0x32
    ACID = 0x33
    EMBER = 0x34
    FLAMETHROWER = 0x35
    MIST = 0x36
    WATER_GUN = 0x37
    HYDRO_PUMP = 0x38
    SURF = 0x39
    ICE_BEAM = 0x3A
    BLIZZARD = 0x3B
    PSYBEAM = 0x3C
    BUBBLEBEAM = 0x3D
    AURORA_BEAM = 0x3E
    HYPER_BEAM = 0x3F
    PECK = 0x40
    DRILL_PECK = 0x41
    SUBMISSION = 0x42
    LOW_KICK = 0x43
    COUNTER = 0x44
    SEISMIC_TOSS = 0x45
    STRENGTH = 0x46
    ABSORB = 0x47
    MEGA_DRAIN = 0x48
    LEECH_SEED = 0x49
    GROWTH = 0x4A
    RAZOR_LEAF = 0x4B
    SOLARBEAM = 0x4C
    POISONPOWDER = 0x4D
    STUN_SPORE = 0x4E
    SLEEP_POWDER = 0x4F
    PETAL_DANCE = 0x50
    STRING_SHOT = 0x51
    DRAGON_RAGE = 0x52
    FIRE_SPIN = 0x53
    THUNDERSHOCK = 0x54
    THUNDERBOLT = 0x55
    THUNDER_WAVE = 0x56
    THUNDER = 0x57
    ROCK_THROW = 0x58
    EARTHQUAKE = 0x59
    FISSURE = 0x5A
    DIG = 0x5B
    TOXIC = 0x5C
    CONFUSION = 0x5D
    PSYCHIC = 0x5E
    HYPNOSIS = 0x5F
    MEDITATE = 0x60
    AGILITY = 0x61
    QUICK_ATTACK = 0x62
    RAGE = 0x63
    TELEPORT = 0x64
    NIGHT_SHADE = 0x65
    MIMIC = 0x66
    SCREECH = 0x67
    DOUBLE_TEAM = 0x68
    RECOVER = 0x69
    HARDEN = 0x6A
    MINIMIZE = 0x6B
    SMOKESCREEN = 0x6C
    CONFUSE_RAY = 0x6D
    WITHDRAW = 0x6E
    DEFENSE_CURL = 0x6F
    BARRIER = 0x70
    LIGHT_SCREEN = 0x71
    HAZE = 0x72
    REFLECT = 0x73
    FOCUS_ENERGY = 0x74
    BIDE = 0x75
    METRONOME = 0x76
    MIRROR_MOVE = 0x77
    SELFDESTRUCT = 0x78
    EGG_BOMB = 0x79
    LICK = 0x7A
    SMOG = 0x7B
    SLUDGE = 0x7C
    BONE_CLUB = 0x7D
    FIRE_BLAST = 0x7E
    WATERFALL = 0x7F
    CLAMP = 0x80
    SWIFT = 0x81
    SKULL_BASH = 0x82
    SPIKE_CANNON = 0x83
    CONSTRICT = 0x84
    AMNESIA = 0x85
    KINESIS = 0x86
    SOFTBOILED = 0x87
    HI_JUMP_KICK = 0x88
    GLARE = 0x89
    DREAM_EATER = 0x8A
    POISON_GAS = 0x8B
    BARRAGE = 0x8C
    LEECH_LIFE = 0x8D
    LOVELY_KISS = 0x8E
    SKY_ATTACK = 0x8F
    TRANSFORM = 0x90
    BUBBLE = 0x91
    DIZZY_PUNCH = 0x92
    SPORE = 0x93
    FLASH = 0x94
    PSYWAVE = 0x95
    SPLASH = 0x96
    ACID_ARMOR = 0x97
    CRABHAMMER = 0x98
    EXPLOSION = 0x99
    FURY_SWIPES = 0x9A
    BONEMERANG = 0x9B
    REST = 0x9C
    ROCK_SLIDE = 0x9D
    HYPER_FANG = 0x9E
    SHARPEN = 0x9F
    CONVERSION = 0xA0
    TRI_ATTACK = 0xA1
    SUPER_FANG = 0xA2
    SLASH = 0xA3
    SUBSTITUTE = 0xA4
    STRUGGLE = 0xA5


class MapLocation(IntEnum):
    """Maps location IDs to their names"""

    PALLET_TOWN = 0x00
    VIRIDIAN_CITY = 0x01
    PEWTER_CITY = 0x02
    CERULEAN_CITY = 0x03
    LAVENDER_TOWN = 0x04
    VERMILION_CITY = 0x05
    CELADON_CITY = 0x06
    FUCHSIA_CITY = 0x07
    CINNABAR_ISLAND = 0x08
    INDIGO_PLATEAU = 0x09
    SAFFRON_CITY = 0x0A
    UNUSED_0B = 0x0B
    ROUTE_1 = 0x0C
    ROUTE_2 = 0x0D
    ROUTE_3 = 0x0E
    ROUTE_4 = 0x0F
    ROUTE_5 = 0x10
    ROUTE_6 = 0x11
    ROUTE_7 = 0x12
    ROUTE_8 = 0x13
    ROUTE_9 = 0x14
    ROUTE_10 = 0x15
    ROUTE_11 = 0x16
    ROUTE_12 = 0x17
    ROUTE_13 = 0x18
    ROUTE_14 = 0x19
    ROUTE_15 = 0x1A
    ROUTE_16 = 0x1B
    ROUTE_17 = 0x1C
    ROUTE_18 = 0x1D
    ROUTE_19 = 0x1E
    ROUTE_20 = 0x1F
    ROUTE_21 = 0x20
    ROUTE_22 = 0x21
    ROUTE_23 = 0x22
    ROUTE_24 = 0x23
    ROUTE_25 = 0x24
    PLAYERS_HOUSE_1F = 0x25
    PLAYERS_HOUSE_2F = 0x26
    RIVALS_HOUSE = 0x27
    OAKS_LAB = 0x28
    VIRIDIAN_POKECENTER = 0x29
    VIRIDIAN_MART = 0x2A
    VIRIDIAN_SCHOOL = 0x2B
    VIRIDIAN_HOUSE = 0x2C
    VIRIDIAN_GYM = 0x2D
    DIGLETTS_CAVE_ROUTE2 = 0x2E
    VIRIDIAN_FOREST_NORTH_GATE = 0x2F
    ROUTE_2_HOUSE = 0x30
    ROUTE_2_GATE = 0x31
    VIRIDIAN_FOREST_SOUTH_GATE = 0x32
    VIRIDIAN_FOREST = 0x33
    MUSEUM_1F = 0x34
    MUSEUM_2F = 0x35
    PEWTER_GYM = 0x36
    PEWTER_HOUSE_1 = 0x37
    PEWTER_MART = 0x38
    PEWTER_HOUSE_2 = 0x39
    PEWTER_POKECENTER = 0x3A
    MT_MOON_1F = 0x3B
    MT_MOON_B1F = 0x3C
    MT_MOON_B2F = 0x3D
    CERULEAN_TRASHED_HOUSE = 0x3E
    CERULEAN_TRADE_HOUSE = 0x3F
    CERULEAN_POKECENTER = 0x40
    CERULEAN_GYM = 0x41
    BIKE_SHOP = 0x42
    CERULEAN_MART = 0x43
    MT_MOON_POKECENTER = 0x44
    ROUTE_5_GATE = 0x46
    UNDERGROUND_PATH_ROUTE5 = 0x47
    DAYCARE = 0x48
    ROUTE_6_GATE = 0x49
    UNDERGROUND_PATH_ROUTE6 = 0x4A
    ROUTE_7_GATE = 0x4C
    UNDERGROUND_PATH_ROUTE7 = 0x4D
    ROUTE_8_GATE = 0x4F
    UNDERGROUND_PATH_ROUTE8 = 0x50
    ROCK_TUNNEL_POKECENTER = 0x51
    ROCK_TUNNEL_1F = 0x52
    POWER_PLANT = 0x53
    ROUTE_11_GATE_1F = 0x54
    DIGLETTS_CAVE_ROUTE11 = 0x55
    ROUTE_11_GATE_2F = 0x56
    ROUTE_12_GATE_1F = 0x57
    BILLS_HOUSE = 0x58
    VERMILION_POKECENTER = 0x59
    FAN_CLUB = 0x5A
    VERMILION_MART = 0x5B
    VERMILION_GYM = 0x5C
    VERMILION_HOUSE_1 = 0x5D
    VERMILION_DOCK = 0x5E
    SS_ANNE_1F = 0x5F
    SS_ANNE_2F = 0x60
    SS_ANNE_3F = 0x61
    SS_ANNE_B1F = 0x62
    SS_ANNE_BOW = 0x63
    SS_ANNE_KITCHEN = 0x64
    SS_ANNE_CAPTAINS_ROOM = 0x65
    SS_ANNE_1F_ROOMS = 0x66
    SS_ANNE_2F_ROOMS = 0x67
    SS_ANNE_B1F_ROOMS = 0x68
    VICTORY_ROAD_1F = 0x6C
    LANCE = 0x71
    HALL_OF_FAME = 0x76
    UNDERGROUND_PATH_NS = 0x77
    CHAMPIONS_ROOM = 0x78
    UNDERGROUND_PATH_WE = 0x79
    CELADON_MART_1F = 0x7A
    CELADON_MART_2F = 0x7B
    CELADON_MART_3F = 0x7C
    CELADON_MART_4F = 0x7D
    CELADON_MART_ROOF = 0x7E
    CELADON_MART_ELEVATOR = 0x7F
    CELADON_MANSION_1F = 0x80
    CELADON_MANSION_2F = 0x81
    CELADON_MANSION_3F = 0x82
    CELADON_MANSION_ROOF = 0x83
    CELADON_MANSION_ROOF_HOUSE = 0x84
    CELADON_POKECENTER = 0x85
    CELADON_GYM = 0x86
    GAME_CORNER = 0x87
    CELADON_MART_5F = 0x88
    GAME_CORNER_PRIZE_ROOM = 0x89
    CELADON_DINER = 0x8A
    CELADON_HOUSE = 0x8B
    CELADON_HOTEL = 0x8C
    LAVENDER_POKECENTER = 0x8D
    POKEMON_TOWER_1F = 0x8E
    POKEMON_TOWER_2F = 0x8F
    POKEMON_TOWER_3F = 0x90
    POKEMON_TOWER_4F = 0x91
    POKEMON_TOWER_5F = 0x92
    POKEMON_TOWER_6F = 0x93
    POKEMON_TOWER_7F = 0x94
    LAVENDER_HOUSE_1 = 0x95
    LAVENDER_MART = 0x96
    LAVENDER_HOUSE_2 = 0x97
    FUCHSIA_MART = 0x98
    FUCHSIA_HOUSE_1 = 0x99
    FUCHSIA_POKECENTER = 0x9A
    FUCHSIA_HOUSE_2 = 0x9B
    SAFARI_ZONE_ENTRANCE = 0x9C
    FUCHSIA_GYM = 0x9D
    FUCHSIA_MEETING_ROOM = 0x9E
    SEAFOAM_ISLANDS_B1F = 0x9F
    SEAFOAM_ISLANDS_B2F = 0xA0
    SEAFOAM_ISLANDS_B3F = 0xA1
    SEAFOAM_ISLANDS_B4F = 0xA2
    VERMILION_HOUSE_2 = 0xA3
    VERMILION_HOUSE_3 = 0xA4
    POKEMON_MANSION_1F = 0xA5
    CINNABAR_GYM = 0xA6
    CINNABAR_LAB_1 = 0xA7
    CINNABAR_LAB_2 = 0xA8
    CINNABAR_LAB_3 = 0xA9
    CINNABAR_LAB_4 = 0xAA
    CINNABAR_POKECENTER = 0xAB
    CINNABAR_MART = 0xAC
    INDIGO_PLATEAU_LOBBY = 0xAE
    COPYCATS_HOUSE_1F = 0xAF
    COPYCATS_HOUSE_2F = 0xB0
    FIGHTING_DOJO = 0xB1
    SAFFRON_GYM = 0xB2
    SAFFRON_HOUSE_1 = 0xB3
    SAFFRON_MART = 0xB4
    SILPH_CO_1F = 0xB5
    SAFFRON_POKECENTER = 0xB6
    SAFFRON_HOUSE_2 = 0xB7
    ROUTE_15_GATE_1F = 0xB8
    ROUTE_15_GATE_2F = 0xB9
    ROUTE_16_GATE_1F = 0xBA
    ROUTE_16_GATE_2F = 0xBB
    ROUTE_16_HOUSE = 0xBC
    ROUTE_12_HOUSE = 0xBD
    ROUTE_18_GATE_1F = 0xBE
    ROUTE_18_GATE_2F = 0xBF
    SEAFOAM_ISLANDS_1F = 0xC0
    ROUTE_22_GATE = 0xC1
    VICTORY_ROAD_2F = 0xC2
    ROUTE_12_GATE_2F = 0xC3
    VERMILION_HOUSE_4 = 0xC4
    DIGLETTS_CAVE = 0xC5
    VICTORY_ROAD_3F = 0xC6
    ROCKET_HIDEOUT_B1F = 0xC7
    ROCKET_HIDEOUT_B2F = 0xC8
    ROCKET_HIDEOUT_B3F = 0xC9
    ROCKET_HIDEOUT_B4F = 0xCA
    ROCKET_HIDEOUT_ELEVATOR = 0xCB
    SILPH_CO_2F = 0xCF
    SILPH_CO_3F = 0xD0
    SILPH_CO_4F = 0xD1
    SILPH_CO_5F = 0xD2
    SILPH_CO_6F = 0xD3
    SILPH_CO_7F = 0xD4
    SILPH_CO_8F = 0xD5
    POKEMON_MANSION_2F = 0xD6
    POKEMON_MANSION_3F = 0xD7
    POKEMON_MANSION_B1F = 0xD8
    SAFARI_ZONE_EAST = 0xD9
    SAFARI_ZONE_NORTH = 0xDA
    SAFARI_ZONE_WEST = 0xDB
    SAFARI_ZONE_CENTER = 0xDC
    SAFARI_ZONE_CENTER_REST_HOUSE = 0xDD
    SAFARI_ZONE_SECRET_HOUSE = 0xDE
    SAFARI_ZONE_WEST_REST_HOUSE = 0xDF
    SAFARI_ZONE_EAST_REST_HOUSE = 0xE0
    SAFARI_ZONE_NORTH_REST_HOUSE = 0xE1
    CERULEAN_CAVE_2F = 0xE2
    CERULEAN_CAVE_B1F = 0xE3
    CERULEAN_CAVE_1F = 0xE4
    NAME_RATERS_HOUSE = 0xE5
    CERULEAN_BADGE_HOUSE = 0xE6
    ROCK_TUNNEL_B1F = 0xE8
    SILPH_CO_9F = 0xE9
    SILPH_CO_10F = 0xEA
    SILPH_CO_11F = 0xEB
    SILPH_CO_ELEVATOR = 0xEC
    TRADE_CENTER = 0xEF
    COLOSSEUM = 0xF0
    LORELEI = 0xF5
    BRUNO = 0xF6
    AGATHA = 0xF7


class Badge(IntFlag):
    """Flags for gym badges"""

    BOULDER = 1 << 0
    CASCADE = 1 << 1
    THUNDER = 1 << 2
    RAINBOW = 1 << 3
    SOUL = 1 << 4
    MARSH = 1 << 5
    VOLCANO = 1 << 6
    EARTH = 1 << 7


@dataclass
class PokemonData:
    """Complete Pokemon data structure"""

    species_id: int
    species_name: str
    current_hp: int
    max_hp: int
    level: int
    status: StatusCondition
    type1: PokemonType
    type2: PokemonType | None
    moves: list[str]  # Move names
    move_pp: list[int]  # PP for each move
    trainer_id: int
    nickname: str | None = None
    experience: int | None = None
    
    @property
    def is_asleep(self) -> bool:
        """Check if the Pokémon is asleep"""
        return self.status.is_asleep
        
    @property
    def status_name(self) -> str:
        """Return a human-readable status name"""
        if self.is_asleep:
            return "SLEEP"
        elif self.status & StatusCondition.PARALYSIS:
            return "PARALYSIS"
        elif self.status & StatusCondition.FREEZE:
            return "FREEZE"
        elif self.status & StatusCondition.BURN:
            return "BURN"
        elif self.status & StatusCondition.POISON:
            return "POISON"
        else:
            return "OK"


class PokemonRedReader:
    """Reads and interprets memory values from Pokemon Red"""

    def __init__(self, memory_view):
        """Initialize with a PyBoy memory view object"""
        self.memory = memory_view

    def get_warps(self) -> list[tuple[int, int]]:
        """Get all the warps listed for the current map.
        If necessary we can also get where these warps go, but later.
        
        Not sure this is all warps, but it works for Viridian Forest. Best effort. Also I can't figure out how to determine which _direction_ the warp is.
        """
        # This is the number of warps
        num_warps = self.memory[0xD3AE]
        # Each warp now comes in groups of 4, row col of location on this map and row col of where it goes in the _next_ map
        warps = []  # this could be a set, but for something like this let's keep it simpler.
        for n in range(num_warps):  # turning this to col, row
            warps.append((
                self.memory[0xD3AF + n * 4 + 1],
                self.memory[0xD3AF + n * 4]
            ))
        return warps

    def read_in_combat(self) -> bool:
        """Are we in combat?"""
        b = self.memory[0xD057]
        return bool(b)

    def read_money(self) -> int:
        """Read the player's money in Binary Coded Decimal format"""
        b1 = self.memory[0xD349]  # Least significant byte
        b2 = self.memory[0xD348]  # Middle byte
        b3 = self.memory[0xD347]  # Most significant byte
        money = (
            ((b3 >> 4) * 100000)
            + ((b3 & 0xF) * 10000)
            + ((b2 >> 4) * 1000)
            + ((b2 & 0xF) * 100)
            + ((b1 >> 4) * 10)
            + (b1 & 0xF)
        )
        return money

    def _convert_text(self, bytes_data: list[int]) -> str:
        """Convert Pokemon text format to ASCII"""
        result = ""
        for b in bytes_data:
            if b == 0x50:  # End marker
                break
            elif b == 0x4E:  # Line break
                result += "\n"
            # Main character ranges
            elif 0x80 <= b <= 0x99:  # A-Z
                result += chr(b - 0x80 + ord("A"))
            elif 0xA0 <= b <= 0xB9:  # a-z
                result += chr(b - 0xA0 + ord("a"))
            elif 0xF6 <= b <= 0xFF:  # Numbers 0-9
                result += str(b - 0xF6)
            # Punctuation characters (9A-9F)
            elif b == 0x9A:  # (
                result += "("
            elif b == 0x9B:  # )
                result += ")"
            elif b == 0x9C:  # :
                result += ":"
            elif b == 0x9D:  # ;
                result += ";"
            elif b == 0x9E:  # [
                result += "["
            elif b == 0x9F:  # ]
                result += "]"
            # Special characters
            elif b == 0x7F:  # Space
                result += " "
            elif b == 0x6D:  # : (also appears here)
                result += ":"
            elif b == 0x54:  # POKé control character
                result += "POKé"
            elif b == 0xBA:  # é
                result += "é"
            elif b == 0xBB:  # 'd
                result += "'d"
            elif b == 0xBC:  # 'l
                result += "'l"
            elif b == 0xBD:  # 's
                result += "'s"
            elif b == 0xBE:  # 't
                result += "'t"
            elif b == 0xBF:  # 'v
                result += "'v"
            elif b == 0xE1:  # PK
                result += "Pk"
            elif b == 0xE2:  # MN
                result += "Mn"
            elif b == 0xE3:  # -
                result += "-"
            elif b == 0xE6:  # ?
                result += "?"
            elif b == 0xE7:  # !
                result += "!"
            elif b == 0xE8:  # .
                result += "."
            elif b == 0xE9:  # .
                result += "."
            # E-register special characters
            elif b == 0xE0:  # '
                result += "'"
            elif b == 0xE1:  # PK
                result += "POKé"
            elif b == 0xE2:  # MN
                result += "MON"
            elif b == 0xE3:  # -
                result += "-"
            elif b == 0xE4:  # 'r
                result += "'r"
            elif b == 0xE5:  # 'm
                result += "'m"
            elif b == 0xE6:  # ?
                result += "?"
            elif b == 0xE7:  # !
                result += "!"
            elif b == 0xE8:  # .
                result += "."
            elif b == 0xE9:  # ア
                result += "ア"
            elif b == 0xEA:  # ウ
                result += "ウ"
            elif b == 0xEB:  # エ
                result += "エ"
            elif b == 0xEC:  # ▷
                result += "▷"
            elif b == 0xED:  # ►
                result += "►"
            elif b == 0xEE:  # ▼
                result += "▼"
            elif b == 0xEF:  # ♂
                result += "♂"
            # F-register special characters
            elif b == 0xF0:  # ♭
                result += "♭"
            elif b == 0xF1:  # ×
                result += "×"
            elif b == 0xF2:  # .
                result += "."
            elif b == 0xF3:  # /
                result += "/"
            elif b == 0xF4:  # ,
                result += ","
            elif b == 0xF5:  # ♀
                result += "♀"
            # Numbers 0-9 (0xF6-0xFF)
            elif 0xF6 <= b <= 0xFF:
                result += str(b - 0xF6)
            else:
                # For debugging, show the hex value of unknown characters
                result += f"[{b:02X}]"
        return result.strip()

    def read_player_name(self) -> str:
        """Read the player's name"""
        name_bytes = self.memory[0xD158:0xD163]
        return self._convert_text(name_bytes)

    def read_rival_name(self) -> str:
        """Read rival's name"""
        name_bytes = self.memory[0xD34A:0xD351]
        return self._convert_text(name_bytes)

    def read_badges(self) -> list[str]:
        """Read obtained badges as list of names"""
        badge_byte = self.memory[0xD356]
        badges = []

        if badge_byte & Badge.BOULDER:
            badges.append("BOULDER")
        if badge_byte & Badge.CASCADE:
            badges.append("CASCADE")
        if badge_byte & Badge.THUNDER:
            badges.append("THUNDER")
        if badge_byte & Badge.RAINBOW:
            badges.append("RAINBOW")
        if badge_byte & Badge.SOUL:
            badges.append("SOUL")
        if badge_byte & Badge.MARSH:
            badges.append("MARSH")
        if badge_byte & Badge.VOLCANO:
            badges.append("VOLCANO")
        if badge_byte & Badge.EARTH:
            badges.append("EARTH")

        return badges

    def read_party_size(self) -> int:
        """Read number of Pokemon in party"""
        return self.memory[0xD163]

    def read_party_pokemon(self) -> list[PokemonData]:
        """Read all Pokemon currently in the party with full data"""
        party = []
        party_size = self.read_party_size()

        # Base addresses for party Pokemon data
        base_addresses = [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]
        nickname_addresses = [0xD2B5, 0xD2C0, 0xD2CB, 0xD2D6, 0xD2E1, 0xD2EC]

        for i in range(party_size):
            addr = base_addresses[i]

            # Read experience (3 bytes)
            exp = (
                (self.memory[addr + 0x1A] << 16)
                + (self.memory[addr + 0x1B] << 8)
                + self.memory[addr + 0x1C]
            )

            # Read moves and PP
            moves = []
            move_pp = []
            for j in range(4):
                move_id = self.memory[addr + 8 + j]
                if move_id != 0:
                    moves.append(Move(move_id).name.replace("_", " "))
                    move_pp.append(self.memory[addr + 0x1D + j])

            # Read nickname
            nickname = self._convert_text(
                self.memory[nickname_addresses[i] : nickname_addresses[i] + 11]
            )

            type1 = PokemonType(self.memory[addr + 5])
            type2 = PokemonType(self.memory[addr + 6])
            # If both types are the same, only show one type
            if type1 == type2:
                type2 = None

            try:
                species_id = self.memory[addr]
                species_name = Pokemon(species_id).name.replace("_", " ")
            except ValueError:
                continue
            status_value = self.memory[addr + 4]
            
            pokemon = PokemonData(
                species_id=self.memory[addr],
                species_name=species_name,
                current_hp=(self.memory[addr + 1] << 8) + self.memory[addr + 2],
                max_hp=(self.memory[addr + 0x22] << 8) + self.memory[addr + 0x23],
                level=self.memory[addr + 0x21],  # Using actual level
                status=StatusCondition(status_value),
                type1=type1,
                type2=type2,
                moves=moves,
                move_pp=move_pp,
                trainer_id=(self.memory[addr + 12] << 8) + self.memory[addr + 13],
                nickname=nickname,
                experience=exp,
            )
            party.append(pokemon)

        return party

    def read_game_time(self) -> tuple[int, int, int]:
        """Read game time as (hours, minutes, seconds)"""
        hours = (self.memory[0xDA40] << 8) + self.memory[0xDA41]
        minutes = self.memory[0xDA42]
        seconds = self.memory[0xDA44]
        return (hours, minutes, seconds)

    def read_location(self) -> str:
        """Read current location name"""
        map_id = self.memory[0xD35E]
        return MapLocation(map_id).name.replace("_", " ")

    def read_tileset(self) -> str:
        """Read current map's tileset name"""
        tileset_id = self.memory[0xD367]
        return Tileset(tileset_id).name.replace("_", " ")

    def read_coordinates(self) -> tuple[int, int]:
        """Read player's current X,Y coordinates"""
        return (self.memory[0xD362], self.memory[0xD361])

    def read_coins(self) -> int:
        """Read game corner coins"""
        return (self.memory[0xD5A4] << 8) + self.memory[0xD5A5]

    def read_item_count(self) -> int:
        """Read number of items in inventory"""
        return self.memory[0xD31D]

    def read_items(self) -> list[tuple[str, int]]:
        """Read all items in inventory with proper item names"""
        # Revised mapping based on the game's internal item numbering
        ITEM_NAMES = {
            0x01: "MASTER BALL",
            0x02: "ULTRA BALL",
            0x03: "GREAT BALL",
            0x04: "POKé BALL",
            0x05: "TOWN MAP",
            0x06: "BICYCLE",
            0x07: "???",
            0x08: "SAFARI BALL",
            0x09: "POKéDEX",
            0x0A: "MOON STONE",
            0x0B: "ANTIDOTE",
            0x0C: "BURN HEAL",
            0x0D: "ICE HEAL",
            0x0E: "AWAKENING",
            0x0F: "PARLYZ HEAL",
            0x10: "FULL RESTORE",
            0x11: "MAX POTION",
            0x12: "HYPER POTION",
            0x13: "SUPER POTION",
            0x14: "POTION",
            # Badges 0x15-0x1C
            0x1D: "ESCAPE ROPE",
            0x1E: "REPEL",
            0x1F: "OLD AMBER",
            0x20: "FIRE STONE",
            0x21: "THUNDERSTONE",
            0x22: "WATER STONE",
            0x23: "HP UP",
            0x24: "PROTEIN",
            0x25: "IRON",
            0x26: "CARBOS",
            0x27: "CALCIUM",
            0x28: "RARE CANDY",
            0x29: "DOME FOSSIL",
            0x2A: "HELIX FOSSIL",
            0x2B: "SECRET KEY",
            0x2C: "???",  # Blank item
            0x2D: "BIKE VOUCHER",
            0x2E: "X ACCURACY",
            0x2F: "LEAF STONE",
            0x30: "CARD KEY",
            0x31: "NUGGET",
            0x32: "PP UP",
            0x33: "POKé DOLL",
            0x34: "FULL HEAL",
            0x35: "REVIVE",
            0x36: "MAX REVIVE",
            0x37: "GUARD SPEC",
            0x38: "SUPER REPEL",
            0x39: "MAX REPEL",
            0x3A: "DIRE HIT",
            0x3B: "COIN",
            0x3C: "FRESH WATER",
            0x3D: "SODA POP",
            0x3E: "LEMONADE",
            0x3F: "S.S. TICKET",
            0x40: "GOLD TEETH",
            0x41: "X ATTACK",
            0x42: "X DEFEND",
            0x43: "X SPEED",
            0x44: "X SPECIAL",
            0x45: "COIN CASE",
            0x46: "OAK's PARCEL",
            0x47: "ITEMFINDER",
            0x48: "SILPH SCOPE",
            0x49: "POKé FLUTE",
            0x4A: "LIFT KEY",
            0x4B: "EXP.ALL",
            0x4C: "OLD ROD",
            0x4D: "GOOD ROD",
            0x4E: "SUPER ROD",
            0x4F: "PP UP",
            0x50: "ETHER",
            0x51: "MAX ETHER",
            0x52: "ELIXER",
            0x53: "MAX ELIXER",
        }

        items = []
        count = self.read_item_count()

        for i in range(count):
            item_id = self.memory[0xD31E + (i * 2)]
            quantity = self.memory[0xD31F + (i * 2)]

            # Handle TMs (0xC9-0xFE)
            if 0xC9 <= item_id <= 0xFE:
                tm_num = item_id - 0xC8
                item_name = f"TM{tm_num:02d}"
            elif 0xC4 <= item_id <= 0xC8:
                hm_num = item_id - 0xC3
                item_name = f"HM{hm_num:02d}"
            elif item_id in ITEM_NAMES:
                item_name = ITEM_NAMES[item_id]
            else:
                item_name = f"UNKNOWN_{item_id:02X}"

            items.append((item_name, quantity))

        return items

    def read_dialog(self) -> str:
        """Read any dialog text currently on screen by scanning the tilemap buffer"""
        # Tilemap buffer is from C3A0 to C507
        buffer_start = 0xC3A0
        buffer_end = 0xC507

        # Get all bytes from the buffer
        buffer_bytes = [self.memory[addr] for addr in range(buffer_start, buffer_end)]

        # Look for sequences of text (ignoring long sequences of 0x7F/spaces)
        text_lines = []
        current_line = []
        space_count = 0
        last_was_border = False

        for b in buffer_bytes:
            if b == 0x7C:  # ║ character
                if last_was_border:
                    # If the last character was a border and this is ║, treat as newline
                    text = self._convert_text(current_line)
                    if text.strip():
                        text_lines.append(text)
                    current_line = []
                    space_count = 0
                else:
                    # current_line.append(b)
                    pass
                last_was_border = True
            elif b == 0x7F:  # Space
                space_count += 1
                current_line.append(b)  # Always keep spaces
                last_was_border = False
            # All text characters: uppercase, lowercase, special chars, punctuation, symbols
            elif (
                # Box drawing (0x79-0x7E)
                # (0x79 <= b <= 0x7E)
                # or
                # Uppercase (0x80-0x99)
                (0x80 <= b <= 0x99)
                or
                # Punctuation (0x9A-0x9F)
                (0x9A <= b <= 0x9F)
                or
                # Lowercase (0xA0-0xB9)
                (0xA0 <= b <= 0xB9)
                or
                # Contractions (0xBA-0xBF)
                (0xBA <= b <= 0xBF)
                or
                # Special characters in E-row (0xE0-0xEF)
                (0xE0 <= b <= 0xEF)
                or
                # Special characters in F-row (0xF0-0xF5)
                (0xF0 <= b <= 0xF5)
                or
                # Numbers (0xF6-0xFF)
                (0xF6 <= b <= 0xFF)
                or
                # Line break
                b == 0x4E
            ):
                space_count = 0
                current_line.append(b)
                last_was_border = (
                    0x79 <= b <= 0x7E
                )  # Track if this is a border character

            # If we see a lot of spaces, might be end of line
            if space_count > 10 and current_line:
                text = self._convert_text(current_line)
                if text.strip():  # Only add non-empty lines
                    text_lines.append(text)
                current_line = []
                space_count = 0
                last_was_border = False

        # Add final line if any
        if current_line:
            text = self._convert_text(current_line)
            if text.strip():
                text_lines.append(text)

        text = "\n".join(text_lines)

        # Post-process for name entry context
        if "lower case" in text.lower() or "UPPER CASE" in text:
            # We're in name entry, replace ♭ with ED
            text = text.replace("♭", "ED\n")

        return text

    def read_pokedex_caught_count(self) -> int:
        """Read how many unique Pokemon species have been caught"""
        # Pokedex owned flags are stored in D2F7-D309
        # Each byte contains 8 flags for 8 Pokemon
        # Total of 19 bytes = 152 Pokemon
        caught_count = 0
        for addr in range(0xD2F7, 0xD30A):
            byte = self.memory[addr]
            # Count set bits in this byte
            caught_count += bin(byte).count("1")
        return caught_count

    def read_pokedex_seen_count(self) -> int:
        """Read how many unique Pokemon species have been seen"""
        # Pokedex seen flags are stored in D30A-D31C  
        # Each byte contains 8 flags for 8 Pokemon
        seen_count = 0
        for addr in range(0xD30A, 0xD31D):
            byte = self.memory[addr]
            # Count set bits in this byte
            seen_count += bin(byte).count("1")
        return seen_count

    def read_current_menu_selection(self) -> int:
        """Read the currently selected menu item"""
        return self.memory[0xCC26]

    def read_current_map_id(self) -> int:
        """Read the current map ID"""
        return self.memory[0xD35E]

    def read_facing_direction(self) -> str:
        """Read which direction the player is facing"""
        direction = self.memory[0xC109]
        directions = {0: "down", 4: "up", 8: "left", 12: "right"}
        return directions.get(direction, "unknown")

    def read_steps_taken(self) -> int:
        """Read total steps taken by player"""
        # Steps are stored as 3-byte value
        return (self.memory[0xDA53] << 16) + (self.memory[0xDA54] << 8) + self.memory[0xDA55]

    def read_safari_steps_remaining(self) -> int:
        """Read remaining steps in Safari Zone"""
        return (self.memory[0xDA47] << 8) + self.memory[0xDA48]

    def read_safari_balls_remaining(self) -> int:
        """Read remaining Safari Balls"""
        return self.memory[0xDA46]

    def is_in_safari_zone(self) -> bool:
        """Check if player is currently in Safari Zone"""
        location_id = self.read_current_map_id()
        safari_zones = [0xD9, 0xDA, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF, 0xE0, 0xE1]  # Safari Zone map IDs
        return location_id in safari_zones

    def read_pc_box_count(self) -> int:
        """Read number of Pokemon in current PC box"""
        return self.memory[0xDA80]

    def read_daycare_pokemon_count(self) -> int:
        """Read number of Pokemon in daycare (0 or 1)"""
        return 1 if self.memory[0xD2C7] != 0 else 0

    def read_battle_type(self) -> str:
        """Read current battle type"""
        battle_type = self.memory[0xD05A]
        types = {
            0: "wild",
            1: "trainer", 
            2: "old_man_tutorial"
        }
        return types.get(battle_type, "unknown")

    def read_enemy_pokemon_level(self) -> int:
        """Read level of enemy Pokemon in battle"""
        return self.memory[0xCFE8] if self.read_in_combat() else 0

    def read_last_repel_used(self) -> str:
        """Read the last repel item used"""
        repel_id = self.memory[0xD72E]
        repels = {0x1E: "REPEL", 0x38: "SUPER REPEL", 0x39: "MAX REPEL"}
        return repels.get(repel_id, "NONE")

    def read_repel_steps_remaining(self) -> int:
        """Read remaining steps for active repel"""
        return self.memory[0xD72D]