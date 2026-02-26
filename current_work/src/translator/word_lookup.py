"""
Bilingual word lookup using content word dictionaries + NLTK OMW fallback.

Architecture:
- Function words (possessives, demonstratives, questions): manual dict (OMW can't handle)
- Content words (nouns, verbs, adjectives, adverbs): manual dict for common words,
  OMW for anything not in the dict. Lemma-to-lemma only (no inflected forms —
  morphology.py handles inflection).
- Inflected reverse lookups: slim table of irregular forms only.
"""

from typing import Optional
from ..parser.enums import Tag


# NLTK WordNet POS mapping
_TAG_TO_WN_POS = {
    Tag.NOUN: 'n',
    Tag.VERB: 'v',
    Tag.AUX: 'v',
    Tag.ADJ: 'a',
    Tag.ADV: 'r',
    Tag.PRON: 'n',
    Tag.PROPN: 'n',
}

# OMW language codes
OMW_CODES = {
    'english': 'eng',
    'spanish': 'spa',
    'french': 'fra',
    'german': 'deu',
    'portuguese': 'por',
    'japanese': 'jpn',
}

# ============================================================================
# English -> target language dictionaries (lemma-to-lemma only)
#
# Each entry maps an English lemma to the target language lemma.
# Inflection (conjugation, gender agreement) is handled by morphology.py.
# No inflected forms (e.g. 'runs' -> 'correr') — only base forms.
# ============================================================================

_ENGLISH_TO = {
    'spanish': {
        # --- Function words (OMW can't handle these) ---
        'my': 'mi', 'your': 'tu', 'his': 'su', 'her': 'su',
        'our': 'nuestro', 'their': 'su',
        'this': 'este', 'that': 'ese', 'these': 'estos', 'those': 'esos',
        'what': 'qué', 'who': 'quién', 'where': 'dónde', 'when': 'cuándo',
        'why': 'por qué', 'how': 'cómo', 'which': 'cuál',
        # --- Copula / auxiliaries ---
        'is': 'ser', 'am': 'ser', 'are': 'ser',
        # --- Nouns ---
        'dog': 'perro', 'cat': 'gato', 'bird': 'pájaro', 'mouse': 'ratón',
        'horse': 'caballo', 'fish': 'pez',
        'man': 'hombre', 'woman': 'mujer', 'child': 'niño', 'boy': 'niño',
        'girl': 'niña', 'baby': 'bebé',
        'mother': 'madre', 'father': 'padre', 'sister': 'hermana',
        'brother': 'hermano', 'family': 'familia',
        'person': 'persona', 'people': 'gente',
        'friend': 'amigo', 'teacher': 'profesor', 'student': 'estudiante',
        'doctor': 'doctor', 'king': 'rey', 'queen': 'reina',
        'hand': 'mano', 'head': 'cabeza', 'eye': 'ojo', 'heart': 'corazón',
        'house': 'casa', 'home': 'hogar', 'room': 'habitación',
        'door': 'puerta', 'window': 'ventana', 'school': 'escuela',
        'city': 'ciudad', 'park': 'parque', 'car': 'coche',
        'table': 'mesa', 'chair': 'silla', 'book': 'libro',
        'tree': 'árbol', 'flower': 'flor', 'sun': 'sol', 'moon': 'luna',
        'star': 'estrella', 'water': 'agua', 'food': 'comida',
        'day': 'día', 'night': 'noche', 'time': 'tiempo',
        'name': 'nombre', 'world': 'mundo', 'life': 'vida',
        'money': 'dinero', 'work': 'trabajo', 'country': 'país',
        'color': 'color', 'colour': 'color',
        'favorite': 'favorito', 'favourite': 'favorito',
        # --- Verbs (base form only) ---
        'run': 'correr', 'eat': 'comer', 'chase': 'perseguir',
        'see': 'ver', 'make': 'hacer', 'have': 'tener',
        'live': 'vivir', 'read': 'leer', 'hit': 'golpear',
        'love': 'amar', 'want': 'querer', 'know': 'saber',
        'think': 'pensar', 'go': 'ir', 'come': 'venir',
        'give': 'dar', 'take': 'tomar', 'find': 'encontrar',
        'say': 'decir', 'tell': 'decir', 'call': 'llamar',
        'sing': 'cantar', 'play': 'jugar', 'write': 'escribir',
        'walk': 'caminar', 'sleep': 'dormir', 'drink': 'beber',
        'buy': 'comprar', 'open': 'abrir', 'close': 'cerrar',
        'help': 'ayudar', 'speak': 'hablar',
        'like': 'gustar', 'need': 'necesitar',
        # --- Adjectives ---
        'big': 'grande', 'small': 'pequeño',
        'white': 'blanco', 'black': 'negro',
        'red': 'rojo', 'blue': 'azul', 'green': 'verde',
        'good': 'bueno', 'bad': 'malo',
        'old': 'viejo', 'new': 'nuevo',
        'beautiful': 'bonito', 'ugly': 'feo',
        'tall': 'alto', 'short': 'bajo',
        'happy': 'feliz', 'sad': 'triste',
        'fast': 'rápido', 'slow': 'lento',
        'young': 'joven', 'strong': 'fuerte', 'weak': 'débil',
        'pretty': 'bonito', 'rich': 'rico', 'poor': 'pobre',
        'hot': 'caliente', 'cold': 'frío',
        'long': 'largo', 'important': 'importante',
        # --- Adverbs ---
        'quickly': 'rapidamente', 'slowly': 'lentamente', 'very': 'muy',
        'well': 'bien', 'badly': 'mal', 'always': 'siempre', 'never': 'nunca',
    },
    'french': {
        'my': 'mon', 'your': 'ton', 'his': 'son', 'her': 'son',
        'our': 'notre', 'their': 'leur',
        'this': 'ce', 'that': 'ce', 'these': 'ces', 'those': 'ces',
        'what': 'quoi', 'who': 'qui', 'where': 'où', 'when': 'quand',
        'why': 'pourquoi', 'how': 'comment', 'which': 'quel',
        'is': 'être', 'am': 'être', 'are': 'être',
        'dog': 'chien', 'cat': 'chat', 'bird': 'oiseau', 'mouse': 'souris',
        'horse': 'cheval', 'fish': 'poisson',
        'man': 'homme', 'woman': 'femme', 'child': 'enfant',
        'boy': 'garçon', 'girl': 'fille', 'baby': 'bébé',
        'mother': 'mère', 'father': 'père', 'sister': 'soeur',
        'brother': 'frère', 'family': 'famille',
        'person': 'personne', 'people': 'gens',
        'friend': 'ami', 'teacher': 'professeur', 'student': 'étudiant',
        'doctor': 'médecin', 'king': 'roi', 'queen': 'reine',
        'hand': 'main', 'head': 'tête', 'eye': 'oeil', 'heart': 'coeur',
        'house': 'maison', 'home': 'foyer', 'room': 'chambre',
        'door': 'porte', 'window': 'fenêtre', 'school': 'école',
        'city': 'ville', 'park': 'parc', 'car': 'voiture',
        'table': 'table', 'chair': 'chaise', 'book': 'livre',
        'tree': 'arbre', 'flower': 'fleur', 'sun': 'soleil', 'moon': 'lune',
        'star': 'étoile', 'water': 'eau', 'food': 'nourriture',
        'day': 'jour', 'night': 'nuit', 'time': 'temps',
        'name': 'nom', 'world': 'monde', 'life': 'vie',
        'money': 'argent', 'work': 'travail', 'country': 'pays',
        'color': 'couleur', 'colour': 'couleur',
        'favorite': 'favori', 'favourite': 'favori',
        'run': 'courir', 'eat': 'manger', 'chase': 'poursuivre',
        'see': 'voir', 'make': 'faire', 'have': 'avoir',
        'live': 'vivre', 'read': 'lire', 'hit': 'frapper',
        'love': 'aimer', 'want': 'vouloir', 'know': 'savoir',
        'think': 'penser', 'go': 'aller', 'come': 'venir',
        'give': 'donner', 'take': 'prendre', 'find': 'trouver',
        'say': 'dire', 'tell': 'dire', 'call': 'appeler',
        'sing': 'chanter', 'play': 'jouer', 'write': 'écrire',
        'walk': 'marcher', 'sleep': 'dormir', 'drink': 'boire',
        'buy': 'acheter', 'open': 'ouvrir', 'close': 'fermer',
        'help': 'aider', 'speak': 'parler',
        'like': 'aimer', 'need': 'avoir besoin',
        'big': 'grand', 'small': 'petit',
        'white': 'blanc', 'black': 'noir',
        'red': 'rouge', 'blue': 'bleu', 'green': 'vert',
        'good': 'bon', 'bad': 'mauvais',
        'old': 'vieux', 'new': 'nouveau',
        'beautiful': 'beau', 'ugly': 'laid',
        'tall': 'grand', 'short': 'court',
        'happy': 'heureux', 'sad': 'triste',
        'fast': 'rapide', 'slow': 'lent',
        'young': 'jeune', 'strong': 'fort', 'weak': 'faible',
        'pretty': 'joli', 'rich': 'riche', 'poor': 'pauvre',
        'hot': 'chaud', 'cold': 'froid',
        'long': 'long', 'important': 'important',
        'quickly': 'vite', 'slowly': 'lentement', 'very': 'très',
        'well': 'bien', 'badly': 'mal', 'always': 'toujours', 'never': 'jamais',
    },
    'german': {
        'my': 'mein', 'your': 'dein', 'his': 'sein', 'her': 'ihr',
        'our': 'unser', 'their': 'ihr',
        'this': 'dieser', 'that': 'jener', 'these': 'diese', 'those': 'jene',
        'what': 'was', 'who': 'wer', 'where': 'wo', 'when': 'wann',
        'why': 'warum', 'how': 'wie', 'which': 'welcher',
        'is': 'sein', 'am': 'sein', 'are': 'sein',
        'dog': 'Hund', 'cat': 'Katze', 'bird': 'Vogel', 'mouse': 'Maus',
        'horse': 'Pferd', 'fish': 'Fisch',
        'man': 'Mann', 'woman': 'Frau', 'child': 'Kind',
        'boy': 'Junge', 'girl': 'Mädchen', 'baby': 'Baby',
        'mother': 'Mutter', 'father': 'Vater', 'sister': 'Schwester',
        'brother': 'Bruder', 'family': 'Familie',
        'person': 'Person', 'people': 'Leute',
        'friend': 'Freund', 'teacher': 'Lehrer', 'student': 'Student',
        'doctor': 'Arzt', 'king': 'König', 'queen': 'Königin',
        'hand': 'Hand', 'head': 'Kopf', 'eye': 'Auge', 'heart': 'Herz',
        'house': 'Haus', 'home': 'Zuhause', 'room': 'Zimmer',
        'door': 'Tür', 'window': 'Fenster', 'school': 'Schule',
        'city': 'Stadt', 'park': 'Park', 'car': 'Auto',
        'table': 'Tisch', 'chair': 'Stuhl', 'book': 'Buch',
        'tree': 'Baum', 'flower': 'Blume', 'sun': 'Sonne', 'moon': 'Mond',
        'star': 'Stern', 'water': 'Wasser', 'food': 'Essen',
        'day': 'Tag', 'night': 'Nacht', 'time': 'Zeit',
        'name': 'Name', 'world': 'Welt', 'life': 'Leben',
        'money': 'Geld', 'work': 'Arbeit', 'country': 'Land',
        'color': 'Farbe', 'colour': 'Farbe',
        'favorite': 'Lieblings', 'favourite': 'Lieblings',
        'run': 'rennen', 'eat': 'essen', 'chase': 'jagen',
        'see': 'sehen', 'make': 'machen', 'have': 'haben',
        'live': 'leben', 'read': 'lesen', 'hit': 'schlagen',
        'love': 'lieben', 'want': 'wollen', 'know': 'wissen',
        'think': 'denken', 'go': 'gehen', 'come': 'kommen',
        'give': 'geben', 'take': 'nehmen', 'find': 'finden',
        'say': 'sagen', 'tell': 'erzählen', 'call': 'rufen',
        'sing': 'singen', 'play': 'spielen', 'write': 'schreiben',
        'walk': 'gehen', 'sleep': 'schlafen', 'drink': 'trinken',
        'buy': 'kaufen', 'open': 'öffnen', 'close': 'schliessen',
        'help': 'helfen', 'speak': 'sprechen',
        'like': 'mögen', 'need': 'brauchen',
        'big': 'gross', 'small': 'klein',
        'white': 'weiss', 'black': 'schwarz',
        'red': 'rot', 'blue': 'blau', 'green': 'grün',
        'good': 'gut', 'bad': 'schlecht',
        'old': 'alt', 'new': 'neu',
        'beautiful': 'schön', 'ugly': 'hässlich',
        'tall': 'gross', 'short': 'kurz',
        'happy': 'glücklich', 'sad': 'traurig',
        'fast': 'schnell', 'slow': 'langsam',
        'young': 'jung', 'strong': 'stark', 'weak': 'schwach',
        'pretty': 'hübsch', 'rich': 'reich', 'poor': 'arm',
        'hot': 'heiss', 'cold': 'kalt',
        'long': 'lang', 'important': 'wichtig',
        'quickly': 'schnell', 'slowly': 'langsam', 'very': 'sehr',
        'well': 'gut', 'badly': 'schlecht', 'always': 'immer', 'never': 'nie',
    },
    'portuguese': {
        'my': 'meu', 'your': 'teu', 'his': 'seu', 'her': 'sua',
        'our': 'nosso', 'their': 'seu',
        'this': 'este', 'that': 'esse', 'these': 'estes', 'those': 'esses',
        'what': 'o que', 'who': 'quem', 'where': 'onde', 'when': 'quando',
        'why': 'por que', 'how': 'como', 'which': 'qual',
        'is': 'ser', 'am': 'ser', 'are': 'ser',
        'dog': 'cachorro', 'cat': 'gato', 'bird': 'pássaro', 'mouse': 'rato',
        'horse': 'cavalo', 'fish': 'peixe',
        'man': 'homem', 'woman': 'mulher', 'child': 'criança',
        'boy': 'menino', 'girl': 'menina', 'baby': 'bebê',
        'mother': 'mãe', 'father': 'pai', 'sister': 'irmã',
        'brother': 'irmão', 'family': 'família',
        'person': 'pessoa', 'people': 'gente',
        'friend': 'amigo', 'teacher': 'professor', 'student': 'estudante',
        'doctor': 'médico', 'king': 'rei', 'queen': 'rainha',
        'hand': 'mão', 'head': 'cabeça', 'eye': 'olho', 'heart': 'coração',
        'house': 'casa', 'home': 'lar', 'room': 'quarto',
        'door': 'porta', 'window': 'janela', 'school': 'escola',
        'city': 'cidade', 'park': 'parque', 'car': 'carro',
        'table': 'mesa', 'chair': 'cadeira', 'book': 'livro',
        'tree': 'árvore', 'flower': 'flor', 'sun': 'sol', 'moon': 'lua',
        'star': 'estrela', 'water': 'água', 'food': 'comida',
        'day': 'dia', 'night': 'noite', 'time': 'tempo',
        'name': 'nome', 'world': 'mundo', 'life': 'vida',
        'money': 'dinheiro', 'work': 'trabalho', 'country': 'país',
        'color': 'cor', 'colour': 'cor',
        'favorite': 'favorito', 'favourite': 'favorito',
        'run': 'correr', 'eat': 'comer', 'chase': 'perseguir',
        'see': 'ver', 'make': 'fazer', 'have': 'ter',
        'live': 'viver', 'read': 'ler', 'hit': 'bater',
        'love': 'amar', 'want': 'querer', 'know': 'saber',
        'think': 'pensar', 'go': 'ir', 'come': 'vir',
        'give': 'dar', 'take': 'pegar', 'find': 'encontrar',
        'say': 'dizer', 'tell': 'dizer', 'call': 'chamar',
        'sing': 'cantar', 'play': 'jogar', 'write': 'escrever',
        'walk': 'andar', 'sleep': 'dormir', 'drink': 'beber',
        'buy': 'comprar', 'open': 'abrir', 'close': 'fechar',
        'help': 'ajudar', 'speak': 'falar',
        'like': 'gostar', 'need': 'precisar',
        'big': 'grande', 'small': 'pequeno',
        'white': 'branco', 'black': 'preto',
        'red': 'vermelho', 'blue': 'azul', 'green': 'verde',
        'good': 'bom', 'bad': 'mau',
        'old': 'velho', 'new': 'novo',
        'beautiful': 'bonito', 'ugly': 'feio',
        'tall': 'alto', 'short': 'baixo',
        'happy': 'feliz', 'sad': 'triste',
        'fast': 'rápido', 'slow': 'lento',
        'young': 'jovem', 'strong': 'forte', 'weak': 'fraco',
        'pretty': 'bonito', 'rich': 'rico', 'poor': 'pobre',
        'hot': 'quente', 'cold': 'frio',
        'long': 'longo', 'important': 'importante',
        'quickly': 'rapidamente', 'slowly': 'lentamente', 'very': 'muito',
        'well': 'bem', 'badly': 'mal', 'always': 'sempre', 'never': 'nunca',
    },
    'japanese': {
        'my': 'watashi no', 'your': 'anata no', 'his': 'kare no',
        'her': 'kanojo no', 'our': 'watashitachi no', 'their': 'karera no',
        'this': 'kono', 'that': 'sono', 'these': 'korera no',
        'those': 'sorera no',
        'what': 'nani', 'who': 'dare', 'where': 'doko', 'when': 'itsu',
        'why': 'naze', 'how': 'dou', 'which': 'dore',
        'is': 'desu', 'am': 'desu', 'are': 'desu',
        'like': 'suki', 'need': 'iru',
        'favorite': 'okiniiri', 'favourite': 'okiniiri',
        'dog': 'inu', 'cat': 'neko', 'bird': 'tori', 'mouse': 'nezumi',
        'horse': 'uma', 'fish': 'sakana',
        'man': 'otoko', 'woman': 'onna', 'child': 'kodomo',
        'boy': 'shounen', 'girl': 'shoujo', 'baby': 'akachan',
        'mother': 'haha', 'father': 'chichi', 'sister': 'shimai',
        'brother': 'kyoudai', 'family': 'kazoku',
        'person': 'hito', 'people': 'hitobito',
        'friend': 'tomodachi', 'teacher': 'sensei', 'student': 'gakusei',
        'doctor': 'isha', 'king': 'ou', 'queen': 'joou',
        'hand': 'te', 'head': 'atama', 'eye': 'me', 'heart': 'kokoro',
        'house': 'ie', 'home': 'ie', 'room': 'heya',
        'door': 'doa', 'window': 'mado', 'school': 'gakkou',
        'city': 'machi', 'park': 'kouen', 'car': 'kuruma',
        'table': 'tsukue', 'chair': 'isu', 'book': 'hon',
        'tree': 'ki', 'flower': 'hana', 'sun': 'taiyou', 'moon': 'tsuki',
        'star': 'hoshi', 'water': 'mizu', 'food': 'tabemono',
        'day': 'hi', 'night': 'yoru', 'time': 'jikan',
        'name': 'namae', 'world': 'sekai', 'life': 'inochi',
        'money': 'okane', 'work': 'shigoto', 'country': 'kuni',
        'color': 'iro', 'colour': 'iro',
        'run': 'hashiru', 'eat': 'taberu', 'chase': 'ou',
        'see': 'miru', 'make': 'tsukuru', 'have': 'motsu',
        'live': 'ikiru', 'read': 'yomu', 'hit': 'utsu',
        'love': 'aisuru', 'want': 'hoshii', 'know': 'shiru',
        'think': 'omou', 'go': 'iku', 'come': 'kuru',
        'give': 'ageru', 'take': 'toru', 'find': 'mitsukeru',
        'say': 'iu', 'tell': 'iu', 'call': 'yobu',
        'sing': 'utau', 'play': 'asobu', 'write': 'kaku',
        'walk': 'aruku', 'sleep': 'neru', 'drink': 'nomu',
        'buy': 'kau', 'open': 'akeru', 'close': 'shimeru',
        'help': 'tasukeru', 'speak': 'hanasu',
        'big': 'ooki', 'small': 'chiisai',
        'white': 'shiroi', 'black': 'kuroi',
        'red': 'akai', 'blue': 'aoi', 'green': 'midori',
        'good': 'yoi', 'bad': 'warui',
        'old': 'furui', 'new': 'atarashii',
        'beautiful': 'utsukushii', 'ugly': 'minikui',
        'tall': 'takai', 'short': 'hikui',
        'happy': 'ureshii', 'sad': 'kanashii',
        'fast': 'hayai', 'slow': 'osoi',
        'young': 'wakai', 'strong': 'tsuyoi', 'weak': 'yowai',
        'pretty': 'kawaii', 'rich': 'kanemochi', 'poor': 'mazushii',
        'hot': 'atsui', 'cold': 'tsumetai',
        'long': 'nagai', 'important': 'taisetsu',
        'quickly': 'hayaku', 'slowly': 'yukkuri', 'very': 'totemo',
        'well': 'yoku', 'badly': 'waruku', 'always': 'itsumo',
        'never': 'kesshite',
    },
}

# Inflected forms -> English base (only irregular forms that can't be lemmatized)
_INFLECTED_TO_ENGLISH = {
    'spanish': {
        'es': 'is', 'son': 'are', 'soy': 'am',
        'va': 'go', 'van': 'go',
        'dice': 'say', 'dicen': 'say',
        'tiene': 'have', 'tienen': 'have',
        'hace': 'make', 'hacen': 'make',
        've': 'see', 'ven': 'see',
        'da': 'give', 'dan': 'give',
        'viene': 'come', 'vienen': 'come',
        'sabe': 'know', 'saben': 'know',
        'quiere': 'want', 'quieren': 'want',
        'piensa': 'think', 'piensan': 'think',
        'duerme': 'sleep', 'duermen': 'sleep',
        'encuentra': 'find', 'encuentran': 'find',
        'persigue': 'chase', 'persiguen': 'chase',
        'juega': 'play', 'juegan': 'play',
        'cierra': 'close', 'cierran': 'close',
        'gusta': 'like',
        # Regular forms that OMW / lemmatization may miss
        'corre': 'run', 'corren': 'run',
        'come': 'eat', 'comen': 'eat',
        'golpea': 'hit', 'golpean': 'hit',
        'ama': 'love', 'aman': 'love',
        'camina': 'walk', 'caminan': 'walk',
        'habla': 'speak', 'hablan': 'speak',
        'necesita': 'need', 'necesitan': 'need',
    },
    'french': {
        'est': 'is', 'sont': 'are',
        'va': 'go', 'fait': 'make', 'a': 'have',
        'voit': 'see', 'veut': 'want', 'sait': 'know',
        'dit': 'say', 'prend': 'take', 'boit': 'drink',
        'court': 'run', 'dort': 'sleep', 'lit': 'read',
        'vit': 'live', 'vient': 'come',
    },
    'german': {
        'ist': 'is', 'sind': 'are',
        'hat': 'have', 'isst': 'eat', 'sieht': 'see',
        'liest': 'read', 'gibt': 'give', 'nimmt': 'take',
        'spricht': 'speak', 'hilft': 'help',
        'schlägt': 'hit', 'schläft': 'sleep',
        'mag': 'like', 'will': 'want', 'weiss': 'know',
        'rennt': 'run', 'jagt': 'chase',
    },
    'portuguese': {
        'é': 'is', 'são': 'are',
        'vai': 'go', 'faz': 'make', 'tem': 'have',
        'vê': 'see', 'diz': 'say', 'lê': 'read',
        'quer': 'want', 'vem': 'come', 'dá': 'give',
        'corre': 'run',
    },
    'japanese': {},
}

# Map English inflected forms to base (for source-side lemmatization)
_ENGLISH_INFLECTED = {
    'runs': 'run', 'eats': 'eat', 'chases': 'chase', 'sees': 'see',
    'makes': 'make', 'has': 'have', 'lives': 'live', 'reads': 'read',
    'hits': 'hit', 'loves': 'love', 'wants': 'want', 'knows': 'know',
    'thinks': 'think', 'goes': 'go', 'comes': 'come', 'gives': 'give',
    'takes': 'take', 'finds': 'find', 'says': 'say', 'tells': 'tell',
    'calls': 'call', 'sings': 'sing', 'plays': 'play', 'writes': 'write',
    'walks': 'walk', 'sleeps': 'sleep', 'drinks': 'drink', 'buys': 'buy',
    'opens': 'open', 'closes': 'close', 'helps': 'help', 'speaks': 'speak',
    'likes': 'like', 'needs': 'need',
}


def _build_all_dictionaries():
    """Build full bidirectional dictionary set from English hub."""
    dicts = {}

    for target_lang, en_to_target in _ENGLISH_TO.items():
        # English -> target
        dicts[('english', target_lang)] = dict(en_to_target)

        # Target -> English (reverse): start with lemma mapping
        target_to_en = {}
        for en_word, target_word in en_to_target.items():
            if target_word not in target_to_en:
                target_to_en[target_word] = en_word

        # Add inflected forms
        inflected = _INFLECTED_TO_ENGLISH.get(target_lang, {})
        for form, en_base in inflected.items():
            if form not in target_to_en:
                target_to_en[form] = en_base

        dicts[(target_lang, 'english')] = target_to_en

    # Build cross-language pairs via English pivot
    all_langs = list(_ENGLISH_TO.keys())
    for i, lang_a in enumerate(all_langs):
        for lang_b in all_langs[i + 1:]:
            a_to_b = {}
            b_to_a = {}
            en_to_a = _ENGLISH_TO[lang_a]
            en_to_b = _ENGLISH_TO[lang_b]
            for en_word in en_to_a:
                if en_word in en_to_b:
                    a_to_b[en_to_a[en_word]] = en_to_b[en_word]
                    b_to_a[en_to_b[en_word]] = en_to_a[en_word]
            dicts[(lang_a, lang_b)] = a_to_b
            dicts[(lang_b, lang_a)] = b_to_a

    # Add inflected reverse forms to cross-language pairs
    for lang_a in all_langs:
        inflected_a = _INFLECTED_TO_ENGLISH.get(lang_a, {})
        for lang_b in all_langs:
            if lang_a == lang_b:
                continue
            pair_key = (lang_a, lang_b)
            pair_dict = dicts.get(pair_key, {})
            en_to_b = _ENGLISH_TO.get(lang_b, {})
            for inflected_form, en_base in inflected_a.items():
                if inflected_form not in pair_dict and en_base in en_to_b:
                    pair_dict[inflected_form] = en_to_b[en_base]
            dicts[pair_key] = pair_dict

    # Self-translation (identity)
    for lang in ['english'] + all_langs:
        dicts[(lang, lang)] = {}

    return dicts


_ALL_DICTS = _build_all_dictionaries()


class WordLookup:
    """Bilingual word lookup: dict for common words + OMW for unlimited vocab."""

    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.manual_dict = _ALL_DICTS.get((source_lang, target_lang), {})
        self._wn = None
        self._wnl = None
        self._wn_checked = False
        self._cache = {}

    def _ensure_wordnet(self):
        """Lazy-load NLTK WordNet and lemmatizer."""
        if self._wn_checked:
            return self._wn is not None
        self._wn_checked = True
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets('dog', lang='eng')
            self._wn = wn
            try:
                from nltk.stem import WordNetLemmatizer
                self._wnl = WordNetLemmatizer()
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _lemmatize(self, word: str, pos: Tag) -> list:
        """Generate lemma candidates for OMW lookup."""
        candidates = [word]
        if self._wnl:
            try:
                wn_pos = _TAG_TO_WN_POS.get(pos, 'n')
                lemma = self._wnl.lemmatize(word, wn_pos)
                if lemma != word and lemma not in candidates:
                    candidates.append(lemma)
            except Exception:
                pass
        # Manual suffix stripping as fallback
        if pos in (Tag.NOUN, Tag.PRON, Tag.PROPN):
            if word.endswith('ies') and len(word) > 4:
                candidates.append(word[:-3] + 'y')
            elif word.endswith('es') and len(word) > 3:
                candidates.append(word[:-2])
            elif word.endswith('s') and len(word) > 2:
                candidates.append(word[:-1])
        elif pos in (Tag.VERB, Tag.AUX):
            for suffix, cut, add in [
                ('ing', 3, ''), ('ing', 3, 'e'),
                ('ed', 2, ''), ('ed', 2, 'e'),
                ('ies', 3, 'y'), ('es', 2, ''), ('s', 1, ''),
            ]:
                if word.endswith(suffix) and len(word) > len(suffix) + 1:
                    c = word[:-cut] + add
                    if c not in candidates:
                        candidates.append(c)
        return candidates

    def lookup(self, word: str, pos: Tag = None) -> str:
        """
        Find target language equivalent for a source word.

        Lookup chain:
        1. Manual dictionary (lemma-to-lemma, handles common words + function words)
        2. English inflection -> base form -> dict (handles 'runs' -> 'run' -> 'correr')
        3. OMW with POS + lemmatization (handles words not in dict)
        4. OMW without POS restriction (broader search)
        5. Passthrough (return original word)
        """
        if self.source_lang == self.target_lang:
            return word

        lower = word.lower()

        # 1. Direct dict lookup
        if lower in self.manual_dict:
            return self.manual_dict[lower]
        if word in self.manual_dict:
            return self.manual_dict[word]

        # 2. English inflection resolution (runs -> run -> correr)
        if self.source_lang == 'english' and lower in _ENGLISH_INFLECTED:
            base = _ENGLISH_INFLECTED[lower]
            if base in self.manual_dict:
                return self.manual_dict[base]

        # 3. OMW with POS
        if pos and self._ensure_wordnet():
            result = self._omw_lookup(lower, pos)
            if result:
                return result

        # 4. OMW without POS (broader search)
        if self._ensure_wordnet():
            result = self._omw_lookup_any(lower)
            if result:
                return result

        # 5. Passthrough
        return word

    def _omw_lookup(self, word: str, pos: Tag) -> Optional[str]:
        """OMW lookup with lemmatization — tries multiple candidate forms."""
        wn = self._wn
        if wn is None:
            return None

        wn_pos = _TAG_TO_WN_POS.get(pos)
        if not wn_pos:
            return None

        target_code = OMW_CODES.get(self.target_lang)
        if not target_code:
            return None

        source_code = OMW_CODES.get(self.source_lang, 'eng')

        cache_key = (word, wn_pos, source_code, target_code)
        if cache_key in self._cache:
            return self._cache[cache_key]

        candidates = self._lemmatize(word, pos)
        result = None

        for candidate in candidates:
            try:
                synsets = wn.synsets(candidate, pos=wn_pos, lang=source_code)
                for synset in synsets[:3]:
                    lemmas = synset.lemma_names(target_code)
                    if lemmas:
                        result = lemmas[0].replace('_', ' ')
                        break
            except Exception:
                continue
            if result:
                break

        self._cache[cache_key] = result
        return result

    def _omw_lookup_any(self, word: str) -> Optional[str]:
        """OMW lookup trying all POS types — last resort before passthrough."""
        wn = self._wn
        if wn is None:
            return None

        target_code = OMW_CODES.get(self.target_lang)
        if not target_code:
            return None

        source_code = OMW_CODES.get(self.source_lang, 'eng')

        for wn_pos in ('n', 'v', 'a', 'r'):
            cache_key = (word, wn_pos, source_code, target_code)
            if cache_key in self._cache:
                if self._cache[cache_key] is not None:
                    return self._cache[cache_key]
                continue

            try:
                synsets = wn.synsets(word, pos=wn_pos, lang=source_code)
                for synset in synsets[:3]:
                    lemmas = synset.lemma_names(target_code)
                    if lemmas:
                        result = lemmas[0].replace('_', ' ')
                        self._cache[cache_key] = result
                        return result
            except Exception:
                continue
            self._cache[cache_key] = None

        return None
