"""
Bilingual word lookup using manual dictionaries + NLTK OMW fallback.

Provides word equivalency across languages for the translator pipeline.
"""

from typing import Optional
from ..parser.enums import Tag


# NLTK WordNet POS mapping (expanded to cover more tags)
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

# Manual bilingual dictionaries keyed by (source, target) -> {word: translation}
# English is the hub — all pairs go through English lemmas.
# Content words only; function words (determiners, particles) are handled by the linearizer.
ENGLISH_LEMMAS = {
    'spanish': {
        # Nouns — animals
        'dog': 'perro', 'cat': 'gato', 'bird': 'pájaro', 'mouse': 'ratón',
        'horse': 'caballo', 'fish': 'pez',
        # Nouns — people & family
        'man': 'hombre', 'woman': 'mujer', 'child': 'niño', 'boy': 'niño',
        'girl': 'niña', 'baby': 'bebé',
        'mother': 'madre', 'father': 'padre', 'sister': 'hermana',
        'brother': 'hermano', 'family': 'familia',
        'person': 'persona', 'people': 'gente',
        'friend': 'amigo', 'teacher': 'profesor', 'student': 'estudiante',
        'doctor': 'doctor', 'king': 'rey', 'queen': 'reina',
        # Nouns — body
        'hand': 'mano', 'head': 'cabeza', 'eye': 'ojo', 'heart': 'corazón',
        # Nouns — places & things
        'house': 'casa', 'home': 'hogar', 'room': 'habitación',
        'door': 'puerta', 'window': 'ventana', 'school': 'escuela',
        'city': 'ciudad', 'park': 'parque', 'car': 'coche',
        'table': 'mesa', 'chair': 'silla', 'book': 'libro',
        # Nouns — nature & abstract
        'tree': 'árbol', 'flower': 'flor', 'sun': 'sol', 'moon': 'luna',
        'star': 'estrella', 'water': 'agua', 'food': 'comida',
        'day': 'día', 'night': 'noche', 'time': 'tiempo',
        'name': 'nombre', 'world': 'mundo', 'life': 'vida',
        'money': 'dinero', 'work': 'trabajo', 'country': 'país',
        # Verbs (English base + 3rd person -> Spanish infinitive root)
        'run': 'correr', 'runs': 'correr',
        'eat': 'comer', 'eats': 'comer',
        'chase': 'perseguir', 'chases': 'perseguir',
        'see': 'ver', 'sees': 'ver',
        'make': 'hacer', 'makes': 'hacer',
        'have': 'tener', 'has': 'tener',
        'live': 'vivir', 'lives': 'vivir',
        'read': 'leer', 'reads': 'leer',
        'hit': 'golpear', 'hits': 'golpear',
        'love': 'amar', 'loves': 'amar',
        'want': 'querer', 'wants': 'querer',
        'know': 'saber', 'knows': 'saber',
        'think': 'pensar', 'thinks': 'pensar',
        'go': 'ir', 'goes': 'ir',
        'come': 'venir', 'comes': 'venir',
        'give': 'dar', 'gives': 'dar',
        'take': 'tomar', 'takes': 'tomar',
        'find': 'encontrar', 'finds': 'encontrar',
        'say': 'decir', 'says': 'decir',
        'tell': 'decir', 'tells': 'decir',
        'call': 'llamar', 'calls': 'llamar',
        'sing': 'cantar', 'sings': 'cantar',
        'play': 'jugar', 'plays': 'jugar',
        'write': 'escribir', 'writes': 'escribir',
        'walk': 'caminar', 'walks': 'caminar',
        'sleep': 'dormir', 'sleeps': 'dormir',
        'drink': 'beber', 'drinks': 'beber',
        'buy': 'comprar', 'buys': 'comprar',
        'open': 'abrir', 'opens': 'abrir',
        'close': 'cerrar', 'closes': 'cerrar',
        'help': 'ayudar', 'helps': 'ayudar',
        'speak': 'hablar', 'speaks': 'hablar',
        'is': 'ser', 'am': 'ser', 'are': 'ser',
        'like': 'gustar', 'likes': 'gustar',
        'need': 'necesitar', 'needs': 'necesitar',
        # Adjectives (English -> Spanish masc singular base)
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
        # Adverbs
        'quickly': 'rapidamente', 'slowly': 'lentamente', 'very': 'muy',
        'well': 'bien', 'badly': 'mal', 'always': 'siempre', 'never': 'nunca',
        # Possessives
        'my': 'mi', 'your': 'tu', 'his': 'su', 'her': 'su',
        'our': 'nuestro', 'their': 'su',
        # Demonstratives
        'this': 'este', 'that': 'ese', 'these': 'estos', 'those': 'esos',
        # Question words
        'what': 'qué', 'who': 'quién', 'where': 'dónde', 'when': 'cuándo',
        'why': 'por qué', 'how': 'cómo', 'which': 'cuál',
        # Common words OMW may miss
        'color': 'color', 'colour': 'color',
        'favorite': 'favorito', 'favourite': 'favorito',
    },
    'french': {
        # Nouns — animals
        'dog': 'chien', 'cat': 'chat', 'bird': 'oiseau', 'mouse': 'souris',
        'horse': 'cheval', 'fish': 'poisson',
        # Nouns — people & family
        'man': 'homme', 'woman': 'femme', 'child': 'enfant',
        'boy': 'garçon', 'girl': 'fille', 'baby': 'bébé',
        'mother': 'mère', 'father': 'père', 'sister': 'soeur',
        'brother': 'frère', 'family': 'famille',
        'person': 'personne', 'people': 'gens',
        'friend': 'ami', 'teacher': 'professeur', 'student': 'étudiant',
        'doctor': 'médecin', 'king': 'roi', 'queen': 'reine',
        # Nouns — body
        'hand': 'main', 'head': 'tête', 'eye': 'oeil', 'heart': 'coeur',
        # Nouns — places & things
        'house': 'maison', 'home': 'foyer', 'room': 'chambre',
        'door': 'porte', 'window': 'fenêtre', 'school': 'école',
        'city': 'ville', 'park': 'parc', 'car': 'voiture',
        'table': 'table', 'chair': 'chaise', 'book': 'livre',
        # Nouns — nature & abstract
        'tree': 'arbre', 'flower': 'fleur', 'sun': 'soleil', 'moon': 'lune',
        'star': 'étoile', 'water': 'eau', 'food': 'nourriture',
        'day': 'jour', 'night': 'nuit', 'time': 'temps',
        'name': 'nom', 'world': 'monde', 'life': 'vie',
        'money': 'argent', 'work': 'travail', 'country': 'pays',
        # Verbs
        'run': 'courir', 'runs': 'courir',
        'eat': 'manger', 'eats': 'manger',
        'chase': 'poursuivre', 'chases': 'poursuivre',
        'see': 'voir', 'sees': 'voir',
        'make': 'faire', 'makes': 'faire',
        'have': 'avoir', 'has': 'avoir',
        'live': 'vivre', 'lives': 'vivre',
        'read': 'lire', 'reads': 'lire',
        'hit': 'frapper', 'hits': 'frapper',
        'love': 'aimer', 'loves': 'aimer',
        'want': 'vouloir', 'wants': 'vouloir',
        'know': 'savoir', 'knows': 'savoir',
        'think': 'penser', 'thinks': 'penser',
        'go': 'aller', 'goes': 'aller',
        'come': 'venir', 'comes': 'venir',
        'give': 'donner', 'gives': 'donner',
        'take': 'prendre', 'takes': 'prendre',
        'find': 'trouver', 'finds': 'trouver',
        'say': 'dire', 'says': 'dire',
        'tell': 'dire', 'tells': 'dire',
        'call': 'appeler', 'calls': 'appeler',
        'sing': 'chanter', 'sings': 'chanter',
        'play': 'jouer', 'plays': 'jouer',
        'write': 'écrire', 'writes': 'écrire',
        'walk': 'marcher', 'walks': 'marcher',
        'sleep': 'dormir', 'sleeps': 'dormir',
        'drink': 'boire', 'drinks': 'boire',
        'buy': 'acheter', 'buys': 'acheter',
        'open': 'ouvrir', 'opens': 'ouvrir',
        'close': 'fermer', 'closes': 'fermer',
        'help': 'aider', 'helps': 'aider',
        'speak': 'parler', 'speaks': 'parler',
        'is': 'être', 'am': 'être', 'are': 'être',
        'like': 'aimer', 'likes': 'aimer',
        'need': 'avoir besoin', 'needs': 'avoir besoin',
        # Adjectives
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
        # Adverbs
        'quickly': 'vite', 'slowly': 'lentement', 'very': 'très',
        'well': 'bien', 'badly': 'mal', 'always': 'toujours', 'never': 'jamais',
        # Possessives
        'my': 'mon', 'your': 'ton', 'his': 'son', 'her': 'son',
        'our': 'notre', 'their': 'leur',
        # Demonstratives
        'this': 'ce', 'that': 'ce', 'these': 'ces', 'those': 'ces',
        # Question words
        'what': 'quoi', 'who': 'qui', 'where': 'où', 'when': 'quand',
        'why': 'pourquoi', 'how': 'comment', 'which': 'quel',
        # Common words OMW may miss
        'color': 'couleur', 'colour': 'couleur',
        'favorite': 'favori', 'favourite': 'favori',
    },
    'german': {
        # Nouns — animals
        'dog': 'Hund', 'cat': 'Katze', 'bird': 'Vogel', 'mouse': 'Maus',
        'horse': 'Pferd', 'fish': 'Fisch',
        # Nouns — people & family
        'man': 'Mann', 'woman': 'Frau', 'child': 'Kind',
        'boy': 'Junge', 'girl': 'Mädchen', 'baby': 'Baby',
        'mother': 'Mutter', 'father': 'Vater', 'sister': 'Schwester',
        'brother': 'Bruder', 'family': 'Familie',
        'person': 'Person', 'people': 'Leute',
        'friend': 'Freund', 'teacher': 'Lehrer', 'student': 'Student',
        'doctor': 'Arzt', 'king': 'König', 'queen': 'Königin',
        # Nouns — body
        'hand': 'Hand', 'head': 'Kopf', 'eye': 'Auge', 'heart': 'Herz',
        # Nouns — places & things
        'house': 'Haus', 'home': 'Zuhause', 'room': 'Zimmer',
        'door': 'Tür', 'window': 'Fenster', 'school': 'Schule',
        'city': 'Stadt', 'park': 'Park', 'car': 'Auto',
        'table': 'Tisch', 'chair': 'Stuhl', 'book': 'Buch',
        # Nouns — nature & abstract
        'tree': 'Baum', 'flower': 'Blume', 'sun': 'Sonne', 'moon': 'Mond',
        'star': 'Stern', 'water': 'Wasser', 'food': 'Essen',
        'day': 'Tag', 'night': 'Nacht', 'time': 'Zeit',
        'name': 'Name', 'world': 'Welt', 'life': 'Leben',
        'money': 'Geld', 'work': 'Arbeit', 'country': 'Land',
        # Verbs
        'run': 'rennen', 'runs': 'rennen',
        'eat': 'essen', 'eats': 'essen',
        'chase': 'jagen', 'chases': 'jagen',
        'see': 'sehen', 'sees': 'sehen',
        'make': 'machen', 'makes': 'machen',
        'have': 'haben', 'has': 'haben',
        'live': 'leben', 'lives': 'leben',
        'read': 'lesen', 'reads': 'lesen',
        'hit': 'schlagen', 'hits': 'schlagen',
        'love': 'lieben', 'loves': 'lieben',
        'want': 'wollen', 'wants': 'wollen',
        'know': 'wissen', 'knows': 'wissen',
        'think': 'denken', 'thinks': 'denken',
        'go': 'gehen', 'goes': 'gehen',
        'come': 'kommen', 'comes': 'kommen',
        'give': 'geben', 'gives': 'geben',
        'take': 'nehmen', 'takes': 'nehmen',
        'find': 'finden', 'finds': 'finden',
        'say': 'sagen', 'says': 'sagen',
        'tell': 'erzählen', 'tells': 'erzählen',
        'call': 'rufen', 'calls': 'rufen',
        'sing': 'singen', 'sings': 'singen',
        'play': 'spielen', 'plays': 'spielen',
        'write': 'schreiben', 'writes': 'schreiben',
        'walk': 'gehen', 'walks': 'gehen',
        'sleep': 'schlafen', 'sleeps': 'schlafen',
        'drink': 'trinken', 'drinks': 'trinken',
        'buy': 'kaufen', 'buys': 'kaufen',
        'open': 'öffnen', 'opens': 'öffnen',
        'close': 'schliessen', 'closes': 'schliessen',
        'help': 'helfen', 'helps': 'helfen',
        'speak': 'sprechen', 'speaks': 'sprechen',
        'is': 'sein', 'am': 'sein', 'are': 'sein',
        'like': 'mögen', 'likes': 'mögen',
        'need': 'brauchen', 'needs': 'brauchen',
        # Adjectives
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
        # Adverbs
        'quickly': 'schnell', 'slowly': 'langsam', 'very': 'sehr',
        'well': 'gut', 'badly': 'schlecht', 'always': 'immer', 'never': 'nie',
        # Possessives
        'my': 'mein', 'your': 'dein', 'his': 'sein', 'her': 'ihr',
        'our': 'unser', 'their': 'ihr',
        # Demonstratives
        'this': 'dieser', 'that': 'jener', 'these': 'diese', 'those': 'jene',
        # Question words
        'what': 'was', 'who': 'wer', 'where': 'wo', 'when': 'wann',
        'why': 'warum', 'how': 'wie', 'which': 'welcher',
        # Common words OMW may miss
        'color': 'Farbe', 'colour': 'Farbe',
        'favorite': 'Lieblings', 'favourite': 'Lieblings',
    },
    'portuguese': {
        # Nouns — animals
        'dog': 'cachorro', 'cat': 'gato', 'bird': 'pássaro', 'mouse': 'rato',
        'horse': 'cavalo', 'fish': 'peixe',
        # Nouns — people & family
        'man': 'homem', 'woman': 'mulher', 'child': 'criança',
        'boy': 'menino', 'girl': 'menina', 'baby': 'bebê',
        'mother': 'mãe', 'father': 'pai', 'sister': 'irmã',
        'brother': 'irmão', 'family': 'família',
        'person': 'pessoa', 'people': 'gente',
        'friend': 'amigo', 'teacher': 'professor', 'student': 'estudante',
        'doctor': 'médico', 'king': 'rei', 'queen': 'rainha',
        # Nouns — body
        'hand': 'mão', 'head': 'cabeça', 'eye': 'olho', 'heart': 'coração',
        # Nouns — places & things
        'house': 'casa', 'home': 'lar', 'room': 'quarto',
        'door': 'porta', 'window': 'janela', 'school': 'escola',
        'city': 'cidade', 'park': 'parque', 'car': 'carro',
        'table': 'mesa', 'chair': 'cadeira', 'book': 'livro',
        # Nouns — nature & abstract
        'tree': 'árvore', 'flower': 'flor', 'sun': 'sol', 'moon': 'lua',
        'star': 'estrela', 'water': 'água', 'food': 'comida',
        'day': 'dia', 'night': 'noite', 'time': 'tempo',
        'name': 'nome', 'world': 'mundo', 'life': 'vida',
        'money': 'dinheiro', 'work': 'trabalho', 'country': 'país',
        # Verbs
        'run': 'correr', 'runs': 'correr',
        'eat': 'comer', 'eats': 'comer',
        'chase': 'perseguir', 'chases': 'perseguir',
        'see': 'ver', 'sees': 'ver',
        'make': 'fazer', 'makes': 'fazer',
        'have': 'ter', 'has': 'ter',
        'live': 'viver', 'lives': 'viver',
        'read': 'ler', 'reads': 'ler',
        'hit': 'bater', 'hits': 'bater',
        'love': 'amar', 'loves': 'amar',
        'want': 'querer', 'wants': 'querer',
        'know': 'saber', 'knows': 'saber',
        'think': 'pensar', 'thinks': 'pensar',
        'go': 'ir', 'goes': 'ir',
        'come': 'vir', 'comes': 'vir',
        'give': 'dar', 'gives': 'dar',
        'take': 'pegar', 'takes': 'pegar',
        'find': 'encontrar', 'finds': 'encontrar',
        'say': 'dizer', 'says': 'dizer',
        'tell': 'dizer', 'tells': 'dizer',
        'call': 'chamar', 'calls': 'chamar',
        'sing': 'cantar', 'sings': 'cantar',
        'play': 'jogar', 'plays': 'jogar',
        'write': 'escrever', 'writes': 'escrever',
        'walk': 'andar', 'walks': 'andar',
        'sleep': 'dormir', 'sleeps': 'dormir',
        'drink': 'beber', 'drinks': 'beber',
        'buy': 'comprar', 'buys': 'comprar',
        'open': 'abrir', 'opens': 'abrir',
        'close': 'fechar', 'closes': 'fechar',
        'help': 'ajudar', 'helps': 'ajudar',
        'speak': 'falar', 'speaks': 'falar',
        'is': 'ser', 'am': 'ser', 'are': 'ser',
        'like': 'gostar', 'likes': 'gostar',
        'need': 'precisar', 'needs': 'precisar',
        # Adjectives
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
        # Adverbs
        'quickly': 'rapidamente', 'slowly': 'lentamente', 'very': 'muito',
        'well': 'bem', 'badly': 'mal', 'always': 'sempre', 'never': 'nunca',
        # Possessives
        'my': 'meu', 'your': 'teu', 'his': 'seu', 'her': 'sua',
        'our': 'nosso', 'their': 'seu',
        # Demonstratives
        'this': 'este', 'that': 'esse', 'these': 'estes', 'those': 'esses',
        # Question words
        'what': 'o que', 'who': 'quem', 'where': 'onde', 'when': 'quando',
        'why': 'por que', 'how': 'como', 'which': 'qual',
        # Common words OMW may miss
        'color': 'cor', 'colour': 'cor',
        'favorite': 'favorito', 'favourite': 'favorito',
    },
    'japanese': {
        # Nouns — animals
        'dog': 'inu', 'cat': 'neko', 'bird': 'tori', 'mouse': 'nezumi',
        'horse': 'uma', 'fish': 'sakana',
        # Nouns — people & family
        'man': 'otoko', 'woman': 'onna', 'child': 'kodomo',
        'boy': 'shounen', 'girl': 'shoujo', 'baby': 'akachan',
        'mother': 'haha', 'father': 'chichi', 'sister': 'shimai',
        'brother': 'kyoudai', 'family': 'kazoku',
        'person': 'hito', 'people': 'hitobito',
        'friend': 'tomodachi', 'teacher': 'sensei', 'student': 'gakusei',
        'doctor': 'isha', 'king': 'ou', 'queen': 'joou',
        # Nouns — body
        'hand': 'te', 'head': 'atama', 'eye': 'me', 'heart': 'kokoro',
        # Nouns — places & things
        'house': 'ie', 'home': 'ie', 'room': 'heya',
        'door': 'doa', 'window': 'mado', 'school': 'gakkou',
        'city': 'machi', 'park': 'kouen', 'car': 'kuruma',
        'table': 'tsukue', 'chair': 'isu', 'book': 'hon',
        # Nouns — nature & abstract
        'tree': 'ki', 'flower': 'hana', 'sun': 'taiyou', 'moon': 'tsuki',
        'star': 'hoshi', 'water': 'mizu', 'food': 'tabemono',
        'day': 'hi', 'night': 'yoru', 'time': 'jikan',
        'name': 'namae', 'world': 'sekai', 'life': 'inochi',
        'money': 'okane', 'work': 'shigoto', 'country': 'kuni',
        # Verbs
        'run': 'hashiru', 'runs': 'hashiru',
        'eat': 'taberu', 'eats': 'taberu',
        'chase': 'ou', 'chases': 'ou',
        'see': 'miru', 'sees': 'miru',
        'make': 'tsukuru', 'makes': 'tsukuru',
        'have': 'motsu', 'has': 'motsu',
        'live': 'ikiru', 'lives': 'ikiru',
        'read': 'yomu', 'reads': 'yomu',
        'hit': 'utsu', 'hits': 'utsu',
        'love': 'aisuru', 'loves': 'aisuru',
        'want': 'hoshii', 'wants': 'hoshii',
        'know': 'shiru', 'knows': 'shiru',
        'think': 'omou', 'thinks': 'omou',
        'go': 'iku', 'goes': 'iku',
        'come': 'kuru', 'comes': 'kuru',
        'give': 'ageru', 'gives': 'ageru',
        'take': 'toru', 'takes': 'toru',
        'find': 'mitsukeru', 'finds': 'mitsukeru',
        'say': 'iu', 'says': 'iu',
        'tell': 'iu', 'tells': 'iu',
        'call': 'yobu', 'calls': 'yobu',
        'sing': 'utau', 'sings': 'utau',
        'play': 'asobu', 'plays': 'asobu',
        'write': 'kaku', 'writes': 'kaku',
        'walk': 'aruku', 'walks': 'aruku',
        'sleep': 'neru', 'sleeps': 'neru',
        'drink': 'nomu', 'drinks': 'nomu',
        'buy': 'kau', 'buys': 'kau',
        'open': 'akeru', 'opens': 'akeru',
        'close': 'shimeru', 'closes': 'shimeru',
        'help': 'tasukeru', 'helps': 'tasukeru',
        'speak': 'hanasu', 'speaks': 'hanasu',
        'is': 'desu', 'am': 'desu', 'are': 'desu',
        'like': 'suki', 'likes': 'suki',
        'need': 'iru', 'needs': 'iru',
        # Adjectives
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
        # Adverbs
        'quickly': 'hayaku', 'slowly': 'yukkuri', 'very': 'totemo',
        'well': 'yoku', 'badly': 'waruku', 'always': 'itsumo', 'never': 'kesshite',
        # Possessives
        'my': 'watashi no', 'your': 'anata no', 'his': 'kare no', 'her': 'kanojo no',
        'our': 'watashitachi no', 'their': 'karera no',
        # Demonstratives
        'this': 'kono', 'that': 'sono', 'these': 'korera no', 'those': 'sorera no',
        # Question words
        'what': 'nani', 'who': 'dare', 'where': 'doko', 'when': 'itsu',
        'why': 'naze', 'how': 'dou', 'which': 'dore',
        # Common words OMW may miss
        'color': 'iro', 'colour': 'iro',
        'favorite': 'okiniiri', 'favourite': 'okiniiri',
    },
}


# Inflected forms -> English base form (for reverse translation)
# These map conjugated/inflected non-English words to their English equivalent.
INFLECTED_TO_ENGLISH = {
    'spanish': {
        # Verb conjugations -> English base
        'corre': 'run', 'corro': 'run', 'corren': 'run',
        'persigue': 'chase', 'persiguen': 'chase',
        'come': 'eat', 'comen': 'eat',
        've': 'see', 'ven': 'see',
        'hace': 'make', 'hacen': 'make',
        'tiene': 'have', 'tienen': 'have',
        'vive': 'live', 'viven': 'live',
        'lee': 'read', 'leen': 'read',
        'golpea': 'hit', 'golpean': 'hit',
        'ama': 'love', 'aman': 'love',
        'quiere': 'want', 'quieren': 'want',
        'sabe': 'know', 'saben': 'know',
        'piensa': 'think', 'piensan': 'think',
        'va': 'go', 'van': 'go',
        'viene': 'come', 'vienen': 'come',
        'da': 'give', 'dan': 'give',
        'toma': 'take', 'toman': 'take',
        'encuentra': 'find', 'encuentran': 'find',
        'dice': 'say', 'dicen': 'say',
        'llama': 'call', 'llaman': 'call',
        'canta': 'sing', 'cantan': 'sing',
        'juega': 'play', 'juegan': 'play',
        'escribe': 'write', 'escriben': 'write',
        'camina': 'walk', 'caminan': 'walk',
        'duerme': 'sleep', 'duermen': 'sleep',
        'bebe': 'drink', 'beben': 'drink',
        'compra': 'buy', 'compran': 'buy',
        'abre': 'open', 'abren': 'open',
        'cierra': 'close', 'cierran': 'close',
        'ayuda': 'help', 'ayudan': 'help',
        'habla': 'speak', 'hablan': 'speak',
        'es': 'is', 'son': 'are',
        'gusta': 'like',
        'necesita': 'need', 'necesitan': 'need',
        # Adjective inflected forms
        'blanca': 'white', 'blancos': 'white', 'blancas': 'white',
        'negra': 'black', 'negros': 'black', 'negras': 'black',
        'roja': 'red', 'rojos': 'red', 'rojas': 'red',
        'pequeña': 'small', 'pequeños': 'small', 'pequeñas': 'small',
        'grandes': 'big',
        'alta': 'tall', 'altos': 'tall', 'altas': 'tall',
        'baja': 'short', 'bajos': 'short', 'bajas': 'short',
    },
    'french': {
        'court': 'run', 'poursuit': 'chase', 'mange': 'eat', 'voit': 'see',
        'fait': 'make', 'a': 'have', 'vit': 'live', 'lit': 'read',
        'frappe': 'hit', 'aime': 'love', 'veut': 'want',
        'sait': 'know', 'pense': 'think',
        'va': 'go', 'vient': 'come', 'donne': 'give', 'prend': 'take',
        'trouve': 'find', 'dit': 'say', 'appelle': 'call',
        'chante': 'sing', 'joue': 'play', 'écrit': 'write',
        'marche': 'walk', 'dort': 'sleep', 'boit': 'drink',
        'achète': 'buy', 'ouvre': 'open', 'ferme': 'close',
        'aide': 'help', 'parle': 'speak',
        'est': 'is', 'sont': 'are',
        # Adjective inflected forms
        'grande': 'big', 'grands': 'big', 'grandes': 'big',
        'petite': 'small', 'petits': 'small', 'petites': 'small',
        'blanche': 'white', 'blancs': 'white', 'blanches': 'white',
    },
    'german': {
        'rennt': 'run', 'jagt': 'chase', 'isst': 'eat', 'sieht': 'see',
        'macht': 'make', 'hat': 'have', 'lebt': 'live', 'liest': 'read',
        'schlägt': 'hit', 'liebt': 'love', 'will': 'want',
        'weiss': 'know', 'denkt': 'think',
        'geht': 'go', 'kommt': 'come', 'gibt': 'give', 'nimmt': 'take',
        'findet': 'find', 'sagt': 'say', 'erzählt': 'tell',
        'ruft': 'call', 'singt': 'sing', 'spielt': 'play',
        'schreibt': 'write', 'schläft': 'sleep', 'trinkt': 'drink',
        'kauft': 'buy', 'öffnet': 'open', 'schliesst': 'close',
        'hilft': 'help', 'spricht': 'speak',
        'ist': 'is', 'sind': 'are',
        'mag': 'like', 'braucht': 'need',
        # Adjective inflected forms
        'grosse': 'big', 'grosser': 'big', 'grosses': 'big',
        'kleine': 'small', 'kleiner': 'small', 'kleines': 'small',
        'weisse': 'white', 'weisser': 'white', 'weisses': 'white',
    },
    'portuguese': {
        'corre': 'run', 'persegue': 'chase', 'come': 'eat', 'vê': 'see',
        'faz': 'make', 'tem': 'have', 'vive': 'live', 'lê': 'read',
        'bate': 'hit', 'ama': 'love', 'quer': 'want',
        'sabe': 'know', 'pensa': 'think',
        'vai': 'go', 'vem': 'come', 'dá': 'give', 'pega': 'take',
        'encontra': 'find', 'diz': 'say', 'chama': 'call',
        'canta': 'sing', 'joga': 'play', 'escreve': 'write',
        'anda': 'walk', 'dorme': 'sleep', 'bebe': 'drink',
        'compra': 'buy', 'abre': 'open', 'fecha': 'close',
        'ajuda': 'help', 'fala': 'speak',
        'é': 'is', 'são': 'are',
        'gosta': 'like', 'precisa': 'need',
        # Adjective inflected forms
        'branca': 'white', 'brancos': 'white', 'brancas': 'white',
        'alta': 'tall', 'altos': 'tall', 'altas': 'tall',
    },
    'japanese': {
        # Romaji forms (same as base in our system)
    },
}


def _build_all_dictionaries():
    """Build full bidirectional dictionary set from English hub."""
    dicts = {}

    for target_lang, en_to_target in ENGLISH_LEMMAS.items():
        # English -> target
        dicts[('english', target_lang)] = dict(en_to_target)

        # Target -> English (reverse): start with lemma mapping
        target_to_en = {}
        for en_word, target_word in en_to_target.items():
            if target_word not in target_to_en:
                target_to_en[target_word] = en_word

        # Add inflected forms
        inflected = INFLECTED_TO_ENGLISH.get(target_lang, {})
        for form, en_base in inflected.items():
            if form not in target_to_en:
                target_to_en[form] = en_base

        dicts[(target_lang, 'english')] = target_to_en

    # Build cross-language pairs via English pivot
    all_langs = list(ENGLISH_LEMMAS.keys())
    for i, lang_a in enumerate(all_langs):
        for lang_b in all_langs[i + 1:]:
            # lang_a -> lang_b: go lang_a -> English -> lang_b
            a_to_b = {}
            b_to_a = {}
            en_to_a = ENGLISH_LEMMAS[lang_a]
            en_to_b = ENGLISH_LEMMAS[lang_b]
            for en_word in en_to_a:
                if en_word in en_to_b:
                    a_to_b[en_to_a[en_word]] = en_to_b[en_word]
                    b_to_a[en_to_b[en_word]] = en_to_a[en_word]
            dicts[(lang_a, lang_b)] = a_to_b
            dicts[(lang_b, lang_a)] = b_to_a

    # Add inflected forms to cross-language pairs via English pivot.
    # e.g. French "court" → English "run" → Spanish "correr"
    for lang_a in all_langs:
        inflected_a = INFLECTED_TO_ENGLISH.get(lang_a, {})
        for lang_b in all_langs:
            if lang_a == lang_b:
                continue
            pair_key = (lang_a, lang_b)
            pair_dict = dicts.get(pair_key, {})
            en_to_b = ENGLISH_LEMMAS.get(lang_b, {})
            for inflected_form, en_base in inflected_a.items():
                if inflected_form not in pair_dict and en_base in en_to_b:
                    pair_dict[inflected_form] = en_to_b[en_base]
            dicts[pair_key] = pair_dict

    # Self-translation (identity)
    for lang in ['english'] + all_langs:
        dicts[(lang, lang)] = {}  # Will passthrough

    return dicts


_ALL_DICTS = _build_all_dictionaries()


class WordLookup:
    """Bilingual word lookup with manual dictionary + robust OMW fallback."""

    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.manual_dict = _ALL_DICTS.get((source_lang, target_lang), {})
        self._wn = None
        self._wnl = None
        self._wn_checked = False

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
        # Use NLTK lemmatizer if available
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
        1. Manual dictionary (hand-curated, most reliable)
        2. OMW with POS (WordNet + lemmatization)
        3. OMW without POS restriction (broader search)
        4. Passthrough (return original word)
        """
        if self.source_lang == self.target_lang:
            return word

        lower = word.lower()

        # 1. Manual dictionary (most reliable)
        if lower in self.manual_dict:
            return self.manual_dict[lower]
        if word in self.manual_dict:
            return self.manual_dict[word]

        # 2. NLTK OMW fallback (with POS)
        if pos and self._ensure_wordnet():
            result = self._omw_lookup(lower, pos)
            if result:
                return result

        # 3. NLTK OMW fallback (try all POS types as last resort)
        if self._ensure_wordnet():
            result = self._omw_lookup_any(lower)
            if result:
                return result

        # 4. Passthrough
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
        candidates = self._lemmatize(word, pos)

        for candidate in candidates:
            try:
                synsets = wn.synsets(candidate, pos=wn_pos, lang=source_code)
                for synset in synsets[:3]:
                    lemmas = synset.lemma_names(target_code)
                    if lemmas:
                        return lemmas[0].replace('_', ' ')
            except Exception:
                continue

        return None

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
            try:
                synsets = wn.synsets(word, pos=wn_pos, lang=source_code)
                for synset in synsets[:3]:
                    lemmas = synset.lemma_names(target_code)
                    if lemmas:
                        return lemmas[0].replace('_', ' ')
            except Exception:
                continue

        return None
