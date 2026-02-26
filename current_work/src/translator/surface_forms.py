"""
Surface form selection — find the correct inflected form for a target language word.

Given a lemma + required morphological features (gender, number, person),
returns the correctly inflected surface form.
"""

from typing import List, Optional, Dict, Tuple, FrozenSet
from ..parser.enums import SubType, Tag


# ============================================================================
# Verb conjugation tables: lemma -> {(person, number): surface_form}
# ============================================================================

SPANISH_VERBS = {
    'correr': {
        (SubType.FIRST_PERSON, SubType.SINGULAR): 'corro',
        (SubType.SECOND_PERSON, SubType.SINGULAR): 'corres',
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'corre',
        (SubType.FIRST_PERSON, SubType.PLURAL): 'corremos',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'corren',
    },
    'comer': {
        (SubType.FIRST_PERSON, SubType.SINGULAR): 'como',
        (SubType.SECOND_PERSON, SubType.SINGULAR): 'comes',
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'come',
        (SubType.FIRST_PERSON, SubType.PLURAL): 'comemos',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'comen',
    },
    'perseguir': {
        (SubType.FIRST_PERSON, SubType.SINGULAR): 'persigo',
        (SubType.SECOND_PERSON, SubType.SINGULAR): 'persigues',
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'persigue',
        (SubType.FIRST_PERSON, SubType.PLURAL): 'perseguimos',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'persiguen',
    },
    'ver': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 've',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'ven',
    },
    'hacer': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'hace',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'hacen',
    },
    'tener': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'tiene',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'tienen',
    },
    'vivir': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'vive',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'viven',
    },
    'leer': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'lee',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'leen',
    },
    'golpear': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'golpea',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'golpean',
    },
    'amar': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'ama',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'aman',
    },
    'querer': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'quiere',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'quieren',
    },
    'saber': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'sabe',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'saben',
    },
    'pensar': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'piensa',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'piensan',
    },
    'ir': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'va',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'van',
    },
    'venir': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'viene',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'vienen',
    },
    'dar': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'da',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'dan',
    },
    'tomar': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'toma',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'toman',
    },
    'encontrar': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'encuentra',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'encuentran',
    },
    'decir': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'dice',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'dicen',
    },
    'llamar': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'llama',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'llaman',
    },
    'cantar': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'canta',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'cantan',
    },
    'jugar': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'juega',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'juegan',
    },
    'escribir': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'escribe',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'escriben',
    },
    'caminar': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'camina',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'caminan',
    },
    'dormir': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'duerme',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'duermen',
    },
    'beber': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'bebe',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'beben',
    },
    'comprar': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'compra',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'compran',
    },
    'abrir': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'abre',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'abren',
    },
    'cerrar': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'cierra',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'cierran',
    },
    'ayudar': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'ayuda',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'ayudan',
    },
    'hablar': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'habla',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'hablan',
    },
    'ser': {
        (SubType.FIRST_PERSON, SubType.SINGULAR): 'soy',
        (SubType.SECOND_PERSON, SubType.SINGULAR): 'eres',
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'es',
        (SubType.FIRST_PERSON, SubType.PLURAL): 'somos',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'son',
    },
    'gustar': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'gusta',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'gustan',
    },
    'necesitar': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'necesita',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'necesitan',
    },
}

FRENCH_VERBS = {
    'courir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'court'},
    'poursuivre': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'poursuit'},
    'manger': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'mange'},
    'voir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'voit'},
    'faire': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'fait'},
    'avoir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'a'},
    'vivre': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'vit'},
    'lire': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'lit'},
    'frapper': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'frappe'},
    'aimer': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'aime'},
    'vouloir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'veut'},
    'savoir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'sait'},
    'penser': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'pense'},
    'aller': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'va'},
    'venir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'vient'},
    'donner': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'donne'},
    'prendre': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'prend'},
    'trouver': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'trouve'},
    'dire': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'dit'},
    'appeler': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'appelle'},
    'chanter': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'chante'},
    'jouer': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'joue'},
    'écrire': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'écrit'},
    'marcher': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'marche'},
    'dormir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'dort'},
    'boire': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'boit'},
    'acheter': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'achète'},
    'ouvrir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'ouvre'},
    'fermer': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'ferme'},
    'aider': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'aide'},
    'parler': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'parle'},
    'être': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'est',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'sont',
    },
}

GERMAN_VERBS = {
    'rennen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'rennt'},
    'jagen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'jagt'},
    'essen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'isst'},
    'sehen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'sieht'},
    'machen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'macht'},
    'haben': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'hat'},
    'leben': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'lebt'},
    'lesen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'liest'},
    'schlagen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'schlägt'},
    'lieben': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'liebt'},
    'wollen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'will'},
    'wissen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'weiss'},
    'denken': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'denkt'},
    'gehen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'geht'},
    'kommen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'kommt'},
    'geben': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'gibt'},
    'nehmen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'nimmt'},
    'finden': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'findet'},
    'sagen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'sagt'},
    'erzählen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'erzählt'},
    'rufen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'ruft'},
    'singen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'singt'},
    'spielen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'spielt'},
    'schreiben': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'schreibt'},
    'schlafen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'schläft'},
    'trinken': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'trinkt'},
    'kaufen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'kauft'},
    'öffnen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'öffnet'},
    'schliessen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'schliesst'},
    'helfen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'hilft'},
    'sprechen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'spricht'},
    'sein': {
        (SubType.FIRST_PERSON, SubType.SINGULAR): 'bin',
        (SubType.SECOND_PERSON, SubType.SINGULAR): 'bist',
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'ist',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'sind',
    },
    'mögen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'mag'},
    'brauchen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'braucht'},
}

PORTUGUESE_VERBS = {
    'correr': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'corre'},
    'comer': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'come'},
    'perseguir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'persegue'},
    'ver': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'vê'},
    'fazer': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'faz'},
    'ter': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'tem'},
    'viver': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'vive'},
    'ler': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'lê'},
    'bater': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'bate'},
    'amar': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'ama'},
    'querer': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'quer'},
    'saber': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'sabe'},
    'pensar': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'pensa'},
    'ir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'vai'},
    'vir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'vem'},
    'dar': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'dá'},
    'pegar': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'pega'},
    'encontrar': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'encontra'},
    'dizer': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'diz'},
    'chamar': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'chama'},
    'cantar': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'canta'},
    'jogar': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'joga'},
    'escrever': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'escreve'},
    'andar': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'anda'},
    'dormir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'dorme'},
    'beber': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'bebe'},
    'comprar': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'compra'},
    'abrir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'abre'},
    'fechar': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'fecha'},
    'ajudar': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'ajuda'},
    'falar': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'fala'},
    'ser': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'é',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'são',
    },
    'gostar': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'gosta'},
    'precisar': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'precisa'},
}

ENGLISH_VERBS = {
    'run': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'runs'},
    'chase': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'chases'},
    'eat': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'eats'},
    'see': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'sees'},
    'make': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'makes'},
    'have': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'has'},
    'live': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'lives'},
    'read': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'reads'},
    'hit': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'hits'},
    'love': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'loves'},
    'want': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'wants'},
    'know': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'knows'},
    'think': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'thinks'},
    'go': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'goes'},
    'come': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'comes'},
    'give': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'gives'},
    'take': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'takes'},
    'find': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'finds'},
    'say': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'says'},
    'tell': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'tells'},
    'call': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'calls'},
    'sing': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'sings'},
    'play': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'plays'},
    'write': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'writes'},
    'walk': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'walks'},
    'sleep': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'sleeps'},
    'drink': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'drinks'},
    'buy': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'buys'},
    'open': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'opens'},
    'close': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'closes'},
    'help': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'helps'},
    'speak': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'speaks'},
    'be': {
        (SubType.FIRST_PERSON, SubType.SINGULAR): 'am',
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'is',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'are',
    },
    'like': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'likes'},
    'need': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'needs'},
}

VERB_TABLES = {
    'english': ENGLISH_VERBS,
    'spanish': SPANISH_VERBS,
    'french': FRENCH_VERBS,
    'german': GERMAN_VERBS,
    'portuguese': PORTUGUESE_VERBS,
    'japanese': {},  # No conjugation
}


# ============================================================================
# Adjective form tables: lemma -> {(gender, number): surface_form}
# ============================================================================

SPANISH_ADJS = {
    'grande': {
        (): 'grande',
        (SubType.SINGULAR,): 'grande',
        (SubType.PLURAL,): 'grandes',
        (SubType.MASCULINE, SubType.SINGULAR): 'grande',
        (SubType.FEMININE, SubType.SINGULAR): 'grande',
        (SubType.MASCULINE, SubType.PLURAL): 'grandes',
        (SubType.FEMININE, SubType.PLURAL): 'grandes',
    },
    'pequeño': {
        (SubType.MASCULINE, SubType.SINGULAR): 'pequeño',
        (SubType.FEMININE, SubType.SINGULAR): 'pequeña',
        (SubType.MASCULINE, SubType.PLURAL): 'pequeños',
        (SubType.FEMININE, SubType.PLURAL): 'pequeñas',
    },
    'blanco': {
        (SubType.MASCULINE, SubType.SINGULAR): 'blanco',
        (SubType.FEMININE, SubType.SINGULAR): 'blanca',
        (SubType.MASCULINE, SubType.PLURAL): 'blancos',
        (SubType.FEMININE, SubType.PLURAL): 'blancas',
    },
    'negro': {
        (SubType.MASCULINE, SubType.SINGULAR): 'negro',
        (SubType.FEMININE, SubType.SINGULAR): 'negra',
        (SubType.MASCULINE, SubType.PLURAL): 'negros',
        (SubType.FEMININE, SubType.PLURAL): 'negras',
    },
    'rojo': {
        (SubType.MASCULINE, SubType.SINGULAR): 'rojo',
        (SubType.FEMININE, SubType.SINGULAR): 'roja',
        (SubType.MASCULINE, SubType.PLURAL): 'rojos',
        (SubType.FEMININE, SubType.PLURAL): 'rojas',
    },
    'azul': {
        (SubType.SINGULAR,): 'azul',
        (SubType.PLURAL,): 'azules',
        (SubType.MASCULINE, SubType.SINGULAR): 'azul',
        (SubType.FEMININE, SubType.SINGULAR): 'azul',
        (SubType.MASCULINE, SubType.PLURAL): 'azules',
        (SubType.FEMININE, SubType.PLURAL): 'azules',
    },
    'verde': {
        (SubType.SINGULAR,): 'verde',
        (SubType.PLURAL,): 'verdes',
        (SubType.MASCULINE, SubType.SINGULAR): 'verde',
        (SubType.FEMININE, SubType.SINGULAR): 'verde',
        (SubType.MASCULINE, SubType.PLURAL): 'verdes',
        (SubType.FEMININE, SubType.PLURAL): 'verdes',
    },
    'bueno': {
        (SubType.MASCULINE, SubType.SINGULAR): 'bueno',
        (SubType.FEMININE, SubType.SINGULAR): 'buena',
        (SubType.MASCULINE, SubType.PLURAL): 'buenos',
        (SubType.FEMININE, SubType.PLURAL): 'buenas',
    },
    'malo': {
        (SubType.MASCULINE, SubType.SINGULAR): 'malo',
        (SubType.FEMININE, SubType.SINGULAR): 'mala',
        (SubType.MASCULINE, SubType.PLURAL): 'malos',
        (SubType.FEMININE, SubType.PLURAL): 'malas',
    },
    'viejo': {
        (SubType.MASCULINE, SubType.SINGULAR): 'viejo',
        (SubType.FEMININE, SubType.SINGULAR): 'vieja',
        (SubType.MASCULINE, SubType.PLURAL): 'viejos',
        (SubType.FEMININE, SubType.PLURAL): 'viejas',
    },
    'nuevo': {
        (SubType.MASCULINE, SubType.SINGULAR): 'nuevo',
        (SubType.FEMININE, SubType.SINGULAR): 'nueva',
        (SubType.MASCULINE, SubType.PLURAL): 'nuevos',
        (SubType.FEMININE, SubType.PLURAL): 'nuevas',
    },
    'bonito': {
        (SubType.MASCULINE, SubType.SINGULAR): 'bonito',
        (SubType.FEMININE, SubType.SINGULAR): 'bonita',
        (SubType.MASCULINE, SubType.PLURAL): 'bonitos',
        (SubType.FEMININE, SubType.PLURAL): 'bonitas',
    },
    'feo': {
        (SubType.MASCULINE, SubType.SINGULAR): 'feo',
        (SubType.FEMININE, SubType.SINGULAR): 'fea',
        (SubType.MASCULINE, SubType.PLURAL): 'feos',
        (SubType.FEMININE, SubType.PLURAL): 'feas',
    },
    'alto': {
        (SubType.MASCULINE, SubType.SINGULAR): 'alto',
        (SubType.FEMININE, SubType.SINGULAR): 'alta',
        (SubType.MASCULINE, SubType.PLURAL): 'altos',
        (SubType.FEMININE, SubType.PLURAL): 'altas',
    },
    'bajo': {
        (SubType.MASCULINE, SubType.SINGULAR): 'bajo',
        (SubType.FEMININE, SubType.SINGULAR): 'baja',
        (SubType.MASCULINE, SubType.PLURAL): 'bajos',
        (SubType.FEMININE, SubType.PLURAL): 'bajas',
    },
    'feliz': {
        (SubType.SINGULAR,): 'feliz',
        (SubType.PLURAL,): 'felices',
        (SubType.MASCULINE, SubType.SINGULAR): 'feliz',
        (SubType.FEMININE, SubType.SINGULAR): 'feliz',
        (SubType.MASCULINE, SubType.PLURAL): 'felices',
        (SubType.FEMININE, SubType.PLURAL): 'felices',
    },
    'triste': {
        (SubType.SINGULAR,): 'triste',
        (SubType.PLURAL,): 'tristes',
        (SubType.MASCULINE, SubType.SINGULAR): 'triste',
        (SubType.FEMININE, SubType.SINGULAR): 'triste',
        (SubType.MASCULINE, SubType.PLURAL): 'tristes',
        (SubType.FEMININE, SubType.PLURAL): 'tristes',
    },
    'rápido': {
        (SubType.MASCULINE, SubType.SINGULAR): 'rápido',
        (SubType.FEMININE, SubType.SINGULAR): 'rápida',
        (SubType.MASCULINE, SubType.PLURAL): 'rápidos',
        (SubType.FEMININE, SubType.PLURAL): 'rápidas',
    },
    'lento': {
        (SubType.MASCULINE, SubType.SINGULAR): 'lento',
        (SubType.FEMININE, SubType.SINGULAR): 'lenta',
        (SubType.MASCULINE, SubType.PLURAL): 'lentos',
        (SubType.FEMININE, SubType.PLURAL): 'lentas',
    },
    'joven': {
        (SubType.SINGULAR,): 'joven',
        (SubType.PLURAL,): 'jóvenes',
        (SubType.MASCULINE, SubType.SINGULAR): 'joven',
        (SubType.FEMININE, SubType.SINGULAR): 'joven',
        (SubType.MASCULINE, SubType.PLURAL): 'jóvenes',
        (SubType.FEMININE, SubType.PLURAL): 'jóvenes',
    },
    'fuerte': {
        (SubType.SINGULAR,): 'fuerte',
        (SubType.PLURAL,): 'fuertes',
        (SubType.MASCULINE, SubType.SINGULAR): 'fuerte',
        (SubType.FEMININE, SubType.SINGULAR): 'fuerte',
        (SubType.MASCULINE, SubType.PLURAL): 'fuertes',
        (SubType.FEMININE, SubType.PLURAL): 'fuertes',
    },
    'débil': {
        (SubType.SINGULAR,): 'débil',
        (SubType.PLURAL,): 'débiles',
        (SubType.MASCULINE, SubType.SINGULAR): 'débil',
        (SubType.FEMININE, SubType.SINGULAR): 'débil',
        (SubType.MASCULINE, SubType.PLURAL): 'débiles',
        (SubType.FEMININE, SubType.PLURAL): 'débiles',
    },
    'rico': {
        (SubType.MASCULINE, SubType.SINGULAR): 'rico',
        (SubType.FEMININE, SubType.SINGULAR): 'rica',
        (SubType.MASCULINE, SubType.PLURAL): 'ricos',
        (SubType.FEMININE, SubType.PLURAL): 'ricas',
    },
    'pobre': {
        (SubType.SINGULAR,): 'pobre',
        (SubType.PLURAL,): 'pobres',
        (SubType.MASCULINE, SubType.SINGULAR): 'pobre',
        (SubType.FEMININE, SubType.SINGULAR): 'pobre',
        (SubType.MASCULINE, SubType.PLURAL): 'pobres',
        (SubType.FEMININE, SubType.PLURAL): 'pobres',
    },
    'caliente': {
        (SubType.SINGULAR,): 'caliente',
        (SubType.PLURAL,): 'calientes',
        (SubType.MASCULINE, SubType.SINGULAR): 'caliente',
        (SubType.FEMININE, SubType.SINGULAR): 'caliente',
        (SubType.MASCULINE, SubType.PLURAL): 'calientes',
        (SubType.FEMININE, SubType.PLURAL): 'calientes',
    },
    'frío': {
        (SubType.MASCULINE, SubType.SINGULAR): 'frío',
        (SubType.FEMININE, SubType.SINGULAR): 'fría',
        (SubType.MASCULINE, SubType.PLURAL): 'fríos',
        (SubType.FEMININE, SubType.PLURAL): 'frías',
    },
    'largo': {
        (SubType.MASCULINE, SubType.SINGULAR): 'largo',
        (SubType.FEMININE, SubType.SINGULAR): 'larga',
        (SubType.MASCULINE, SubType.PLURAL): 'largos',
        (SubType.FEMININE, SubType.PLURAL): 'largas',
    },
    'importante': {
        (SubType.SINGULAR,): 'importante',
        (SubType.PLURAL,): 'importantes',
        (SubType.MASCULINE, SubType.SINGULAR): 'importante',
        (SubType.FEMININE, SubType.SINGULAR): 'importante',
        (SubType.MASCULINE, SubType.PLURAL): 'importantes',
        (SubType.FEMININE, SubType.PLURAL): 'importantes',
    },
    'favorito': {
        (SubType.MASCULINE, SubType.SINGULAR): 'favorito',
        (SubType.FEMININE, SubType.SINGULAR): 'favorita',
        (SubType.MASCULINE, SubType.PLURAL): 'favoritos',
        (SubType.FEMININE, SubType.PLURAL): 'favoritas',
    },
}

FRENCH_ADJS = {
    'grand': {
        (SubType.MASCULINE, SubType.SINGULAR): 'grand',
        (SubType.FEMININE, SubType.SINGULAR): 'grande',
        (SubType.MASCULINE, SubType.PLURAL): 'grands',
        (SubType.FEMININE, SubType.PLURAL): 'grandes',
    },
    'petit': {
        (SubType.MASCULINE, SubType.SINGULAR): 'petit',
        (SubType.FEMININE, SubType.SINGULAR): 'petite',
        (SubType.MASCULINE, SubType.PLURAL): 'petits',
        (SubType.FEMININE, SubType.PLURAL): 'petites',
    },
    'blanc': {
        (SubType.MASCULINE, SubType.SINGULAR): 'blanc',
        (SubType.FEMININE, SubType.SINGULAR): 'blanche',
        (SubType.MASCULINE, SubType.PLURAL): 'blancs',
        (SubType.FEMININE, SubType.PLURAL): 'blanches',
    },
    'noir': {
        (SubType.MASCULINE, SubType.SINGULAR): 'noir',
        (SubType.FEMININE, SubType.SINGULAR): 'noire',
        (SubType.MASCULINE, SubType.PLURAL): 'noirs',
        (SubType.FEMININE, SubType.PLURAL): 'noires',
    },
    'rouge': {
        (SubType.MASCULINE, SubType.SINGULAR): 'rouge',
        (SubType.FEMININE, SubType.SINGULAR): 'rouge',
        (SubType.MASCULINE, SubType.PLURAL): 'rouges',
        (SubType.FEMININE, SubType.PLURAL): 'rouges',
    },
    'bon': {
        (SubType.MASCULINE, SubType.SINGULAR): 'bon',
        (SubType.FEMININE, SubType.SINGULAR): 'bonne',
        (SubType.MASCULINE, SubType.PLURAL): 'bons',
        (SubType.FEMININE, SubType.PLURAL): 'bonnes',
    },
    'beau': {
        (SubType.MASCULINE, SubType.SINGULAR): 'beau',
        (SubType.FEMININE, SubType.SINGULAR): 'belle',
        (SubType.MASCULINE, SubType.PLURAL): 'beaux',
        (SubType.FEMININE, SubType.PLURAL): 'belles',
    },
    'vieux': {
        (SubType.MASCULINE, SubType.SINGULAR): 'vieux',
        (SubType.FEMININE, SubType.SINGULAR): 'vieille',
        (SubType.MASCULINE, SubType.PLURAL): 'vieux',
        (SubType.FEMININE, SubType.PLURAL): 'vieilles',
    },
    'nouveau': {
        (SubType.MASCULINE, SubType.SINGULAR): 'nouveau',
        (SubType.FEMININE, SubType.SINGULAR): 'nouvelle',
        (SubType.MASCULINE, SubType.PLURAL): 'nouveaux',
        (SubType.FEMININE, SubType.PLURAL): 'nouvelles',
    },
}

GERMAN_ADJS = {
    # Simplified: strong inflection nominative only for demo
    'gross': {
        (SubType.MASCULINE, SubType.SINGULAR): 'grosse',
        (SubType.FEMININE, SubType.SINGULAR): 'grosse',
        (SubType.NEUTER, SubType.SINGULAR): 'grosses',
        (SubType.PLURAL,): 'grossen',
    },
    'klein': {
        (SubType.MASCULINE, SubType.SINGULAR): 'kleine',
        (SubType.FEMININE, SubType.SINGULAR): 'kleine',
        (SubType.NEUTER, SubType.SINGULAR): 'kleines',
    },
    'weiss': {
        (SubType.MASCULINE, SubType.SINGULAR): 'weisser',
        (SubType.FEMININE, SubType.SINGULAR): 'weisse',
        (SubType.NEUTER, SubType.SINGULAR): 'weisses',
    },
    'gut': {
        (SubType.MASCULINE, SubType.SINGULAR): 'guter',
        (SubType.FEMININE, SubType.SINGULAR): 'gute',
        (SubType.NEUTER, SubType.SINGULAR): 'gutes',
    },
}

PORTUGUESE_ADJS = {
    'grande': {
        (): 'grande',
        (SubType.SINGULAR,): 'grande',
        (SubType.PLURAL,): 'grandes',
        (SubType.MASCULINE, SubType.SINGULAR): 'grande',
        (SubType.FEMININE, SubType.SINGULAR): 'grande',
        (SubType.MASCULINE, SubType.PLURAL): 'grandes',
        (SubType.FEMININE, SubType.PLURAL): 'grandes',
    },
    'branco': {
        (SubType.MASCULINE, SubType.SINGULAR): 'branco',
        (SubType.FEMININE, SubType.SINGULAR): 'branca',
        (SubType.MASCULINE, SubType.PLURAL): 'brancos',
        (SubType.FEMININE, SubType.PLURAL): 'brancas',
    },
    'preto': {
        (SubType.MASCULINE, SubType.SINGULAR): 'preto',
        (SubType.FEMININE, SubType.SINGULAR): 'preta',
        (SubType.MASCULINE, SubType.PLURAL): 'pretos',
        (SubType.FEMININE, SubType.PLURAL): 'pretas',
    },
    'pequeno': {
        (SubType.MASCULINE, SubType.SINGULAR): 'pequeno',
        (SubType.FEMININE, SubType.SINGULAR): 'pequena',
        (SubType.MASCULINE, SubType.PLURAL): 'pequenos',
        (SubType.FEMININE, SubType.PLURAL): 'pequenas',
    },
    'bonito': {
        (SubType.MASCULINE, SubType.SINGULAR): 'bonito',
        (SubType.FEMININE, SubType.SINGULAR): 'bonita',
        (SubType.MASCULINE, SubType.PLURAL): 'bonitos',
        (SubType.FEMININE, SubType.PLURAL): 'bonitas',
    },
    'velho': {
        (SubType.MASCULINE, SubType.SINGULAR): 'velho',
        (SubType.FEMININE, SubType.SINGULAR): 'velha',
        (SubType.MASCULINE, SubType.PLURAL): 'velhos',
        (SubType.FEMININE, SubType.PLURAL): 'velhas',
    },
    'novo': {
        (SubType.MASCULINE, SubType.SINGULAR): 'novo',
        (SubType.FEMININE, SubType.SINGULAR): 'nova',
        (SubType.MASCULINE, SubType.PLURAL): 'novos',
        (SubType.FEMININE, SubType.PLURAL): 'novas',
    },
}

ADJ_TABLES = {
    'english': {},  # English adjectives don't inflect
    'spanish': SPANISH_ADJS,
    'french': FRENCH_ADJS,
    'german': GERMAN_ADJS,
    'portuguese': PORTUGUESE_ADJS,
    'japanese': {},  # Japanese adjectives in romaji don't inflect for demo
}


# ============================================================================
# Noun gender registry: lemma -> gender (for target language agreement)
# ============================================================================

NOUN_GENDERS = {
    'spanish': {
        # Animals
        'perro': SubType.MASCULINE, 'gato': SubType.MASCULINE,
        'pájaro': SubType.MASCULINE, 'ratón': SubType.MASCULINE,
        'caballo': SubType.MASCULINE, 'pez': SubType.MASCULINE,
        # People & family
        'hombre': SubType.MASCULINE, 'mujer': SubType.FEMININE,
        'niño': SubType.MASCULINE, 'niña': SubType.FEMININE,
        'bebé': SubType.MASCULINE,
        'madre': SubType.FEMININE, 'padre': SubType.MASCULINE,
        'hermana': SubType.FEMININE, 'hermano': SubType.MASCULINE,
        'familia': SubType.FEMININE,
        'persona': SubType.FEMININE, 'gente': SubType.FEMININE,
        'amigo': SubType.MASCULINE, 'profesor': SubType.MASCULINE,
        'estudiante': SubType.MASCULINE, 'doctor': SubType.MASCULINE,
        'rey': SubType.MASCULINE, 'reina': SubType.FEMININE,
        # Body
        'mano': SubType.FEMININE, 'cabeza': SubType.FEMININE,
        'ojo': SubType.MASCULINE, 'corazón': SubType.MASCULINE,
        # Places & things
        'casa': SubType.FEMININE, 'hogar': SubType.MASCULINE,
        'habitación': SubType.FEMININE,
        'puerta': SubType.FEMININE, 'ventana': SubType.FEMININE,
        'escuela': SubType.FEMININE, 'ciudad': SubType.FEMININE,
        'parque': SubType.MASCULINE, 'coche': SubType.MASCULINE,
        'mesa': SubType.FEMININE, 'silla': SubType.FEMININE,
        'libro': SubType.MASCULINE,
        # Nature & abstract
        'árbol': SubType.MASCULINE, 'flor': SubType.FEMININE,
        'sol': SubType.MASCULINE, 'luna': SubType.FEMININE,
        'estrella': SubType.FEMININE, 'agua': SubType.FEMININE,
        'comida': SubType.FEMININE, 'día': SubType.MASCULINE,
        'noche': SubType.FEMININE, 'tiempo': SubType.MASCULINE,
        'nombre': SubType.MASCULINE, 'mundo': SubType.MASCULINE,
        'vida': SubType.FEMININE, 'dinero': SubType.MASCULINE,
        'trabajo': SubType.MASCULINE, 'país': SubType.MASCULINE,
        'color': SubType.MASCULINE,
    },
    'french': {
        'chien': SubType.MASCULINE, 'chat': SubType.MASCULINE,
        'oiseau': SubType.MASCULINE, 'souris': SubType.FEMININE,
        'cheval': SubType.MASCULINE, 'poisson': SubType.MASCULINE,
        'homme': SubType.MASCULINE, 'femme': SubType.FEMININE,
        'enfant': SubType.MASCULINE, 'garçon': SubType.MASCULINE,
        'fille': SubType.FEMININE, 'bébé': SubType.MASCULINE,
        'mère': SubType.FEMININE, 'père': SubType.MASCULINE,
        'soeur': SubType.FEMININE, 'frère': SubType.MASCULINE,
        'famille': SubType.FEMININE,
        'personne': SubType.FEMININE, 'ami': SubType.MASCULINE,
        'professeur': SubType.MASCULINE, 'étudiant': SubType.MASCULINE,
        'médecin': SubType.MASCULINE, 'roi': SubType.MASCULINE,
        'reine': SubType.FEMININE,
        'main': SubType.FEMININE, 'tête': SubType.FEMININE,
        'oeil': SubType.MASCULINE, 'coeur': SubType.MASCULINE,
        'maison': SubType.FEMININE, 'foyer': SubType.MASCULINE,
        'chambre': SubType.FEMININE, 'porte': SubType.FEMININE,
        'fenêtre': SubType.FEMININE, 'école': SubType.FEMININE,
        'ville': SubType.FEMININE, 'parc': SubType.MASCULINE,
        'voiture': SubType.FEMININE, 'table': SubType.FEMININE,
        'chaise': SubType.FEMININE, 'livre': SubType.MASCULINE,
        'arbre': SubType.MASCULINE, 'fleur': SubType.FEMININE,
        'soleil': SubType.MASCULINE, 'lune': SubType.FEMININE,
        'étoile': SubType.FEMININE, 'eau': SubType.FEMININE,
        'nourriture': SubType.FEMININE, 'jour': SubType.MASCULINE,
        'nuit': SubType.FEMININE, 'temps': SubType.MASCULINE,
        'nom': SubType.MASCULINE, 'monde': SubType.MASCULINE,
        'vie': SubType.FEMININE, 'argent': SubType.MASCULINE,
        'travail': SubType.MASCULINE, 'pays': SubType.MASCULINE,
        'couleur': SubType.FEMININE,
    },
    'german': {
        'Hund': SubType.MASCULINE, 'Katze': SubType.FEMININE,
        'Vogel': SubType.MASCULINE, 'Maus': SubType.FEMININE,
        'Pferd': SubType.NEUTER, 'Fisch': SubType.MASCULINE,
        'Mann': SubType.MASCULINE, 'Frau': SubType.FEMININE,
        'Kind': SubType.NEUTER, 'Junge': SubType.MASCULINE,
        'Mädchen': SubType.NEUTER, 'Baby': SubType.NEUTER,
        'Mutter': SubType.FEMININE, 'Vater': SubType.MASCULINE,
        'Schwester': SubType.FEMININE, 'Bruder': SubType.MASCULINE,
        'Familie': SubType.FEMININE,
        'Person': SubType.FEMININE, 'Freund': SubType.MASCULINE,
        'Lehrer': SubType.MASCULINE, 'Student': SubType.MASCULINE,
        'Arzt': SubType.MASCULINE, 'König': SubType.MASCULINE,
        'Königin': SubType.FEMININE,
        'Hand': SubType.FEMININE, 'Kopf': SubType.MASCULINE,
        'Auge': SubType.NEUTER, 'Herz': SubType.NEUTER,
        'Haus': SubType.NEUTER, 'Zimmer': SubType.NEUTER,
        'Tür': SubType.FEMININE, 'Fenster': SubType.NEUTER,
        'Schule': SubType.FEMININE, 'Stadt': SubType.FEMININE,
        'Park': SubType.MASCULINE, 'Auto': SubType.NEUTER,
        'Tisch': SubType.MASCULINE, 'Stuhl': SubType.MASCULINE,
        'Buch': SubType.NEUTER,
        'Baum': SubType.MASCULINE, 'Blume': SubType.FEMININE,
        'Sonne': SubType.FEMININE, 'Mond': SubType.MASCULINE,
        'Stern': SubType.MASCULINE, 'Wasser': SubType.NEUTER,
        'Essen': SubType.NEUTER, 'Tag': SubType.MASCULINE,
        'Nacht': SubType.FEMININE, 'Zeit': SubType.FEMININE,
        'Name': SubType.MASCULINE, 'Welt': SubType.FEMININE,
        'Leben': SubType.NEUTER, 'Geld': SubType.NEUTER,
        'Arbeit': SubType.FEMININE, 'Land': SubType.NEUTER,
        'Farbe': SubType.FEMININE,
    },
    'portuguese': {
        'cachorro': SubType.MASCULINE, 'gato': SubType.MASCULINE,
        'pássaro': SubType.MASCULINE, 'rato': SubType.MASCULINE,
        'cavalo': SubType.MASCULINE, 'peixe': SubType.MASCULINE,
        'homem': SubType.MASCULINE, 'mulher': SubType.FEMININE,
        'criança': SubType.FEMININE, 'menino': SubType.MASCULINE,
        'menina': SubType.FEMININE, 'bebê': SubType.MASCULINE,
        'mãe': SubType.FEMININE, 'pai': SubType.MASCULINE,
        'irmã': SubType.FEMININE, 'irmão': SubType.MASCULINE,
        'família': SubType.FEMININE,
        'pessoa': SubType.FEMININE, 'gente': SubType.FEMININE,
        'amigo': SubType.MASCULINE, 'professor': SubType.MASCULINE,
        'estudante': SubType.MASCULINE, 'médico': SubType.MASCULINE,
        'rei': SubType.MASCULINE, 'rainha': SubType.FEMININE,
        'mão': SubType.FEMININE, 'cabeça': SubType.FEMININE,
        'olho': SubType.MASCULINE, 'coração': SubType.MASCULINE,
        'casa': SubType.FEMININE, 'lar': SubType.MASCULINE,
        'quarto': SubType.MASCULINE, 'porta': SubType.FEMININE,
        'janela': SubType.FEMININE, 'escola': SubType.FEMININE,
        'cidade': SubType.FEMININE, 'parque': SubType.MASCULINE,
        'carro': SubType.MASCULINE, 'mesa': SubType.FEMININE,
        'cadeira': SubType.FEMININE, 'livro': SubType.MASCULINE,
        'árvore': SubType.FEMININE, 'flor': SubType.FEMININE,
        'sol': SubType.MASCULINE, 'lua': SubType.FEMININE,
        'estrela': SubType.FEMININE, 'água': SubType.FEMININE,
        'comida': SubType.FEMININE, 'dia': SubType.MASCULINE,
        'noite': SubType.FEMININE, 'tempo': SubType.MASCULINE,
        'nome': SubType.MASCULINE, 'mundo': SubType.MASCULINE,
        'vida': SubType.FEMININE, 'dinheiro': SubType.MASCULINE,
        'trabalho': SubType.MASCULINE, 'país': SubType.MASCULINE,
        'cor': SubType.FEMININE,
    },
}


# ============================================================================
# Definite article tables: (gender, number) -> article
# ============================================================================

DEFINITE_ARTICLES = {
    'english': lambda g, n: 'the',
    'spanish': {
        (SubType.MASCULINE, SubType.SINGULAR): 'el',
        (SubType.FEMININE, SubType.SINGULAR): 'la',
        (SubType.MASCULINE, SubType.PLURAL): 'los',
        (SubType.FEMININE, SubType.PLURAL): 'las',
    },
    'french': {
        (SubType.MASCULINE, SubType.SINGULAR): 'le',
        (SubType.FEMININE, SubType.SINGULAR): 'la',
        (SubType.PLURAL,): 'les',
        (SubType.MASCULINE, SubType.PLURAL): 'les',
        (SubType.FEMININE, SubType.PLURAL): 'les',
    },
    'german': {
        (SubType.MASCULINE, SubType.SINGULAR): 'der',
        (SubType.FEMININE, SubType.SINGULAR): 'die',
        (SubType.NEUTER, SubType.SINGULAR): 'das',
        (SubType.PLURAL,): 'die',
        (SubType.MASCULINE, SubType.PLURAL): 'die',
        (SubType.FEMININE, SubType.PLURAL): 'die',
        (SubType.NEUTER, SubType.PLURAL): 'die',
    },
    'portuguese': {
        (SubType.MASCULINE, SubType.SINGULAR): 'o',
        (SubType.FEMININE, SubType.SINGULAR): 'a',
        (SubType.MASCULINE, SubType.PLURAL): 'os',
        (SubType.FEMININE, SubType.PLURAL): 'as',
    },
    'japanese': None,  # No articles
}


class SurfaceFormSelector:
    """Select the correct inflected surface form given a lemma + features."""

    def __init__(self, language: str):
        self.language = language
        self.verb_table = VERB_TABLES.get(language, {})
        self.adj_table = ADJ_TABLES.get(language, {})
        self.noun_genders = NOUN_GENDERS.get(language, {})

    def conjugate_verb(self, lemma: str, features: List[SubType]) -> str:
        """Get conjugated verb form."""
        if lemma not in self.verb_table:
            return lemma

        # Extract person and number
        person = None
        number = SubType.SINGULAR  # default
        for f in features:
            if f in (SubType.FIRST_PERSON, SubType.SECOND_PERSON, SubType.THIRD_PERSON):
                person = f
            if f in (SubType.SINGULAR, SubType.PLURAL):
                number = f

        if person is None:
            person = SubType.THIRD_PERSON  # default

        key = (person, number)
        forms = self.verb_table[lemma]
        if key in forms:
            return forms[key]
        return lemma

    def inflect_adjective(self, lemma: str, gender: SubType = None,
                          number: SubType = SubType.SINGULAR) -> str:
        """Get correctly inflected adjective form."""
        if lemma not in self.adj_table:
            return lemma

        forms = self.adj_table[lemma]

        # Try most specific key first, then fall back
        keys_to_try = []
        if gender and number:
            keys_to_try.append((gender, number))
        if number:
            keys_to_try.append((number,))
        keys_to_try.append(())

        for key in keys_to_try:
            if key in forms:
                return forms[key]

        # Return lemma if no match
        return lemma

    def get_noun_gender(self, lemma: str) -> Optional[SubType]:
        """Get the grammatical gender of a noun in this language."""
        return self.noun_genders.get(lemma)

    def get_definite_article(self, gender: SubType = None,
                             number: SubType = SubType.SINGULAR) -> Optional[str]:
        """Get the correct definite article for given gender/number."""
        articles = DEFINITE_ARTICLES.get(self.language)
        if articles is None:
            return None  # Japanese, etc.
        if callable(articles):
            return articles(gender, number)

        # Dict lookup
        keys_to_try = []
        if gender and number:
            keys_to_try.append((gender, number))
        if number:
            keys_to_try.append((number,))
        if gender:
            keys_to_try.append((gender,))

        for key in keys_to_try:
            if key in articles:
                return articles[key]

        return None
