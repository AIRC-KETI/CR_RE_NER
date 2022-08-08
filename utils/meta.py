_KLUE_NER_TAGS = [
    'DT',  # date
    'LC',  # location
    'OG',  # organization
    'PS',  # person
    'QT',  # quantity
    'TI'  # time
]

_KLUE_NER_IOB2_TAGS = [
    'B-DT',
    'I-DT',
    'B-LC',
    'I-LC',
    'B-OG',
    'I-OG',
    'B-PS',
    'I-PS',
    'B-QT',
    'I-QT',
    'B-TI',
    'I-TI',
    'O'
]

_KLUE_RE_RELATIONS = [
    "no_relation",
    "org:dissolved",
    "org:founded",
    "org:place_of_headquarters",
    "org:alternate_names",
    "org:member_of",
    "org:members",
    "org:political/religious_affiliation",
    "org:product",
    "org:founded_by",
    "org:top_members/employees",
    "org:number_of_employees/members",
    "per:date_of_birth",
    "per:date_of_death",
    "per:place_of_birth",
    "per:place_of_death",
    "per:place_of_residence",
    "per:origin",
    "per:employee_of",
    "per:schools_attended",
    "per:alternate_names",
    "per:parents",
    "per:children",
    "per:siblings",
    "per:spouse",
    "per:other_family",
    "per:colleagues",
    "per:product",
    "per:religion",
    "per:title"
]

_DEFAULT_SPAN_TAGS = ['O', 'B', 'I']