# Scheme heavily adapted from https://github.com/deepmind/pycolab/
# '@' means "wall"
# 'P' means "player" spawn point
# 'A' means apply spawn point
# '' is empty space

HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ P   P      A    P AAAAA    P  A P  @',
    '@  P     A P AA    P    AAA    A  A  @',
    '@     A AAA  AAA    A    A AA AAAA   @',
    '@ A  AAA A    A  A AAA  A  A   A A   @',
    '@AAA  A A    A  AAA A  AAA        A P@',
    '@ A A  AAA  AAA  A A    A AA   AA AA @',
    '@  A A  AAA    A A  AAA    AAA  A    @',
    '@   AAA  A      AAA  A    AAAA       @',
    '@ P  A       A  A AAA    A  A      P @',
    '@A  AAA  A  A  AAA A    AAAA     P   @',
    '@    A A   AAA  A A      A AA   A  P @',
    '@     AAA   A A  AAA      AA   AAA P @',
    '@ A    A     AAA  A  P          A    @',
    '@       P     A         P  P P     P @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']

LARGE_HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@P A P   A    A    A    A  A    A    @',
    '@  AAA  AAA  AAA  AAA  AAAAAA  AAA   @',
    '@ A A    A    A    A    A  A    A   P@',
    '@PA             A      A       A     @',
    '@ A   A    A    A    A  A A  A    A  @',
    '@PAA AAA  AAA  AAA  AAA     AAA  AAA @',
    '@ A   A    A  A A  A A   P   A    A  @',
    '@PA                                P @',
    '@ A    A    A    A    A  A    A    A @',
    '@AAA  AAA  AAA  AAA  AA AAA  AAA  AA @',
    '@ A    A    A    A    A  A    A    A @',
    '@P A A A               P             @',
    '@P  A    A    A    A       P     P   @',
    '@  AAA  AAA  AAA  AAA         P    P @',
    '@P  A    A    A    A   P   P  P  P   @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@', ]

SMALL_HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@P A    A    A    A  P AP@',
    '@PAAA  AAA  AAA  AAA  AAA@',
    '@  A    A    A    A    A @',
    '@P                       @',
    '@    A    A    A    A    @',
    '@   AAA  AAA  AAA  AAA   @',
    '@P P A    A    A    A P P@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@', ]

MEDIUM_HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@P      A    A    A  P  P@',
    '@P     AAA  AAA  AAA     @',
    '@       A    A    A      @',
    '@P                       @',
    '@    A    A    A    A    @',
    '@   AAA  AAA  AAA  AAA   @',
    '@P P A    A    A    A P P@',
    '@P                       @',
    '@P      A    A    A  P  P@',
    '@P     AAA  AAA  AAA     @',
    '@       A    A    A      @',
    '@P                       @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@', ]

