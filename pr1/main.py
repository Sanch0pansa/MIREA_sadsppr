from collections.abc import Callable

from .models import Gene, miRNA, mRNA, lncRNA, Protein, RNA


ATM_gene = Gene(name="ATM")
BAX_gene = Gene(name="BAX")
CDKN1A_gene = Gene(name="CDKN1A")
LINC_p21_gene = Gene(name="LINC-p21")
MIR34A_gene = Gene(name="MIR34A")
SIRT1_gene = Gene(name="SIRT1")
PUMA_gene = Gene(name="PUMA")
TP53_gene = Gene(name="TP53")

ATM_rna = mRNA(name="ATM")
BAX_rna = mRNA(name="BAX")
CDKN1A_rna = mRNA(name="CDKN1A")
PUMA_rna = mRNA(name="PUMA")
SIRT1_rna = mRNA(name="SIRT1")
TP53_rna = mRNA(name="TP53")

MIR34A_rna = miRNA(name="miR-34a")

LINC_p21_rna = lncRNA(name="LINC-p21", function="profiliration-related genes repressor")

ATM_protein = Protein(name="ATM", function="DNA damage detector")
BAX_protein = Protein(name="BAX", function="pro-apaptotic")
BCL_2_protein = Protein(name="BCL-2", function="anti-apaptotic")
CDK_protein = Protein(name="CDK", function="phosphorilates Rb")
MDM2_protein = Protein(name="MDM2", function="p53 inhibitor")
p21_protein = Protein(name="p21", function="CDK inhibitor")
p53_protein = Protein(name="p53", function="transcription factor")
PUMA_protein = Protein(name="PUMA", function="pro-apaptopic")
SIRT1_protein = Protein(name="SIRT1", function="p53 inhibitor")

ATM_gene.transcribed_to = ATM_rna
ATM_rna.translated_to = ATM_protein

BAX_gene.transcribed_to = BAX_rna
BAX_rna.translated_to = BAX_protein

CDKN1A_gene.transcribed_to = CDKN1A_rna
CDKN1A_rna.translated_to = p21_protein

LINC_p21_gene.transcribed_to = LINC_p21_rna

MIR34A_gene.transcribed_to = MIR34A_rna

PUMA_gene.transcribed_to = PUMA_rna
PUMA_rna.translated_to = PUMA_protein

SIRT1_gene.transcribed_to = SIRT1_rna
SIRT1_rna.translated_to = SIRT1_protein

TP53_gene.transcribed_to = TP53_rna
TP53_rna.translated_to = p53_protein

MIR34A_rna.targets.append(SIRT1_rna)

ATM_protein.represses.append(MDM2_protein)
BCL_2_protein.represses.append(BAX_protein)
MDM2_protein.represses.append(p53_protein)
p21_protein.represses.append(CDK_protein)
p53_protein.activates.extend([
    BAX_gene,
    CDKN1A_gene,
    LINC_p21_gene,
    PUMA_gene,
    MIR34A_gene,
])
PUMA_protein.represses.append(BCL_2_protein)
SIRT1_protein.represses.append(p53_protein)

BAX_gene.promoters.append(p53_protein)
CDKN1A_gene.promoters.append(p53_protein)
LINC_p21_gene.promoters.append(p53_protein)
PUMA_gene.promoters.append(p53_protein)
MIR34A_gene.promoters.append(p53_protein)

objects = [
    BAX_gene,
    BAX_protein,
    BAX_rna,
    ATM_gene,
    ATM_protein,
    ATM_rna,
    BCL_2_protein,
    CDK_protein,
    CDKN1A_gene,
    CDKN1A_rna,
    PUMA_gene,
    PUMA_protein,
    PUMA_rna,
    LINC_p21_gene,
    LINC_p21_rna,
    p21_protein,
    MIR34A_gene,
    MIR34A_rna,
    SIRT1_gene,
    SIRT1_rna,
    SIRT1_protein,
    TP53_gene,
    TP53_rna,
    p53_protein,
    MDM2_protein,
]

def find(
    class_to_find: type[Gene] | type[RNA] | type[Protein],
    query: Callable[..., bool],
) -> list[RNA | Gene | Protein]:
    return list(filter(lambda x: isinstance(x, class_to_find) and query(x), objects))



print("Получение генов, содержащих в промотере p53-активный элемент")
print(find(Gene, lambda x: p53_protein in x.promoters))

print("\nПолучение матричной РНК, из которой транслируется белок p21")
result = find(mRNA, lambda x: x.translated_to == p21_protein)
print(result)

print("\nПолучение связей гена ATM")
print(ATM_gene.transcribed_to)
print("\nПолучение связей рнк ATM через ген ATM")
print(ATM_gene.transcribed_to.translated_to)
print("\nПолучение связей белка ATM через ген и рнк ATM")
print(ATM_gene.transcribed_to.translated_to.represses)