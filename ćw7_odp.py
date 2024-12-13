class Punkt:
    """Klasa reprezentująca punkty na płaszczyźnie."""

    def __init__(self, x=0, y=0):  # konstuktor
        self.x = x
        self.y = y
        
    def __str__(self): # zwraca string "(x, y)"
        return f'String (x,y): {self.x, self.y}'

    def __repr__(self): # zwraca string "Punkt(x, y)" #reprezentacja punktu
        return f'Punkt({self.x}, {self.y})'     
    
    def __eq__(self, other): # obsługa point1 == point2
        return self.x == other.x, self.y == other.y
    
    def __ne__(self, other): # obsługa point1 != point2
        return self != other
    
    def __add__(self, other):  # v1 + v2
        return Punkt(self.x + other.x , self.y + other.y)
        
    def __sub__(self, other):  # v1 - v2
        return Punkt(self.x - other.x, self.y - other.y)
    
    def __mul__(self, other):   # v1 * v2, iloczyn skalarny (suma iloczynow wspolrzędnych)
        return self.x * other.x + self.y * other.y
    
    def length(self):   # długość wektora
        return (self.x**2 + self.y**2)**0.5
    
    def cross(self, other):  # v1 x v2, iloczyn wektorowy 2D
        return self.x * other.y - self.y * other.x






from string import maketrans
class DNAseq:
    
    def __init__(self, name, seq):
        self.name = name
        self.seq = seq
    def __repr__(self):  #zwraca reprezentacje w formacie FASTA
        return f'>{self.name}\s{self.seq}'
    def __add__(self,other):  #dodawanie sekwencji rozumiane jako konktenacja nazw i sekwenecji (tez obiekt klasy DNAseq!!) #sklejanie 
        add_name = f'{self.name}_{other.name}'
        add_seq = self.seq + other.seq
        return DNAseq(add_name, add_seq)
    def __len__(self): #dlugosc zdefiniowana jako dlugosc sekwencji
        return len(self.seq)
    def __getitem__(self, key): #slicowanie, operacja typu obiekt[4:6] (tez obiekt klasy DNAseq!!)
        return DNAseq(self.name, self.seq[key])
    def __eq__(self, other): #sprawdza czy dwa obiekty klasy DNAseq są takie same (te same nazwy i sekwencje) 
        return self.name == other.name and self.seq == other.seq
    def __contains__(self, value): #przyklad "ATGC" in "TCGCGCGAAA"
        return value in self.seq
    def komplementarna(self): #zwraca komplementarną (tez obiekt klasy DNAseq!!)
        komplement_seq = str.maketrans("ATCG", "TAGC")
        komplementarna_seq = self.seq.translate(komplement_seq)
        return DNAseq(f'komplementarna_{self.name}', komplementarna_seq)
    def odwrotnie_komplementarna(self): #zwraca odwrotnie komplementaerną (tez obiekt klasy DNAseq!!)
        komplem_seq = self.komplementarna().seq[::-1]
        return DNAseq(f"reverse_complement_{self.name}", komplem_seq)
    def zapisz(self): #zapisze do pliku w formacie FASTA, nazwa pliku to nazwa sekwencji
        file_name = f"{self.name}.fasta"
        with open(file_name, 'w') as file:
            file.write(self.__repr__())
    def sklad(self):  # częstosci A?, C?, G?, T? w postaci slownika
        klucz = ['A', 'C', 'G', 'T']
        wartość = [self.sekwencja.count(x) for x in klucz]
        słownik = dict(zip(klucz, wartość))
    def translacja(self):  #zwraca sekwencje aminokwasową (o ile to mozliwe)
        tabela = {'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L','TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '#', 'TAG': '#','TGT': 'C', 'TGC': 'C', 'TGA': '#', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CUG': 'L','CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q','CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M','ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K','AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V','GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E','GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'}
        translacja1 = ''.join(tabela[self.seq[i:i+3]] for i in range(0, len(self.seq), 3) if self.seq[i:i+3] in tabela)
        return (f'Translacja_{self.name}:', translacja1)
    def doklej(self, x):  #zwraca wydluzoną sekwencje (tez obiekt klasy DNAseq!!)
        return DNAseq(f"doklej_{self.name}", self.seq + x.seq)
