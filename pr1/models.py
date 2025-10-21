from __future__ import annotations

from abc import ABC

from pydantic import BaseModel, Field


class Gene(BaseModel):
    name: str
    promoters: list[Protein] = Field(default_factory=list)
    transcribed_to: RNA | None = Field(default=None)

    def __str__(self) -> str:
        return f"{self.name}({self.__class__.__name__})"
    
    def __repr__(self):
        return self.__str__()


class RNA(ABC, BaseModel):
    name: str

    def __str__(self) -> str:
        return f"{self.name}({self.__class__.__name__})"
    
    def __repr__(self):
        return self.__str__()


class mRNA(RNA):
    translated_to: Protein | None = Field(default=None)


class miRNA(RNA):
    targets: list[RNA] = Field(default_factory=list)


class lncRNA(RNA):
    function: str


class Protein(BaseModel):
    activates: list[Gene] = Field(default_factory=list)
    function: str
    name: str
    represses: list[Protein] = Field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.name}({self.__class__.__name__})"
    
    def __repr__(self):
        return self.__str__()