# Beal-Conjecture-Counterexample-Search

There is a currently unproven conjecture in number theory known as the 'Beal Conjecture'. It is a generalisation of Fermat's Last Theorem and can be stated as follows. There are no positive integer solutions to the equation

x^a + y^b = z^c

where a,b,c >= 3 and x,y,z are all relatively prime. I wrote a program to find all integer solutions with sum < 2^80, and checked that they all have common factors. At the time of writing, there is a $1m bounty to anyone who can prove or disprove the conjecture. One way to disprove it is to find a counterexample. Much more extensive searches have been conducted, but I thought it would be an interesting exercise to list all the integer solution below some limit. The results are in beal4_80.txt.

An easy change to the program allowed me to list integer solutions to the same equation but restricted to fourth or higher powers. Results with sum up to 2^104 can be found in beal5_104.txt.

