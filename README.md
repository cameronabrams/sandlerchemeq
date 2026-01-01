# Sandlerchemeq

> Chemical equilibrium calculation utilities based on Sandler's 5th ed.

Sandlerchemeq implements computational tools for chemical equilibrium calculations based on _Chemical, Biochemical, and Engineering Thermodynamics_ (5th edition) by Stan Sandler (Wiley, USA). It should be used for educational purposes only.

## Installation 

Sandlerchemeq is available via `pip`:

```sh
pip install sandlerchemeq
```

## Usage

### Command-line

The general structure of a `sandlerchemeq` command is

```sh
$ sandlerchemeq [<global-options>] <tool> [<tool-options>]
```
Currently available tool(s) is/are
- `solve`: Solve a chemical equilibrium system

```sh
$ sandlerchemeq --help
usage: sandlerchemeq [-h] [-b | --banner | --no-banner] [--logging-level {None,info,debug,warning}] [-l LOG] <command> ...

Chemical equilibrium calculations via Gibbs energy minimization

options:
  -h, --help            show this help message and exit
  -b, --banner, --no-banner
                        toggle banner message
  --logging-level {None,info,debug,warning}
                        Logging level for messages written to diagnostic log
  -l LOG, --log LOG     File to which diagnostic log messages are written

subcommands:
  <command>
    solve               Solve chemical equilibrium
```

#### `sandlertools solve`

Below are examples of using `sandlerchemeq solve` to 
solve chemical equilibrium systems.

The first set uses the Lagrange multiplier method to determine
equilibrium compositions; hence, there is no concept of "reactions" here.

```sh
$ sandlerchemeq solve --components hydrogen nitrogen ammonia -T 400 -P 100 -n0 3. 1. 0.
N_{H2}=0.1113 y_{H2}=0.0536
N_{N2}=0.0371 y_{N2}=0.0179
N_{NH3}=1.9258 y_{NH3}=0.9285
$ sandlerchemeq solve --components hydrogen nitrogen ammonia -T 600 -P 100 -n0 3. 1. 0.
N_{H2}=1.2110 y_{H2}=0.4314
N_{N2}=0.4037 y_{N2}=0.1438
N_{NH3}=1.1926 y_{NH3}=0.4248
$ sandlerchemeq solve --components hydrogen nitrogen ammonia -T 600 -P 50 -n0 3. 1. 0.
N_{H2}=1.5881 y_{H2}=0.5192
N_{N2}=0.5294 y_{N2}=0.1731
N_{NH3}=0.9412 y_{NH3}=0.3077
```

The second set explicitly declares the reaction by the ordered list of compounds:

```sh
$ sandlerchemeq solve --reactions hydrogen,nitrogen,ammonia -T 400 -P 100 -n0 3. 1. 0. -xinit 0.95
Reaction    I:  3 H2  +  1 N2   <->   2 NH3  |  Ka(400.00 K)=3.12266e+01 => Xeq=9.62910e-01
N_{H2}=0.1113 y_{H2}=0.0536
N_{N2}=0.0371 y_{N2}=0.0179
N_{NH3}=1.9258 y_{NH3}=0.9285
$ sandlerchemeq solve --reactions hydrogen,nitrogen,ammonia -T 600 -P 100 -n0 3. 1. 0. -xinit 0.8
Reaction    I:  3 H2  +  1 N2   <->   2 NH3  |  Ka(600.00 K)=1.56354e-03 => Xeq=5.96321e-01
N_{H2}=1.2110 y_{H2}=0.4314
N_{N2}=0.4037 y_{N2}=0.1438
N_{NH3}=1.1926 y_{NH3}=0.4248
$ sandlerchemeq solve --reactions hydrogen,nitrogen,ammonia -T 600 -P 50 -n0 3. 1. 0. -xinit 0.3
Reaction    I:  3 H2  +  1 N2   <->   2 NH3  |  Ka(600.00 K)=1.56354e-03 => Xeq=4.70618e-01
N_{H2}=1.5881 y_{H2}=0.5192
N_{N2}=0.5294 y_{N2}=0.1731
N_{NH3}=0.9412 y_{NH3}=0.3077
```

When specifying `--reactions` you have to also specify `-xinit` for each reaction.  These have to be pretty good guesses to arrive at the right answer.

`sandlerchemeq` can handle systems of multiple reactions, but as long as you have an exhaustive list of the compounds you expect, the Lagrange method is much more robust.

### API

`sandlerchemeq` exposes several classes, objects, and functions from its component packages:

#### `Component`

`Component` is a class that inherits from the `Compound` class of `sandlerprops`, and it
has some additional attributes.  If you set a component's temperature you can query its
standard-state Gibbs energy of formation at that temperature:

```python
>>> from sandlerchemeq.component import Component 
>>> from sandlerprops.properties import get_database
>>> d = get_database()
>>> ammonia = Component.from_compound(d.get_compound('ammonia'), T=500) 
>>> ammonia.dGf
-16160.0
>>> ammonia.dGf_T
1732.1145064099328
```

#### `Reaction`

`Reaction` is a class for handling reaction stoichiometries.  For example, initializing a
`Reaction` object with an ordered list of components, one can determine the reaction stoichiometry (via the stoichiometric coefficients) and the property-changes upon reaction (including enthalpy of formation, Gibbs energy of formation, and ideal-gas heat capacity).


```python
>>> from sandlerchemeq.reaction import Reaction
>>> from sandlerchemeq.component import Component 
>>> from sandlerprops.properties import get_database
>>> ammonia = Component.from_compound(d.get_compound('ammonia'), T=298.15, P=1.0)                 
>>> hydrogen = Component.from_compound(d.get_compound('hydrogen (equilib)'), T=298.15, P=1.0)               
>>> nitrogen = Component.from_compound(d.get_compound('nitrogen'), T=298.15, P=1.0)               
>>> rxn = Reaction(components=[ammonia, nitrogen, hydrogen])
>>> rxn.nu
array([-2.,  1.,  3.])
>>> rxn.stoProps
{'dGf': np.float64(32320.0), 'dHf': np.float64(91460.0), 'Cp': array([ 5.7950e+01, -3.3408e-02, -4.8770e-05,  3.4955e-08])}
```

#### `ChemEqSystem`

`ChemEqSystem` is a class for handling chemical equilibria.  Initializing with an ordered list of components and initial molar amounts, it uses the Lagrange multiplier method to determine equilibrium amounts of all components:

```python
>>> from sandlerchemeq.reaction import Reaction
>>> from sandlerchemeq.component import Component 
>>> from sandlerprops.properties import get_database
>>> ammonia = Component.from_compound(d.get_compound('ammonia'), T=298.15, P=1.0)                 
>>> hydrogen = Component.from_compound(d.get_compound('hydrogen (equilib)'), T=298.15, P=1.0)               
>>> nitrogen = Component.from_compound(d.get_compound('nitrogen'), T=298.15, P=1.0)
>>> system = ChemEqSystem(Components=[ammonia, nitrogen, hydrogen], N0=np.array([0.0, 1.0, 3.0]),
                              T=500.0, P=100.0)
>>> system.solve_lagrange()
>>> print(system.report())
N_{NH3}=1.6828 y_{NH3}=0.7262
N_{N2}=0.1586 y_{N2}=0.0684
N_{H2}=0.4757 y_{H2}=0.2053
```

Alternatively, initializing with one or more reactions and initial guesses for extents of reaction
results in the use of equilibrium constants to solve for the equilibrium compositions:

```python
>>> rxn = Reaction(components=[nitrogen, hydrogen, ammonia])
>>> system = ChemEqSystem(Components=[nitrogen, hydrogen, ammonia],
                              Reactions=[rxn],
                              N0=np.array([1.0, 3.0, 0.0]),
                              T=500.0,
                              P=100.0)   
>>> system.solve_implicit(Xinit=[0.76])
>>> print(system.report())
Reaction    I:  1 N2  +  3 H2   <->   2 NH3  |  Ka(500.00 K)=8.90496e-02 => Xeq=8.41419e-01
N_{N2}=0.1586 y_{N2}=0.0684
N_{H2}=0.4757 y_{H2}=0.2053
N_{NH3}=1.6828 y_{NH3}=0.7262
```

## Release History

* 0.1.0
    * Initial release

## Meta

Cameron F. Abrams – cfa22@drexel.edu

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/cameronabrams](https://github.com/cameronabrams/)

## Contributing

1. Fork it (<https://github.com/cameronabrams/sandlerchemeq/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
