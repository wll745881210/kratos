<TeXmacs|2.1.4>

<style|<tuple|generic|chinese>>

<\body>
  <section|Conversions of variables>

  <subsection|Abundances of chemical species>

  The conversion of variables from number density
  <math|<around*|{|n<rsub|i>|}>> to the "molar abundance
  <math|<around*|{|y<rsub|i>|}>> starts from the relation (<math|m<rsub|A>>
  is the atomic mass unit with mass dimension, and <math|\<rho\>> is the mass
  density),

  <\equation>
    y<rsub|i>\<equiv\><frac|n<rsub|i>m<rsub|A>|\<rho\>> .
  </equation>

  The conventional way of expressing ODEs for chemical kinetics is to use
  <math|<around*|{|n<rsub|i>|}>>, yet this may well cause problems when FP32
  is used while <math|<around*|{|n<rsub|i>|}>>'s order-of-magnitude deviates
  significantly from unity. The time derivative of the <math|i>th species
  <math|n<rsub|i>> can be contributed by multiple reactions, and the part
  contributed by the <math|X>th reaction shall be expressed in
  <math|<around*|(|\<mathd\>n<rsub|i>/\<mathd\>t|)><rsub|X>>, which further
  reads from the mass action law (<math|R<rsub|X>> is the number of
  reactants, and <around*|{|<math|I<rsub|X,j>>|}> are the indices for
  reactants; note that <math|I<rsub|X,j>> could be the same for different
  <math|j>'s, e.g., <math|\<Eta\>+\<Eta\>+\<Eta\>\<rightarrow\>\<Eta\><rsub|2>+\<Eta\>>),

  <\equation>
    <around*|(|<frac|\<mathd\>n<rsub|i>|\<mathd\>t>|)><rsub|X>=\<xi\><rsub|X,i>
    <around*|(|k<rsub|X><big|prod><rsub|j=0><rsup|R<rsub|X>-1>n<rsub|I<rsub|X,j>>|)>,
  </equation>

  where <math|<big|sum><rsub|i>\<xi\><rsub|X,i>> is the stoichiometric form
  of the <math|X>th reaction formula (for example,
  <math|\<Eta\>+\<Omicron\>\<rightarrow\>OH> shall be expressed as
  <math|\<xi\><rsub|\<Eta\>>+\<xi\><rsub|\<Omicron\>>+\<xi\><rsub|OH>>, where
  <math|\<xi\><rsub|\<Eta\>>=\<xi\><rsub|\<Omicron\>>=-1> and
  <math|\<xi\><rsub|OH>=1>). Converting the basic variables
  <math|<around*|{|n<rsub|i>|}>> into <math|<around*|{|y<rsub|i>|}>>, the
  rate equations shall be written as,

  <\equation>
    <around*|(|<frac|\<mathd\>y<rsub|i>|\<mathd\>t>|)><rsub|X>=<frac|m<rsub|A>|\<rho\>><around*|(|<frac|\<mathd\>n<rsub|i>|\<mathd\>t>|)><rsub|X>=\<xi\><rsub|X,i>
    <around*|[|<around*|(|<frac|m<rsub|A>|\<rho\>>|)>k<rsub|X><big|prod><rsub|j=0><rsup|R<rsub|X>-1><frac|y<rsub|I<rsub|X,j>>\<rho\>|m<rsub|A>>|]>=\<xi\><rsub|X,i>
    <around*|[|<around*|(|<frac|\<rho\>|m<rsub|A>>|)><rsup|R<rsub|X>-1>k<rsub|X><big|prod><rsub|j=0><rsup|R<rsub|X>-1>y<rsub|I<rsub|X,j>>|]>
    .
  </equation>

  More often than not, the chemistry databases one meets always provide the
  formula in the following form, where <math|\<lambda\><rsub|X>> is called
  the "rate coefficient" and is usually a function of temperature,

  <\equation>
    <around*|(|<frac|\<mathd\>y<rsub|i>|\<mathd\>t>|)><rsub|X>=\<xi\><rsub|X,i><around*|(|
    \<rho\><rsup|R<rsub|X>-1>\<lambda\><rsub|X><big|prod><rsub|j=0><rsup|R<rsub|X>-1>y<rsub|I<rsub|X,j>>|)>\<equiv\>f<rsub|X>
    \<xi\><rsub|X,i>
  </equation>

  The multiplication of <math|\<rho\><rsup|R<rsub|X>-1>> is conducted by the
  Kratos solver itself. In the hydrodynamic step, Kratos actually advects
  <math|>the "reduced number density" <math|<wide|y|~><rsub|i>\<equiv\>n<rsub|i>m<rsub|A>/\<rho\><rsub|0>>,
  where <math|\<rho\><rsub|0>> is the code unit of mass density in CGS
  (therefore, <math|<wide|y|~><rsub|i>> are dimensionless), and the
  conversion from <math|<wide|y<rsub|i>|~>> to the molar abundance
  <math|y<rsub|i>> is accomplished simply by dividing the total density in
  code unit (as the code-unit density is actually
  <math|<wide|\<rho\>|~>\<equiv\>\<rho\>/\<rho\><rsub|0>>; see circa line 154
  in \ src/modules/chemistry/solver/dat.h). When doing conversion for the
  rate coefficient (like the pre-factor in the generalized Arrhenius
  expressions), please remember: we shall start from the pre-exponential
  coefficient based on particle number density <math|k>, which, in
  generalized Arrhenius expression,

  <\equation>
    k<rsub|X>=A<around*|(|<frac|T|300 \<Kappa\>>|)><rsup|\<beta\>>exp<around*|(|-<frac|T|\<gamma\>>|)>
    .
  </equation>

  Note that only <math|A> is affected in unit conversions. For reactions with
  <math|R<rsub|X>> reactants (or referred to as <math|N>-body reactions,
  where <math|N> is indeed <math|R<rsub|X>>), the Kratos-readable pre-factor
  <math|<wide|A|~>> should be converted from <math|A> as,

  <\equation>
    <wide|A|~>\<equiv\>A m<rsub|A><rsup|-<around*|(|R<rsub|X>-1|)>>=A
    N<rsub|A><rsup|R<rsub|X>-1> <around*|(|in CGS|)>,
  </equation>

  where <math|N<rsub|A>> is the Avogadro constant and
  <math|N<rsub|A>\<equiv\>m<rsub|A><rsup|-1>> in CGS. Note that, for
  single-body reactions (thermal dissociation, for example), the conversion
  does NOT change the pre-factor <math|A>. With this conversion, the rate
  coefficient within the Kratos system reads,

  <\equation>
    \<lambda\><rsub|X>=<wide|A|~><around*|(|<frac|T|300
    \<Kappa\>>|)><rsup|\<beta\>>exp<around*|(|-<frac|T|\<gamma\>>|)> .
  </equation>

  <subsection|Thermal energy>

  The mostly used variable for the thermal effects of chemical kinetics is
  the internal energy density <math|\<varepsilon\>> (note that other
  thermodynamic potentials shall <with|font-shape|italic|not> be used, as the
  fluid-related thermodynamic effects, including expansion works, are already
  involved in the hydrodynamic steps and should not be double-counted).
  Assuming the energy released by each of the <math|X>th reaction is
  <math|e<rsub|X>>, one gets,

  <\equation>
    <around*|(|<frac|\<mathd\>\<varepsilon\>|\<mathd\>t>|)><rsub|X>=e<rsub|X>
    k<rsub|X><big|prod><rsub|j=0><rsup|R<rsub|X>-1>n<rsub|I<rsub|X,j>> .
  </equation>

  In what follows we shall use specific energy
  <math|E\<equiv\>\<varepsilon\>/\<rho\>> rather than <math|\<varepsilon\>>,
  which yields,

  <\equation>
    <around*|(|<frac|\<mathd\>E|\<mathd\>t>|)><rsub|X>=<frac|e<rsub|X>|\<rho\>>
    k<rsub|X> <big|prod><rsub|j=0><rsup|R<rsub|X>-1><frac|y<rsub|I<rsub|X,j>>\<rho\>|m<rsub|A>>=<frac|e<rsub|X>|m<rsub|A>><around*|(|<frac|\<rho\>|m<rsub|A>>|)><rsup|R<rsub|X>-1>k<rsub|X><big|prod><rsub|j=0><rsup|R<rsub|X>-1>y<rsub|I<rsub|X,j>>=<frac|e<rsub|X>|m<rsub|A>>f<rsub|X>.
  </equation>

  <subsection|Jacobians>

  We should also take care of the Jacobian matrices for the reactions for
  (semi-)implicit solvers. The derivatives each reaction contributing to the
  Jacobian is divided into two parts. The first part is about the specific
  internal energy (<math|C<rsub|V>\<equiv\>E/T> is the specific heat
  capacity),

  <\equation>
    \<xi\><rsub|X,i><rsup|-1><around*|(|<frac|\<partial\>|\<partial\>E>|)><rsub|y<rsub|k>><around*|(|<frac|\<mathd\>y<rsub|i>|\<mathd\>t>|)><rsub|X>\<equiv\><around*|(|<frac|\<partial\>f<rsub|X>|\<partial\>E>|)><rsub|y<rsub|k>>=<frac|f<rsub|X>|\<lambda\><rsub|X>><around*|(|<frac|\<partial\>\<lambda\><rsub|X>|\<partial\>E>|)><rsub|y<rsub|k>>=<frac|f<rsub|X>|E><around*|(|<frac|\<partial\>
    ln E|\<partial\> ln T>|)><rsub|y<rsub|k>><rsup|-1><around*|(|<frac|\<partial\>
    ln\<lambda\><rsub|X>|\<partial\> ln T>|)><rsub|y<rsub|k>> .
  </equation>

  And the abundance part should also involve their impacts on the heat
  capacity,

  <\equation>
    <around*|(|<frac|\<partial\>f<rsub|X>|\<partial\>y<rsub|k>>|)><rsub|E>=<around*|(|<frac|\<partial\>f<rsub|X>|\<partial\>y<rsub|k>>|)><rsub|T>+<around*|(|<frac|\<partial\>f<rsub|X>|\<partial\>T>|)><rsub|y<rsub|k>><around*|(|<frac|\<partial\>T|\<partial\>y<rsub|k>>|)><rsub|E>=<around*|(|<frac|\<partial\>f<rsub|X>|\<partial\>y<rsub|k>>|)><rsub|T>-<around*|(|<frac|\<partial\>f<rsub|X>|\<partial\>T>|)><rsub|y<rsub|k>><frac|<around*|(|\<partial\>E/\<partial\>y<rsub|k>|)><rsub|T>|<around*|(|\<partial\>E/\<partial\>T|)><rsub|y<rsub|k>>>
    .
  </equation>

  The first term on the RHS of the equation above can be equated to,

  <\equation>
    <around*|(|<frac|\<partial\>f<rsub|X>|\<partial\>y<rsub|k>>|)><rsub|T>=<big|sum><rsub|j=0><rsup|R<rsub|X>-1><frac|f<rsub|X>\<delta\><rsub|k,I<rsub|X,j>>|y<rsub|k>>
    .
  </equation>

  The second term can be evaluated for the pre-factor:

  <\equation>
    <around*|(|<frac|\<partial\>f<rsub|X>|\<partial\>T>|)><rsub|y<rsub|k>>=<frac|f<rsub|X>|T><around*|(|<frac|\<partial\>
    ln f<rsub|X>|\<partial\> ln T>|)><rsub|y<rsub|k>>=<frac|f<rsub|X>|T><around*|(|<frac|\<mathd\>
    ln \<lambda\><rsub|X>|\<mathd\> ln T>|)> ,
  </equation>

  and the denominator is related to the specific heat capacity by,

  <\equation>
    <around*|(|<frac|\<partial\> ln E|\<partial\> ln
    T>|)><rsub|y<rsub|k>>=<around*|[|1+<around*|(|<frac|\<partial\> ln
    C<rsub|V>|\<partial\> ln T>|)><rsub|y<rsub|k>>|]> .
  </equation>

  The whole term then becomes,

  <\equation>
    <around*|(|<frac|\<partial\>f<rsub|X>|\<partial\>T>|)><rsub|y<rsub|k>><frac|<around*|(|\<partial\>E/\<partial\>y<rsub|k>|)><rsub|T>|<around*|(|\<partial\>E/\<partial\>T|)><rsub|y<rsub|k>>>=<around*|(|<frac|\<partial\>f<rsub|X>|\<partial\>E>|)><rsub|y<rsub|k>><around*|(|<frac|\<partial\>
    E|\<partial\> y<rsub|k>>|)><rsub|T> .
  </equation>

  For ideal gases, <math|<around*|(|\<partial\> ln E/\<partial\> ln
  T|)><rsub|y<rsub|k>>=1> since <math|<around*|(|\<partial\> ln
  C<rsub|V>/\<partial\> ln T|)><rsub|y<rsub|k>>=0>. The interface we leave in
  the EoS is indeed <math|<around*|(|\<partial\>\<varepsilon\>/\<partial\>n<rsub|k>|)><rsub|T>>,
  which is equated to <math|<around*|(|\<partial\>E/\<partial\>y<rsub|k>|)><rsub|T>>
  by,

  <\equation>
    <around*|(|<frac|\<partial\>E|\<partial\>y<rsub|k>>|)><rsub|T>=<around*|(|<frac|\<partial\>\<varepsilon\>|\<partial\>n>|)><rsub|T><around*|(|<frac|1|\<rho\>>|)><around*|(|<frac|\<rho\>|m<rsub|A>>|)>=m<rsub|A><rsup|-1><around*|(|<frac|\<partial\>\<varepsilon\>|\<partial\>n<rsub|k>>|)><rsub|T>
    .
  </equation>
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
    <associate|preamble|false>
    <associate|prog-scripts|python>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|1.1|1>>
    <associate|auto-3|<tuple|1.2|1>>
    <associate|auto-4|<tuple|1.3|2>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Conversions
      of variables> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>Abundances of chemical
      species <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|1tab>|1.2<space|2spc>Thermal energy
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|1.3<space|2spc>Jacobians
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>
    </associate>
  </collection>
</auxiliary>