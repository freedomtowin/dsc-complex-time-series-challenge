## How signal processing can be used to identify patterns in complex time series

This was a fun, machine learning side project so I didn't have any business context to do it. Originally, I included links and references, but they were against the "one link" rule of posting blogs on here. Also, I wanted to included additional material, but I think there is a limit to how many pics I can post. I will try to show what objective I was trying to accomplish.

#### The original problem statement was given here: http://www.datasciencecentral.com/forum/topics/challenge-of-the-week-identifying-patterns-in-complex-time-series
There was also a verbal solution given in the members only section. I'm not sure if its legal to share the whole thing, but here is an excerpt of the solution. " The time series has a weekly periodicity with two peaks: Monday and Thursday, corresponding respectively to the publication of the Monday and Thursday digests. The impact of the Monday and Thursday email blasts extent over the next day; this makes measuring the yield more difficult, unless you use additional data, e.g. from our newsletter vendor. However, the bulk of the impact is really on Monday and Thursday."

I saw a DSC article that talked about finding trends using signal processing techniques. http://www.datasciencecentral.com/profiles/blogs/how-we-combined-different-methods-to-create-advanced-time-series . The trend component could be created and entered into the regression model as an independent variable. I should the trend component in one of the figures above. I think it wouldn't make sense to reuse frequencies components from the trend component because their periods (cycles/seconds) are very large (upwards of 200 days).

The trend and seasonality can be accounted for in a linear model by including sinusoidal components with a given frequency. However, finding the appropriate frequency for each sinusoidal component requires a little more digging. This post shows how to use fast Fourier transforms to find these frequencies.

#### Defining the model:
y = P(t) + S(t) + T(t) + R(t)

P(t)~Polynomial component
S(t)~Seasonal component
T(t)~Trend component
R(t)~Residual error

For the purposes of this post, we will only focus on the T(t) and S(t) components. The actual model fitting will be done in a separate post.

600 observations were used in the training set. The result was tested on the full dataset with 731 observations.

<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22960357/a9e2d77c-f302-11e6-83f7-f8a218b88e06.png"  height="300px" align="bottom" hspace="250">
</p>

#### Find the overall trend:
I used an FFT transformation to visualize the magnitude of the frequency components in the time series. To be specific, the absolute magnitude is plotted.

<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22960372/ba10e31e-f302-11e6-84c9-d8a5cd8f61ed.png"  height="300px" align="bottom" hspace="250">
</p>

Frequency Component, Magnitude

[  1.41666667e-01   1.82239797e+05]

[  1.43333333e-01   5.67160341e+05]

[  2.83333333e-01   1.66899918e+05]

[  2.85000000e-01   4.59942544e+05]

[  2.86666667e-01   3.95441559e+05]

[  4.28333333e-01   2.03492985e+05]

#### Does it make sense to reuse frequencies for the trend and seasonal components?
On one hand, it might be better not to miss anything. I doubt there will be a prominant trend for -a weekday- every 28 weeks.
For the trend component, it would makes sense to use the lowest frequencies with the highest magnitudes.

I found dominant frequencies at .143, .285, and .428. These correspond to T=7.14,3.5, and 2.33. The reason why those frequencies are interesting is because they are "sticking out." Which is shows there is probably a seasonal component that has a reasonable period, T (cycles/second). There were also some frequencies around the e-3 orders of magnitude. These were at .00166, .00333, and 0.005, T>200, which were included in the trend.


<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22960376/c0ed8f34-f302-11e6-9c11-811685606545.png"  height="300px" align="bottom" hspace="250">
</p>

<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22960379/c7549692-f302-11e6-935e-29271041eef0.png"  height="300px" align="bottom" hspace="250">
</p>

This trend component would be entered into the regression model as an independent variable.

#### Finding seasonal patterns in the target variable:
The overall trend could be removed by creating a differenced variable for Pageviews The differenced variable allows for seasonal components to be identified more clearly.

<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22960381/cd49520e-f302-11e6-945b-6c26f899ee4e.png"  height="300px" align="bottom" hspace="250">
</p>

<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22960388/d31bb118-f302-11e6-8a57-8800cead4425.png"  height="300px" align="bottom" hspace="250">
</p>


Frequency Component, Magnitude

[  1.43333333e-01   5.00831933e+05]

[  2.83333333e-01   2.65832489e+05] 

[  2.85000000e-01   7.24904464e+05] 

[  2.86666667e-01   6.13035227e+05] 

[  2.88333333e-01   1.92922452e+05] 

[  4.28333333e-01   4.04206565e+05]

The lower frequency components were removed and the other, distinct frequencies were amplified. This makes the frequencies easier to filter! Also it makes it easier to compare to possible seasonal variables.
 
#### Finding the seasonal predictor variable:

<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22960389/d8bdb8aa-f302-11e6-9056-d781a2dd8b34.png"  height="300px" align="bottom" hspace="250">
</p>


Frequency Component, Magnitude

[  1.41666667e-01   2.42782136e+02] 

[  1.43333333e-01   6.00386477e+02] 

[  1.45000000e-01   1.31981640e+02] 

[  2.85000000e-01   2.78344410e+02] 

[  2.86666667e-01   2.07887576e+02] 

[  4.28333333e-01   2.97539156e+02]

#### Eureka! Weekday shares the same frequency components as Pageviews!
I found dominant frequencies at .143, .285, and .428. These correspond to T=7.14,3.5, and 2.33. There were also some frequencies around the e-3 orders of magnitude. These were at .00166, .00333, and 0.005 and had periods upwards of 200. 
If you want to see how I included these frequency components in a regression model please see my Github. The results are compared to straight up dummy coding (the results are the same).  


This table shows the index, weekday, and other time variables that will help me illustrate what I mean by periods or cycles.  My index starts at 1, but this corresponds to weekday 3 (or Wednesday). So index 6 corresponds to Monday and index 2 corresponds to Thursday. 

<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22960618/ce7fe118-f304-11e6-8a45-d4606b04b4b3.png"  height="300px" align="bottom" hspace="250">
</p>

#### Fitting the Model

I will include a P(t) component as the squared value of the index, sq_index, the trend component as T(t), and the seasonal components as S(t). Since T(t) and P(t) are already created, this part of the blog will focus on S(t).

<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22960693/812d6646-f305-11e6-8973-a9f4dbe269ac.png"  height="300px" align="bottom" hspace="250">
</p>

Each seasonal component is made up of two sinusoidal waves, i.e. β1*sin(t/T) + β2*cos(t/T). The cosine term is include to account for phase shift. The beta coefficients are the estimated parameter weights for each sinusoid. Since 3 frequencies are included in the model, there will be 6 sinusoids and 6 beta coefficients to estimate. For the time component, t, I used the actual "weekday" variable.

In the regression model, each of these components will get a weight associated with it. When all of them are added up together with their respective weights, you should see a pattern similar to Pageviews.

<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22960739/f75b0044-f305-11e6-9bd3-507a8a1d0591.png"  height="300px" align="bottom" hspace="250">
</p>

<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22960753/12203d90-f306-11e6-95f5-ea15621b8417.png"  height="300px" align="bottom" hspace="250">
</p>

There are two peaks at indexes 2 and 6 which correspond to Monday and Thursday. This is consistent with the problem solution given above.

The model was fit on the training set with 600 observations. Its interesting that the model fitted with sinusoids gave almost the same R-squared value as including a dummy coded weekday variable in the model. The coefficients are obviously much different, but the results are eerily similar.


<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22960805/78ab44d8-f306-11e6-8b2f-fe767ac09aa1.png"  height="300px" align="bottom" hspace="10">
</p>

#### Model fitted with sinusoid terms:

SSE: 1824619798.78
 
SST: 9765197379.99
 
R^2: 0.813150750796

#### Coefficients:

[ 8.10030680e+03]

[ 1.55909280e+03]

[ 2.23320858e+03]

[ 1.52632878e+02]

[ -1.26500151e+03]

[ -3.93921325e+02]

[ -7.03090816e+02]

[ 7.94011068e-01]

[ 2.50742897e-02]


#### Model fitted with dummy coded results

SSE: 1824619798.78
 
SST: 9765197379.99
 
R^2: 0.813150750796

#### Coefficients:

[ 1.15000991e+04]

[ -2.75598163e+03]

[ -3.28854678e+03]

[ -1.44911713e+03]

[ -3.65450388e+03]

[ -6.91319555e+03]

[ -5.75584347e+03]

[ 7.94011068e-01]

[ 2.50742897e-02]


<p> <img src="https://cloud.githubusercontent.com/assets/21232362/22960929/6f849642-f307-11e6-9b42-904412dc8633.png"  height="300px" align="bottom" hspace="250">
</p>


<p> <img src="https://cloud.githubusercontent.com/assets/21232362/22960934/7e9e7a3a-f307-11e6-96c4-b34a64791b40.png"  height="300px" align="bottom" hspace="250">
</p>

There is a slightly overall trend in the residual plot. Including lower frequency components could help stablize errors.

#### The results are almost (if not) exactly the same, so which method to choose?

Dummy coding is one thing that can be automated, but its not always easy to type every variable to include in a linear model. The advantage of  dummy coding is that its more difficult to go wrong with because each level is accounted for. Using signal processing, frequencies need to be chosen for the seasonal components which requires some inspection and maybe some trail and error. The plus side is that you can choose which periods make sense and you can choose frequencies for trend component while you work on the seasonal part.

All in all, both methods ended up with the same number of parameters and had the same performance.
