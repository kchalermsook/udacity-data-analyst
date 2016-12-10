# Summary

This project explains the survival rate on the Titanic using stacked barchart with Dimple.js. It shows the number of survivals and non-survivals by seat class in absolute value mode and percentage mode. 
The viewers can choose to filter passengers by Child Male, Child Female, Men and Woman.

# Design

First of all, I downloaded the Titanic dataset from the link given by Udacity. Then, I opened it and see the raw data in excel. 

I looked through all the data and selected the fields "Survived", "Pclass", "Sex" and "Age". The "Survived" field is quite clear that it is the main field that we want to know. Next, based on the Titanic movie, we saw that the classes of tickets, sex and ages are the choices that the Titanic's officers took into account when they decided to help any passengers. For me, I think the classes should be the most interesting factor that can determine the survival rate. So, I put the classes in x-axis and put the number of survivals on the y-axis using stacked barchart.

After that, I think the easiest way to illustrate the survival rate should be not only showing the count, I should also show the chart in percentage of survivals and non-survivals. So, I added the "mode" button on the right side of the graph so that the viewers can choose to display the graph by count or percentage. Next, I chose the field "Sex" and "Age" for filtering purpose. But I needed to clean up the data a little bit because some rows had empty Age field. I added the code to filter out and select only rows with non-empty Age. 

I chose to have 3 filters button and added more code to add field 'AgeType' into data array using below condition.
- "Child" = passengers with age lower that 12
- "Men" = passengers with age greater that 12 and sex = male
- "Women" = passengers with age greater that 12 and sex = female

Then I built the graph which can filter with conditions above (index.html), sent to friends and get response from them. 

After I got feedback, I decided to changed the code based on their feedbacks.
- I added "Select Passenger Type" and "Select Graph Mode" into the right side of the graph
- Added text into the bar to show count and percentage.
- Made the filter to be tick and non-tick method and separate Child into "Child Male" and "Child Female"

# Feedback

## Response 1 

"The overall of the graph looked quite cool, very easy to understand and good filter concept. The main problem that I saw is it is quite not clear in the buttons sections. You should have some labels for that buttons"

## Response 2 

"The main focus is I would like to see the detail in the graph without the need to use mouse to hover, for example, you may put the count inside the graph because there is an area left to put the text."

## Response 3 

"I really like the interactive that we can filter the graph by gender and child. How about the gender of child? any way to show it? Can you also make more groups of the passengers such as Old Female and Old Male."

# Resources

- http://dimplejs.org/
- https://d3js.org/