\# Redfin Explorer 🏠📈

A lightweight \*\*Plotly Dash\*\* dashboard for exploring Redfin listing exports with filters, charts, and a “deal” signal based on \*\*low $/sqft\*\*.



This app is designed for quick local analysis: download a Redfin CSV, point the app at it, and explore price, $/sqft, beds/baths, age, status, and location—fast.



---



\## Features

\- \*\*Powerful filters\*\*: status, city, property type, price range, sqft, beds, baths, year built, days on market

\- \*\*Deal highlighting\*\*: flags listings in the \*\*bottom X percentile of $/sqft\*\* (configurable)

\- \*\*Deals-only toggle\*\*: instantly filter to only flagged “deal” listings

\- \*\*Readable values\*\*:

&nbsp; - Currency formatting for medians and charts

&nbsp; - A live “Price: $X – $Y” readout above the price slider

\- \*\*Charts\*\*

&nbsp; - $/sqft distribution (with deal threshold)

&nbsp; - Price vs. square feet (deals highlighted)

&nbsp; - $/sqft vs. age

\- \*\*Interactive map\*\*

&nbsp; - Zoom + pan (scrollwheel supported)

&nbsp; - Preserves your map view when filters change

\- \*\*Clickable listing links\*\* (DataTable URL column opens in a new tab)



---



\## Screenshots

Add screenshots to a `/screenshots` folder and reference them here, e.g.:



```md

!\[Dashboard](screenshots/dashboard.png)



