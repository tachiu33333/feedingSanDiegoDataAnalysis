Prerequisite: User will need the feeding san diego data. Particularly several xlsx files and one csv file.
The volunteer data (csv file) name must be 'new_volunteer_data.csv', if you want the base code to work. The waste log (xlsx files) will have their own way to be included.

First download the requirement.txt

To run the code, simply replace the volunteer data at line 124 with the data that you have, and then run feeding_san_diego_produce_waste_number_two.py. afterward run app.py using the new drop_predictor.pkl that the earlier file should have printed using the command line: streamlit run app.py.

#important for waste log updates#
If you would like to update or change the waste data, there is a few changes that must be made. I created a function called 'fix_waste_log()' who sole purpose is to give any waste log a date, and reconfigure it to make it readable. To delete past data, just comment or remove the function 'set_up_waste_log()' and instead create your own.


Update 5/2: Created an app to read the predictions seamlessly. I wanted to make sure that anyone can understand where these information goes into.
