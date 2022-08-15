import os
import schedule
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import json
from datetime import datetime
from datetime import timedelta
import math
from sklearn.linear_model import LinearRegression
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import matplotlib.backends.backend_pdf
from datatools import get_nodes_by_serial, historical_measurements, to_dataframe
import smtplib, ssl, email
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def job():
    
    # Assigning EC & VWC at certain depths with colors for plotting purposes
    color_dict_VWC = {15 : "blue", 30 : "orange", 45 : "green", 60 : "red"}
    color_dict_EC = {15 : "red", 30 : "green", 45 : "orange", 60 : "blue"}

    today = pd.Timestamp(datetime.today().date()) 

    prior_five = pd.Timestamp(today - timedelta(days = 5)) # 5 days of data - data that is going to be predicted

    prior_six = pd.Timestamp(today - timedelta(days = 6))

    prior_seven = prior_five - timedelta(days = 5)

    pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")

    for i in range(0, len(current)):

        sensor = current.iloc[i]

        try:

            historical = historical_measurements(sensor['sensor'], sensor['startTime'], int(time.time()*1000), raw_vwc=True)

            data = to_dataframe(historical, with_raw_vwc=True)

            data['Date'] = pd.to_datetime(data['DT']).dt.date

            data['Time'] = pd.to_datetime(data['DT']).dt.time

            data = data.set_index('DT')

    #         data = data[(data.Date >= prior_sixty) & (data.Date <= today)]

            if datetime.today().date() in set(data.Date.values):

                print('----------------------------------------------------------------------------------------------------------------------------------------------------')
                print('Farm: {}, Sensor: {}'.format(sensor['name'], sensor['sensor']))
                print()

            else:
                continue

            # Predictions & Plots
            for key, grp in data.groupby(['Depth']):

                all_days = grp[(grp.Date >= prior_seven) & (grp.Date <= today)]

                train = all_days[(all_days.Date >= prior_seven) & (all_days.Date < prior_five)] # data which we use to determine the correlation between EC and VWC

                test = all_days[(all_days.Date >= prior_five)] # data we are predicting, which includes the latest data for today

                ### Testing ###

                bound_days = all_days[all_days.Date >= prior_six]

                date_before = bound_days[bound_days.Date == prior_six].iloc[-1:]

                bound_days = bound_days[bound_days.index >= date_before.index[0]]

                ### Testing ###

                train_x = train.VWC.values.reshape(-1,1)
                train_y = train.EC.values.reshape(-1,1)

                test_x = test.VWC.values.reshape(-1,1)
                test_y = test.EC.values.reshape(-1,1)

                reg = LinearRegression()

                reg.fit(train_x, train_y)

                c = reg.intercept_[0]
                a = reg.coef_.ravel().tolist()[0]

                futureEC = c + a*test.VWC

                ub = []
                lb = []

                predEC = [train.iloc[len(train)-1].EC] + list(futureEC)

                upperI = predEC[0]
                lowerI = predEC[0]

                print('EC Predictions for a depth of {}:'.format(key))
                print()
                print('intercept: {}' .format(c))
                print('alpha: {}' .format(a))
                print()

                exp = []

                # Gradually adds/subtracts errors to the predictions on each day            
                ub.append(upperI)
                lb.append(lowerI)

                multipleU = 1.15
                multipleL = 0.85

                for prds in range(0, len(futureEC)):

                    upperI = multipleU*futureEC[prds]

                    if test.EC[prds] >= upperI:

                        exp.append('ECs are potentially concerning for farmer {}, sensor {} at depth {} on {}'.format(sensor['name'], sensor['sensor'], key, test.index[prds]))

                    ub.append(upperI)

                    lowerI = multipleL*futureEC[prds]

                    lb.append(lowerI)

                    multipleU *= 1.001
                    multipleL *= 0.999

                if len(exp) != 0:

                    latest_concern = exp[len(exp)-1]

                else:
                    latest_concern = 'No Concerns'         

                # Historical Plots of EC & VWC (along with predictions for the period of interest)

                fig = plt.figure(figsize = (20, 10))

                ax = fig.add_subplot(211)

                ax.set_ylabel('VWC', color = 'black') 
                p1 = ax.plot(grp.index, grp['VWC'], label = 'VWC_With_Offsets_' + str(key), color = color_dict_VWC[key], linewidth=1.5)

                ax2 = ax.twinx()

                ax2.set_ylabel('EC', color = 'black')
                p3 = ax2.plot(grp.index, grp['EC'], label = 'EC_' + str(key) + ' Actuals', color = color_dict_EC[key], linewidth = 1.5)           

                lns = p1 + p3
                labels = [l.get_label() for l in lns]
                plt.legend(lns, labels, loc=0)

                plt.grid(b=True)
                plt.tight_layout()
                plt.title('Historical ECs & VWCs: Farmer - {}, Sensor - {} & Depth - {}'.format(sensor['name'], sensor['sensor'], key))

                ax3 = fig.add_subplot(212)

                ax3.set_xlabel('Time') 
                ax3.set_ylabel('VWC', color = 'black') 
                p1 = ax3.plot(all_days.index, all_days['VWC'], label = 'VWC_With_Offsets_' + str(key), color = color_dict_VWC[key], linewidth=1.5)

                ax4 = ax3.twinx()

                ax4.set_ylabel('EC', color = 'black')
                p3 = ax4.plot(all_days.index, all_days['EC'], label = 'EC_' + str(key) + ' Actuals', color = color_dict_EC[key], linewidth = 1.5)           
                p4 = ax4.plot(bound_days.index, predEC, label = 'EC_' + str(key) + ' Predictions', linestyle = '--', color = color_dict_EC[key], linewidth = 1.5)

                ax4.fill_between(bound_days.index, ub, lb, alpha = 0.15)

                lns = p1 + p3 + p4
                labels = [l.get_label() for l in lns]
                plt.legend(lns, labels, loc=0)

                plt.grid(b=True)
                plt.tight_layout()
                plt.figtext(0,-0.05,latest_concern, ha = 'left', fontsize=12)
                pdf.savefig(fig, bbox_inches='tight')
                # plt.show()

        except Exception as e:
            print(e)

    pdf.close()
    
    sender_email = "ramnarainnair@gmail.com"
    receiver_email = "ram@agurotech.com"
    password = "eols argc iyfj ojsi"

    #Create MIMEMultipart object
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "EC & VWC Plots"
    msg["From"] = sender_email
    msg["To"] = receiver_email
    filename = "output.pdf"

    #HTML Message Part
    html = """\
    <html>
      <body>
        <p><b>EC & VWC Plots</b>
        <br>
           Attached in this email are the EC & VWC Plots.<br>
        </p>
      </body>
    </html>
    """

    part = MIMEText(html, "html")
    msg.attach(part)

    # Add Attachment
    with open(filename, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    encoders.encode_base64(part)

    # Set mail headers
    part.add_header(
        "Content-Disposition",
        "attachment", filename= filename
    )
    msg.attach(part)

    # Create secure SMTP connection and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(
            sender_email, receiver_email, msg.as_string()
        )
        
        
schedule.every().day.at('22:15').do(job)

while True:
    schedule.run_pending()				
    time.sleep(1)
