import yagmail

def sendEmail(name):
    yag = yagmail.SMTPcom("detectionfacemask@gmail.com")
    contents = f"{name}, please wear a mask"
    subject = "Mask Warning"
    if name == 'Rakesh':
        yag.send('bandirakesh98@gmail.com', subject, contents)
