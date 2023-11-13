from flask import Flask, render_template, request, jsonify
import re

app = Flask(__name__)

# Function to extract additional annotation types (dates, gender, IP addresses, Credit Card Numbers, names)
def extract_additional_info(text):
    date_regex = r'(\d{1,2}/\d{1,2}/\d{4})|(\d{4}-\d{1,2}-\d{1,2})'
    gender_regex = r'\b(?:male|female)\b'
    ip_address_regex = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    credit_regex = r'\b(?:\d[- ]?){12}\d[- ]?\d[- ]?\d[- ]?\d\b'
    name_regex = r'[A-Z][a-z]+ [A-Z][a-z]+'
    ssn_regex = r'\b(\d{3}-\d{2}-\d{4}|\d{9})\b'
    address_regex = r'\b\d{1,3}\s+[A-Za-z0-9\s]+\s*,*\s*[A-Za-z]+\s*,*\s*[A-Za-z]+\s+\d{5}(?:-\d{4})?\b'

    dates = re.findall(date_regex, text)
    gender = re.findall(gender_regex, text)
    ip_addresses = re.findall(ip_address_regex, text)
    credit = re.findall(credit_regex, text)
    names = re.findall(name_regex, text)
    social_numbers = re.findall(ssn_regex, text)
    address = re.findall(address_regex, text)

    return dates, gender, ip_addresses, credit, names, social_numbers, address

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    text = request.form.get('text')

    # Extract emails and phone numbers as before
    email_regex = r'\S+@\S+'
    phone_regex = r'\+\d{1,3}\s?\d{1,4}[-\s]?\d{1,10}'
    emails = re.findall(email_regex, text)
    phone_numbers = re.findall(phone_regex, text)

    # Extract additional annotation types
    dates, gender, ip_addresses, credit, names, social_numbers, address = extract_additional_info(text)

    # Combine all extracted information into a dictionary
    extracted_info = {
        'emails': emails,
        'phone_numbers': phone_numbers,
        'dates': dates,
        'gender': gender,
        'ip_addresses': ip_addresses,
        'credit': credit,
        'names': names,
        'social_numbers' : social_numbers,
        'address' : address,
    }

    return jsonify(extracted_info)

if __name__ == '__main__':
    app.run(debug=True)
