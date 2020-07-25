import os
import extract_msg
import numpy as np

class ParseEmails():

    def __init__(self, parent_folder):
        self.parent_folder = parent_folder
        self.dic_dataset = dict(inputs=[], targets=[])

    def parse_content(self, path_email):
        msg = extract_msg.Message(path_email)
        msg_sender = msg.sender
        msg_date = msg.date
        msg_subj = msg.subject
        msg_body = msg.body
        #print('Sender: {}'.format(msg_sender))
        #print('Sent On: {}'.format(msg_date))
        #print('Subject: {}'.format(msg_subj))
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        #print('Body: {}'.format(msg_body))
        pos_last_occurence_subject = msg_body.rindex("Subject:")
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Subject>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        last_subject = msg_body[pos_last_occurence_subject:]
        array_lines = last_subject.split('\n')
        summary = array_lines[0][9:]
        array_text = array_lines[1:]
        text = " ".join(array_text).replace("\r", "")
        #print("last_subject: "+ last_subject)
        self.dic_dataset["inputs"].append(text)
        self.dic_dataset["targets"].append(summary)

    def run(self):
        for folder_emails in os.listdir(self.parent_folder):
            path_emails_folder = os.path.join(self.parent_folder, folder_emails)
            if folder_emails != ".DS_Store":
                for file_email in os.listdir(path_emails_folder):
                    if file_email != ".DS_Store":
                        path_email = os.path.join(path_emails_folder, file_email)
                        self.parse_content(path_email)
                        print(path_email)
        return self.dic_dataset

if __name__ == "__main__":

    parent_folder = "/Users/c325018/ComplaintsProjects/MAIL/"
    pe = ParseEmails(parent_folder)
    dictionary_dataset = pe.run()
    # Save
    np.save('dic_data.npy', dictionary_dataset)
    # Load
    read_dictionary = np.load('dic_data.npy', allow_pickle='TRUE').item()
    print(read_dictionary)

