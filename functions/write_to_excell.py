data_frame_data = []
for i, person in enumerate(persons[:100]):
    print(person)
    data_frame_data.append(person)
    for j, pred in enumerate(prediction[i]):
        if pred:
            if labels[j] == 'other':
                print('other')
                data_frame_data.append('Other')
            else:
                data_frame_data.append(translate_label_dict[labels[j]])
                print(translate_label_dict[labels[j]])

df = pd.DataFrame({'Person and Precited job': data_frame_data})
writer = ExcelWriter('person_and_predicted.xlsx')
df.to_excel(writer,'Sheet1',index=False)
writer.save()
