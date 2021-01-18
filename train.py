import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
from sklearn.model_selection import train_test_split

df = pd.read_csv('news.csv')
df = df.sample(frac=1).reset_index(drop=True)

print(df.shape)


def clean_text(text):
    text = str(text)
    return text


df['title'] = df.apply(lambda x: clean_text(x['title']), axis=1)
df['text'] = df.apply(lambda x: clean_text(x['text']), axis=1)

df["prefix"] = "summarize"
df['input_text'] = df['text']
df['target_text'] = df['title']
train_df, eval_df = train_test_split(df, test_size=0.05, random_state=42)

print(df['input_text'][0])
print(df['target_text'][0])

model_args = T5Args()
model_args.max_seq_length = 256
model_args.train_batch_size = 4
model_args.eval_batch_size = 4
model_args.num_train_epochs = 1
model_args.evaluate_during_training = False
model_args.use_multiprocessing = False
model_args.fp16 = False
model_args.save_steps = -1
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = False
model_args.no_cache = True
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_return_sequences = 1

model = T5Model("mt5", "google/mt5-small", args=model_args)

model.train_model(train_df, eval_data=eval_df)

to_predict = [
    "summarize: На недавно запущенной в Белоруссии атомной электростанции из-за срабатывания системы защиты генератора"
    " отключили первый энергоблок. Об этом в своем Telegram-канале сообщило Минэнерго республики.«Система защиты "
    "сработала в ходе опытно-промышленной эксплуатации первого энергоблока, в рамках которого проводится тестирование "
    "систем и оборудования», — заявили в ведомстве. По его информации, радиационный фон в районе станции находится "
    "в норме. БелАЭС запустили 7 ноября 2020 года. Однако спустя три дня станция временно прекратила выработку "
    "энергии. В сообщении на сайте АЭС говорилось, что в ходе пусконаладочных работ на станции выявили необходимость "
    "заменить отдельное электротехническое оборудование. В Госатомнадзоре Белоруссии заверили, что это решение "
    "не повлияет на радиационную безопасность."
]

predictions = model.predict(to_predict)
print(predictions)

