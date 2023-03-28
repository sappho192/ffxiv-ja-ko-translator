# ffxiv-ja-ko-translator

Japanese→Korean Translator specialized in Final Fantasy XIV

**FINAL FANTASY is a registered trademark of Square Enix Holdings Co., Ltd.**

## Description

This project is started to solve the [[issue](https://github.com/sappho192/IronworksTranslator/issues/45)] in IronworksTranslator, which is to provide more accurate translation result in Final Fantasy XIV game chat.

Papago and DeepL can be a great choice in common situation, but not for the text in specific game. So I'm trying to make the alternative to help people who want to get better understand and communication in Japanese game.

## Limitations of this project

Since the main goal of this project is to help Koreans communicate in Japanese games, so I'm not considering other languages. However, I believe you can use the structure of this project to create your own translator for your own language combinations.

## Goal of this project

### 1. Proof of Concept Phase 1

* [ ] A model trained on a small amount of game terms is able to correctly translate the same terms
* [ ] Somewhat translate sentences that contain some game terms

### 2. Proof of Concept Phase 2

* [ ] Properly translate sentences that contain some game terms
* [ ] Somewhat translate sentences that contain most of game terms

### 3. Beta Phase

* [ ] Properly translate sentences that contain most of game terms

### 4. Future Phase

* [ ] Train a model which can interactively help understanding the Japanese game chat (like a ChatGPT or Bing chatbot)

## Indication of dataset sources

### Helsinki-NLP/tatoeba_mt

The translator model trained in this repository used `jpn-kor` sub-dataset in [[Helsinki-NLP/tatoeba_mt](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt)]. This dataset is shared under the [[CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/)] licence.

### In-game Auto-Translate sentences in FFXIV

`© SQUARE ENIX CO., LTD. All Rights Reserved.`

> The **auto-translator** is a feature in *[Final Fantasy XIV: A Realm Reborn](https://ffxiv.fandom.com/wiki/Final_Fantasy_XIV:_A_Realm_Reborn "Final Fantasy XIV: A Realm Reborn")* that auto-translates text into whatever language a player's client is set to. 
>
> *From [[Final Fantasy XIV: A Realm Reborn Wiki](https://ffxiv.fandom.com/wiki/Auto-translator)] ([CC BY-SA 3.0](https://www.fandom.com/licensing))*

Since the Auto-Translate words and sentences contain essential terms mainly used in the game, I used this dataset as a primary source to accurately train the model.

According to the Materials Usage License ([[EN](https://support.na.square-enix.com/rule.php?id=5382&tag=authc)] [[JP](https://support.jp.square-enix.com/rule.php?id=5381&la=0&tag=authc)]) of Final Fantasy XIV, I can use `All art, text, logos, videos, screenshots, images, sounds, music and recordings from FFXIV` without `any sales or commercial use` and `license fees or advertising revenue`, but even so, I `must immediately comply with any request by Square Enix to remove any Materials, in Square Enix's sole discretion`.

Based on above condition, I have gathered Auto-Translate text ① I see in the game myself, ② referring fandom wiki page [[eLeMeN - FF14 - その他_定型文辞書](http://www5.plala.or.jp/SQR/ff14/etc/dictionary/)].  
When I label the each text, I mainly used machine translation (DeepL, Papago, etc.) but some translation was done by myself and an acquaintance. Proofreading for translation quality was done in same condition.

## Releasing the dataset
I'm going to release the dataset to clarify the [Indication of dataset sources](#indication-of-dataset-sources) under fulfilling TOS or license of each sources.
