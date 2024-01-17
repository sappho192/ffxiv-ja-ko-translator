﻿using CsBindgen;
using FastBertTokenizer;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using NumSharp;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace infer_onnx
{
    public class TestFunctions
    {
        private readonly BertTokenizer srcTokenizer = new();
        private readonly KoGPT2Tokenizer trgTokenizer = new();

        public TestFunctions()
        {
            //tok.LoadFromHuggingFaceAsync("bert-base-uncased").GetAwaiter().GetResult();
            var vocab = File.OpenText("vocab.txt");
            srcTokenizer.LoadVocabulary(vocab, false);
        }

        public void TestBertJapanese(string text)
        {
            Console.WriteLine("Testing encoder(bert-base-japanese-v2)");
            
            //var tokenizerJson = File.OpenText("D:\\DATA\\tokenizer.json");
            //tok.LoadTokenizerJson(tokenizerJson.BaseStream);
            // Full-width symbols should be converted to half-width symbols
            // Example: ！ -> !
            // List of full-width symbols are: https://en.wikipedia.org/wiki/Halfwidth_and_fullwidth_forms
            var (inputIds, attentionMask, tokenTypeIds) = srcTokenizer.Encode(text);
            Console.WriteLine(string.Join(", ", inputIds.ToArray()));
            var decoded = srcTokenizer.Decode(inputIds.Span);
            Console.WriteLine(decoded);
        }

        public void TestKoGPT2()
        {
            Console.WriteLine("Testing dcoder(kogpt2-base-v2)");
            unsafe
            {
                var data = new uint[] { 25906, 8702, 7801, 25856 };
                Console.WriteLine($"Input tokens: {string.Join(", ", data)}");
                fixed (uint* p = data)
                {
                    var decoded = NativeMethods.tokenizer_decode(p, data.Length);
                    try
                    {
                        var str = Encoding.UTF8.GetString(decoded->AsSpan());
                        Console.WriteLine(str);
                    }
                    finally
                    {
                        NativeMethods.free_u8_string(decoded);
                    }
                }
            }
        }

        public void TestONNX()
        {
            const string encoderPath = @"D:\REPO\onnx-hf-test\python\onnx\encoder_model.onnx";
            const string decoderPath = @"D:\REPO\onnx-hf-test\python\onnx\decoder_model_merged.onnx";
            var sessionOptions = new SessionOptions { 
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
            };
            var encoderSession = new InferenceSession(encoderPath, sessionOptions);
            var decoderSession = new InferenceSession(decoderPath, sessionOptions);

            var inputText = "逃げろ!";
            Console.WriteLine($"Input text: {inputText}");
            string generatedText = Translate(encoderSession, decoderSession, inputText);
            Console.WriteLine($"Translated text: {generatedText}");
        }

        private string Translate(InferenceSession encoderSession, InferenceSession decoderSession, string inputText)
        {
            var (inputIds, attentionMask, tokenTypeIds) = srcTokenizer.Encode(inputText);
            Console.WriteLine($"Input tokens: {string.Join(", ", inputIds.ToArray())}");

            // inputIds to NDArray
            NDArray ndInputIds = np.array(inputIds.ToArray());
            ndInputIds = np.expand_dims(ndInputIds, 0);
            NDArray ndAttentionMask = np.array(attentionMask.ToArray());
            ndAttentionMask = np.expand_dims(ndAttentionMask, 0);

            var encoderInput = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", ndInputIds.ToMuliDimArray<long>().ToTensor<long>()),
                NamedOnnxValue.CreateFromTensor("attention_mask", ndAttentionMask.ToMuliDimArray<long>().ToTensor<long>())
            };

            var encoderResults = encoderSession.Run(encoderInput);
            var encoderResult = encoderResults[0];

            var singleBoolArray = new bool[] { false };
            var useCacheBranch = np.array(singleBoolArray);
            //useCacheBranch = np.expand_dims(useCacheBranch, 0);

            //var zeros = np.zeros<float>(1, 12, inputIds.Length, 64);
            var decoderInput = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", ndInputIds.ToMuliDimArray<long>().ToTensor<long>()),
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderResult.AsTensor<float>()),
                NamedOnnxValue.CreateFromTensor("use_cache_branch", useCacheBranch.ToMuliDimArray<bool>().ToTensor<bool>())
            };

            var generatedText = GreedySearch(decoderInput, encoderSession, decoderSession);
            return generatedText;
        }

        private static Tensor<float> InitKeyValues(int inputIdLength)
        {
            return np.zeros<float>(1, 12, inputIdLength, 64).ToMuliDimArray<float>().ToTensor<float>();
        }

        private string GreedySearch(List<NamedOnnxValue> decoderInput, InferenceSession encoderSession, InferenceSession decoderSession, int maxLength = 50)
        {
            // Initialize the input for the decoder with the BOS token ID
            long bosTokenId = 1;
            long eosTokenId = 1;
            var inputIds = new List<long> { bosTokenId };
            var inputIdsTensor = new DenseTensor<long>(inputIds.ToArray(), new[] { 1, 1 });
            // Update input_ids for the decoder
            decoderInput[0] = NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor);

            // Initialize the list to store the generated tokens
            List<long> generatedTokens = [];
            List<DisposableNamedOnnxValue> pastKeyValues = [];
            List<List<DisposableNamedOnnxValue>> packedPastKeyValues = [];

            // Greedy search loop
            for (int i = 0; i < maxLength; i++)
            {
                // Run the decoder model
                (var decoderResults, packedPastKeyValues) = ForwardOnnx(decoderSession, decoderInput, pastKeyValues);

                // Get the logits from the decoder results
                var logits = decoderResults.First().AsTensor<float>();

                // Apply softmax to logits to get probabilities
                var probabilities = Softmax(logits);
                var nextTokenId = ArgMax(probabilities.ToTensor());
                var currentNextTokenId = generatedTokens.LastOrDefault(-1);
                if (currentNextTokenId == nextTokenId)
                {
                    nextTokenId = ArgMax(probabilities.ToTensor(), true);
                }

                // Append the token to the list
                generatedTokens.Add(nextTokenId);

                // Prepare the input for the next iteration
                inputIdsTensor = new DenseTensor<long>(new long[] { nextTokenId }, new[] { 1, 1 });
                decoderInput.Find(input => input.Name.Equals("input_ids")).Value = inputIdsTensor;

                // Update past_key_values with the current output
                //if (decoderResults[1].Value != null)
                //{
                //    decoderInput.Find(input => input.Name.Equals("use_cache_branch")).Value = useCacheBranch.ToMuliDimArray<bool>().ToTensor<bool>();
                //    // Update the past_key_values with the latest decoder output: from 0 to 11
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.0.key")).Value = decoderResults[1].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.0.value")).Value = decoderResults[2].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.1.key")).Value = decoderResults[3].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.1.value")).Value = decoderResults[4].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.2.key")).Value = decoderResults[5].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.2.value")).Value = decoderResults[6].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.3.key")).Value = decoderResults[7].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.3.value")).Value = decoderResults[8].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.4.key")).Value = decoderResults[9].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.4.value")).Value = decoderResults[10].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.5.key")).Value = decoderResults[11].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.5.value")).Value = decoderResults[12].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.6.key")).Value = decoderResults[13].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.6.value")).Value = decoderResults[14].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.7.key")).Value = decoderResults[15].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.7.value")).Value = decoderResults[16].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.8.key")).Value = decoderResults[17].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.8.value")).Value = decoderResults[18].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.9.key")).Value = decoderResults[19].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.9.value")).Value = decoderResults[20].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.10.key")).Value = decoderResults[21].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.10.value")).Value = decoderResults[22].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.11.key")).Value = decoderResults[23].Value;
                //    decoderInput.Find(input => input.Name.Equals("past_key_values.11.value")).Value = decoderResults[24].Value;
                //}

                // Check if EOS token is generated
                if (nextTokenId == eosTokenId)
                {
                    break;
                }
            }

            // Decode the generated tokens into text
            var generatedText = trgTokenizer.Decode(generatedTokens.Select(item => (uint)item).ToArray());
            return generatedText;
        }

        private (IDisposableReadOnlyCollection<DisposableNamedOnnxValue>, List<List<DisposableNamedOnnxValue>>) ForwardOnnx(
            InferenceSession session, List<NamedOnnxValue> input_data, List<DisposableNamedOnnxValue> past_key_values)
        {
            if (past_key_values.Count == 0)
            {
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.0.key", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.0.value", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.1.key", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.1.value", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.2.key", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.2.value", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.3.key", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.3.value", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.4.key", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.4.value", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.5.key", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.5.value", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.6.key", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.6.value", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.7.key", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.7.value", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.8.key", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.8.value", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.9.key", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.9.value", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.10.key", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.10.value", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.11.key", InitKeyValues(1)));
                input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.11.value", InitKeyValues(1)));
            }
            else
            {
                var singleBoolArray = new bool[] { true };
                var useCacheBranch = np.array(singleBoolArray).ToMuliDimArray<bool>().ToTensor<bool>();
                input_data.Find(input => input.Name.Equals("use_cache_branch")).Value = useCacheBranch;
                input_data.Find(input => input.Name.Equals("past_key_values.0.key")).Value = past_key_values[1].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.0.value")).Value = past_key_values[2].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.1.key")).Value = past_key_values[3].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.1.value")).Value = past_key_values[4].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.2.key")).Value = past_key_values[5].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.2.value")).Value = past_key_values[6].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.3.key")).Value = past_key_values[7].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.3.value")).Value = past_key_values[8].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.4.key")).Value = past_key_values[9].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.4.value")).Value = past_key_values[10].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.5.key")).Value = past_key_values[11].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.5.value")).Value = past_key_values[12].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.6.key")).Value = past_key_values[13].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.6.value")).Value = past_key_values[14].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.7.key")).Value = past_key_values[15].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.7.value")).Value = past_key_values[16].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.8.key")).Value = past_key_values[17].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.8.value")).Value = past_key_values[18].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.9.key")).Value = past_key_values[19].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.9.value")).Value = past_key_values[20].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.10.key")).Value = past_key_values[21].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.10.value")).Value = past_key_values[22].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.11.key")).Value = past_key_values[23].Value;
                input_data.Find(input => input.Name.Equals("past_key_values.11.value")).Value = past_key_values[24].Value;
            }

            var results = session.Run(input_data);

            // Please fill in here
            var outPastKeyValues = results.Skip(1).ToList();
            List<List<DisposableNamedOnnxValue>> newPastKeyValues = [];

            // Similar to the Python code, depending on the value of use_cache_branch, pack the out_past_key_values into groups of num_pkv units
            // ((DenseTensor<bool>)input_data[2].Value)[0]
            var useCache = ((DenseTensor<bool>)input_data.Find(input => input.Name.Equals("use_cache_branch")).Value)[0];
            if (useCache == false)
            {
                // Pack them to 4 units
                newPastKeyValues = outPastKeyValues.Select((_, index) => outPastKeyValues.Skip(index * 4).Take(4).ToList()).ToList();
                // Make a list using the 0~5th units of newPastKeyValues
                newPastKeyValues = newPastKeyValues.Take(6).ToList();
            }
            else if (useCache == true)
            {
                // Combine the first two units of out_past_key_values with the last two units of past_key_values
                newPastKeyValues = outPastKeyValues.Select((_, index) => outPastKeyValues.Skip(index * 2).Take(2).Concat(past_key_values.Skip(index * 2 + 2).Take(2)).ToList()).ToList();
            }

            return (results, newPastKeyValues);
        }

        private float[] Softmax(Tensor<float> logits)
        {
            // Find the maximum value in the logits
            float maxLogit = logits.Max();

            // Subtract the max value from each logit to avoid overflow during the exponential operation
            float[] expLogits = logits.Select(logit => (float)Math.Exp(logit - maxLogit)).ToArray();

            // Calculate the sum of the exponentials
            float sumExpLogits = expLogits.Sum();

            // Divide each exponential by the sum to get the softmax probabilities
            float[] softmaxProbabilities = expLogits.Select(expLogit => expLogit / sumExpLogits).ToArray();

            return softmaxProbabilities;
        }

        private int ArgMax(Tensor<float> probabilities, bool getSecond = false)
        {
            int argMaxIndex = 0;
            if (getSecond)
            {
                // Find the index of the second maximum value in the probabilities
                argMaxIndex = Array.IndexOf(probabilities.ToArray(), probabilities.OrderByDescending(x => x).Skip(1).First());
                return argMaxIndex;
            }
            // Find the index of the maximum value in the probabilities
            argMaxIndex = Array.IndexOf(probabilities.ToArray(), probabilities.Max());

            return argMaxIndex;
        }

    }
}
