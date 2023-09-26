using System.Numerics;
using System.Collections.Generic;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices;

using BenchmarkDotNet.Running;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Jobs;

namespace test_avx;

[Config(typeof(BenchmarkConfiguration))]
public class Program
{
  // dimensionality for text-embedding-ada-002 and text-search-davinci-*-001
  [Params(1536,12288)]
  public int Dimensionality;

#if false
  // size of the collection for RAG-like grounding
  [Params(10,50,100)]
  public int DocsCollectionSize;
#endif

  private float[] input; // cannot have this be a `Span` (this class is not a ref struct)
  private float[] input2;
  private List< float[] > docsCollection = new();

  [GlobalSetup]
  public void Setup()
  {
    input = GenerateRandom(Dimensionality);
#if false
    for (int i = 0; i < DocsCollectionSize; i++) {
      docsCollection.Add(GenerateRandom(Dimensionality));
    }
#else
    input2 = GenerateRandom(Dimensionality);
#endif
  }

  [Benchmark]
  public void SimilarityScalar()
  {
#if false
    List<float> similarities = new();
    foreach (var doc in docsCollection) {
      //similarities.Add( CosineSimilarity(input.AsSpan(), doc.AsSpan()));
      similarities.Add( CosineSimilarity(input, doc) );
    }
#else
    CosineSimilarity(input, input2);
#endif
  }

  [Benchmark]
  public void SimilarityScalarVec()
  {
#if false
    List<float> similarities = new();
    foreach (var doc in docsCollection) {
      similarities.Add( CosineSimilarityVec(input, doc) );
    }
#else
    CosineSimilarityVec(input, input2);
#endif
  }

#if false
  [Benchmark]
  public void SimilarityScalarVec512()
  {
    List<float> similarities = new();
    foreach (var doc in docsCollection) {
      similarities.Add( CosineSimilarityVec512(input.AsSpan(), doc.AsSpan()));
    }
  }
#endif

  static void Main(string[] args)
  {
    //if (!Vector512.IsHardwareAccelerated) {
    if (!Vector.IsHardwareAccelerated) {
      Console.WriteLine("This machine doesn't support wide registers (AVX), exiting!");
      Environment.Exit(1);
    }

#if false
    Test();
#else
    BenchmarkRunner.Run<Program>();
#endif
    Console.WriteLine($"The parallelization count for this implementation of Vector<t> is {Vector<float>.Count}");
  }

  private static void Test()
  {
    Console.WriteLine($"The vector width for generic `Vector` of float is {Vector<float>.Count}.");
    Span<float> vec1536_a = GenerateRandom(1536);
    Span<float> vec1536_b = GenerateRandom(1536);
    Console.WriteLine($"The size of this span is: {vec1536_a.Length}");
    var distance = CosineSimilarity(vec1536_a, vec1536_b);
    Console.WriteLine($"The distance between this two is {distance}.");
    var distanceV = CosineSimilarityVec(vec1536_a, vec1536_b);
    Console.WriteLine($"And from the vectorized version: {distanceV}.");
    var distanceV512 = CosineSimilarityVec512(vec1536_a, vec1536_b);
    Console.WriteLine($"And from the vectorized version: {distanceV512}.");
    Console.WriteLine("Done!");
  }

  private static float[] GenerateRandom(int size)
  {
  #if false
    var random = new Random();
    var span = new Span<float>(new float[size]);
    for (int i = 0; i < size; i++) {
      span[i] = (float)random.NextDouble();
    }
    return span;
  #else
    return Enumerable.Range(0,size).Select(_ => Random.Shared.NextSingle()).ToArray();
  #endif
  }

  private static float CosineSimilarity(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
  {
    if (x.Length != y.Length) {
      throw new ArgumentException("Array lengths must be equal");
    }
    float dot = 0, xSumSquared = 0, ySumSquared = 0;

    for (int i = 0; i < x.Length; i++) {
      dot += x[i] * y[i];
      xSumSquared += x[i] * x[i];
      ySumSquared += y[i] * y[i];
    }

    return dot / (MathF.Sqrt(xSumSquared) * MathF.Sqrt(ySumSquared));
  }

  // from https://github.com/microsoft/semantic-kernel/blob/3451a4ebbc9db0d049f48804c12791c681a326cb/dotnet/src/SemanticKernel.Core/AI/Embeddings/VectorOperations/CosineSimilarityOperation.cs#L119
  private static unsafe float CosineSimilarityVec(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
  {
    if (x.Length != y.Length) {
      throw new ArgumentException("Array lengths must be equal");
    }
    fixed (float* pxBuffer = x, pyBuffer = y) {
      double dotSum = 0, lenXSum = 0, lenYSum = 0;

      float* px = pxBuffer, py = pyBuffer;
      float* pxEnd = px + x.Length;

      if (Vector.IsHardwareAccelerated && x.Length >= Vector<float>.Count) {
        float* pxOneVectorFromEnd = pxEnd - Vector<float>.Count;
        do {
          Vector<float> xVec = *(Vector<float>*)px;
          Vector<float> yVec = *(Vector<float>*)py;

          dotSum += Vector.Dot(xVec, yVec); // Dot product
          lenXSum += Vector.Dot(xVec, xVec); // For magnitude of x
          lenYSum += Vector.Dot(yVec, yVec); // For magnitude of y

          px += Vector<float>.Count;
          py += Vector<float>.Count;
        } while (px <= pxOneVectorFromEnd);
      }

      while (px < pxEnd) {
        float xVal = *px;
        float yVal = *py;

        dotSum += xVal * yVal; // Dot product
        lenXSum += xVal * xVal; // For magnitude of x
        lenYSum += yVal * yVal; // For magnitude of y

        ++px;
        ++py;
      }

      // Cosine Similarity of X, Y
      // Sum(X * Y) / |X| * |Y|
      return (float)(dotSum / (Math.Sqrt(lenXSum) * Math.Sqrt(lenYSum)));
    }
  }

  private static unsafe float CosineSimilarityVec512(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
  {
    if (x.Length != y.Length) {
      throw new ArgumentException("Array lengths must be equal");
    }
    fixed (float* pxBuffer = x, pyBuffer = y) {
      double dotSum = 0, lenXSum = 0, lenYSum = 0;

      float* px = pxBuffer, py = pyBuffer;
      float* pxEnd = px + x.Length;

      if (Vector512.IsHardwareAccelerated && x.Length >= Vector512<float>.Count) {
        float* pxOneVectorFromEnd = pxEnd - Vector512<float>.Count;
        do {
          Vector512<float> xVec = *(Vector512<float>*)px;
          Vector512<float> yVec = *(Vector512<float>*)py;

          dotSum += Vector512.Dot(xVec, yVec); // Dot product
          lenXSum += Vector512.Dot(xVec, xVec); // For magnitude of x
          lenYSum += Vector512.Dot(yVec, yVec); // For magnitude of y

          px += Vector512<float>.Count;
          py += Vector512<float>.Count;
        } while (px <= pxOneVectorFromEnd);
      }

      while (px < pxEnd) {
        float xVal = *px;
        float yVal = *py;

        dotSum += xVal * yVal; // Dot product
        lenXSum += xVal * xVal; // For magnitude of x
        lenYSum += yVal * yVal; // For magnitude of y

        ++px;
        ++py;
      }

      // Cosine Similarity of X, Y
      // Sum(X * Y) / |X| * |Y|
      return (float)(dotSum / (Math.Sqrt(lenXSum) * Math.Sqrt(lenYSum)));
    }
  }

  private class BenchmarkConfiguration : ManualConfig
  {
    public BenchmarkConfiguration()
    {
      AddJob(Job.Default.WithRuntime(CoreRuntime.Core80)
         .WithId("AVX512F Enabled"));
#if false	 
      AddJob(Job.Default.WithRuntime(CoreRuntime.Core80)
         .WithEnvironmentVariables(new EnvironmentVariable("DOTNET_EnableAVX512F", "0"))
         .WithId("AVX512F Disabled"));
#endif
    }
  }
}
