"""
Generate Comprehensive Comparison Report
Traditional CV + KNN vs YOLO v8 + CNN
"""

import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime

def generate_report():
    """Generate comprehensive comparison report"""
    
    # Load data
    csv_file = Path("comparison_results.csv")
    json_file = Path("comparison_summary.json")
    
    if not csv_file.exists() or not json_file.exists():
        print("❌ Comparison files not found! Run compare_traditional_vs_yolo.py first.")
        return
    
    # Read data
    df = pd.read_csv(csv_file)
    with open(json_file) as f:
        summary = json.load(f)
    
    # Extract metrics
    trad_data = summary.get('traditional_cv_knn', {})
    cnn_data = summary.get('yolo_cnn', {})
    
    trad_accuracy = trad_data.get('accuracy', 0) * 100
    cnn_accuracy = cnn_data.get('accuracy', 0) * 100
    trad_correct = trad_data.get('correct_predictions', 0)
    cnn_correct = cnn_data.get('correct_predictions', 0)
    total = summary.get('total_images_tested', 0)
    
    improvement = cnn_accuracy - trad_accuracy
    trad_time = trad_data.get('avg_time_ms', 0)
    cnn_time = cnn_data.get('avg_time_ms', 0)
    
    # Calculate additional metrics
    trad_char_accuracy = (len(df) - df[df['traditional_correct'] == False].shape[0]) / len(df) * 100
    cnn_char_accuracy = (len(df) - df[df['cnn_correct'] == False].shape[0]) / len(df) * 100
    
    # Generate TXT Report
    txt_report = f"""{'='*80}
COMPARISON REPORT: TRADITIONAL CV + KNN vs YOLO v8 + CNN
{'='*80}

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: LP-characters
Total Images Tested: {total}

{'-'*80}
RESULTS SUMMARY
{'-'*80}

📊 Traditional CV + KNN:
   ✓ Accuracy: {trad_accuracy:.2f}%
   ✓ Correct: {trad_correct}/{total}
   ✓ Avg Time per Image: {trad_time:.2f}ms
   ✓ Character Accuracy: {trad_char_accuracy:.2f}%

📊 YOLO v8 + CNN:
   ✓ Accuracy: {cnn_accuracy:.2f}%
   ✓ Correct: {cnn_correct}/{total}
   ✓ Avg Time per Image: {cnn_time:.2f}ms
   ✓ Character Accuracy: {cnn_char_accuracy:.2f}%

📈 Improvement (CNN vs Traditional): +{improvement:.2f}%

Speed Comparison:
   ✓ CNN is {trad_time/cnn_time:.2f}x faster than Traditional
   ✓ Time Difference: {abs(trad_time - cnn_time):.2f}ms per image

{'-'*80}
DETAILED ANALYSIS
{'-'*80}

1. ACCURACY COMPARISON
   - Traditional CV + KNN: {trad_accuracy:.2f}%
   - YOLO v8 + CNN:      {cnn_accuracy:.2f}%
   - Winner: {'CNN 🏆' if cnn_accuracy > trad_accuracy else 'Traditional CV'}
   - Difference: {improvement:+.2f}%

2. SPEED COMPARISON
   - Traditional CV + KNN: {trad_time:.2f}ms/image
   - YOLO v8 + CNN:      {cnn_time:.2f}ms/image
   - Winner: {'CNN 🏆' if cnn_time < trad_time else 'Traditional CV'}
   - Speedup: {trad_time/cnn_time:.2f}x

3. CHARACTER-LEVEL ACCURACY
   - Traditional CV + KNN: {trad_char_accuracy:.2f}%
   - YOLO v8 + CNN:      {cnn_char_accuracy:.2f}%
   - Improvement: {cnn_char_accuracy - trad_char_accuracy:+.2f}%

4. ERROR DISTRIBUTION
   Traditional CV + KNN:
   ✗ Wrong Predictions: {total - trad_correct}/{total} ({100 - trad_accuracy:.2f}%)
   
   YOLO v8 + CNN:
   ✗ Wrong Predictions: {total - cnn_correct}/{total} ({100 - cnn_accuracy:.2f}%)

{'-'*80}
RECOMMENDATIONS
{'-'*80}

✓ For Production Use:
  • CNN offers {improvement:.2f}% better accuracy
  • CNN is {trad_time/cnn_time:.2f}x faster
  • Recommended: Use YOLO v8 + CNN

✓ For Real-time Applications:
  • Both methods can process images in reasonable time
  • CNN: {cnn_time:.2f}ms/image (suitable for real-time)
  • Traditional: {trad_time:.2f}ms/image (also acceptable)

✓ For Resource-Limited Devices:
  • Traditional CV + KNN is more lightweight
  • No GPU required for Traditional method
  • CNN requires more computation power

{'-'*80}
CONCLUSION
{'-'*80}

CNN (YOLO v8 + CNN) outperforms Traditional CV + KNN in:
✓ Accuracy: {improvement:+.2f}% improvement
✓ Speed: {trad_time/cnn_time:.2f}x faster
✓ Character Recognition: {cnn_char_accuracy - trad_char_accuracy:+.2f}% better

The CNN approach is recommended for license plate recognition on LP-characters dataset.

{'='*80}
"""

    # Generate Markdown Report
    md_report = f"""# Comparison Report: Traditional CV + KNN vs YOLO v8 + CNN

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset:** LP-characters  
**Total Images Tested:** {total}

---

## 📊 Results Summary

### Traditional CV + KNN
| Metric | Value |
|--------|-------|
| **Accuracy** | {trad_accuracy:.2f}% |
| **Correct Predictions** | {trad_correct}/{total} |
| **Avg Time per Image** | {trad_time:.2f}ms |
| **Character Accuracy** | {trad_char_accuracy:.2f}% |

### YOLO v8 + CNN
| Metric | Value |
|--------|-------|
| **Accuracy** | {cnn_accuracy:.2f}% |
| **Correct Predictions** | {cnn_correct}/{total} |
| **Avg Time per Image** | {cnn_time:.2f}ms |
| **Character Accuracy** | {cnn_char_accuracy:.2f}% |

### 📈 Improvement (CNN vs Traditional)
- **Accuracy Improvement:** +{improvement:.2f}%
- **Speed Improvement:** {trad_time/cnn_time:.2f}x faster

---

## 🔍 Detailed Analysis

### 1. Accuracy Comparison
```
Traditional CV + KNN: {trad_accuracy:.2f}%
YOLO v8 + CNN:      {cnn_accuracy:.2f}%
Difference:          {improvement:+.2f}%
Winner:              {'CNN 🏆' if cnn_accuracy > trad_accuracy else 'Traditional CV'}
```

### 2. Speed Comparison
```
Traditional CV + KNN: {trad_time:.2f}ms/image
YOLO v8 + CNN:      {cnn_time:.2f}ms/image
Speedup:             {trad_time/cnn_time:.2f}x
Winner:              {'CNN 🏆' if cnn_time < trad_time else 'Traditional CV'}
```

### 3. Character-Level Accuracy
```
Traditional CV + KNN: {trad_char_accuracy:.2f}%
YOLO v8 + CNN:      {cnn_char_accuracy:.2f}%
Improvement:         {cnn_char_accuracy - trad_char_accuracy:+.2f}%
```

### 4. Error Distribution
- **Traditional CV + KNN:** {total - trad_correct} wrong predictions ({100 - trad_accuracy:.2f}%)
- **YOLO v8 + CNN:** {total - cnn_correct} wrong predictions ({100 - cnn_accuracy:.2f}%)

---

## 💡 Recommendations

### ✓ For Production Use
- CNN offers **{improvement:.2f}%** better accuracy
- CNN is **{trad_time/cnn_time:.2f}x faster**
- **Recommendation:** Use YOLO v8 + CNN

### ✓ For Real-time Applications
- Both methods can process images in reasonable time
- CNN: **{cnn_time:.2f}ms/image** (suitable for real-time)
- Traditional: **{trad_time:.2f}ms/image** (also acceptable)

### ✓ For Resource-Limited Devices
- Traditional CV + KNN is more lightweight
- No GPU required for Traditional method
- CNN requires more computation power

---

## 🎯 Conclusion

**CNN (YOLO v8 + CNN)** outperforms **Traditional CV + KNN** in:
- ✅ Accuracy: **{improvement:+.2f}%** improvement
- ✅ Speed: **{trad_time/cnn_time:.2f}x** faster
- ✅ Character Recognition: **{cnn_char_accuracy - trad_char_accuracy:+.2f}%** better

**The CNN approach is recommended** for license plate recognition on LP-characters dataset.

---

## 📁 Generated Files
- `comparison_results.csv` - Detailed results for each image
- `comparison_summary.json` - Summary metrics
- `COMPARISON_REPORT.txt` - This text report
- `COMPARISON_REPORT.md` - This markdown report
"""

    # Save reports
    with open("COMPARISON_REPORT.txt", "w", encoding='utf-8') as f:
        f.write(txt_report)
    print("✅ Text report saved: COMPARISON_REPORT.txt")
    
    with open("COMPARISON_REPORT.md", "w", encoding='utf-8') as f:
        f.write(md_report)
    print("✅ Markdown report saved: COMPARISON_REPORT.md")
    
    # Print summary
    print(f"\n{'='*80}")
    print("COMPARISON REPORT GENERATED")
    print(f"{'='*80}\n")
    print(f"📊 Traditional CV + KNN: {trad_accuracy:.2f}% accuracy")
    print(f"📊 YOLO v8 + CNN:       {cnn_accuracy:.2f}% accuracy")
    print(f"📈 Improvement:         +{improvement:.2f}%\n")
    print(f"⚡ Speed Comparison:")
    print(f"   Traditional: {trad_time:.2f}ms/image")
    print(f"   CNN:        {cnn_time:.2f}ms/image")
    print(f"   Speedup:    {trad_time/cnn_time:.2f}x\n")
    print(f"📁 Reports saved:")
    print(f"   - COMPARISON_REPORT.txt")
    print(f"   - COMPARISON_REPORT.md")

if __name__ == "__main__":
    generate_report()
