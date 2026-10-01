"""Run the fixed absolute-opportunity market sentiment comparison."""
import argparse
import json
import tail_formula_pipeline as pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['protocols', 'fit', 'freeze', 'analyze', 'finish'])
    parser.add_argument('--arm', choices=['control', 'memory'])
    parser.add_argument('--fold', choices=['2024h1', '2024h2', '2025h1', '2025h2'])
    args = parser.parse_args()
    if args.stage == 'fit':
        if not (args.arm and args.fold):
            parser.error('fit requires --arm and --fold')
        result = pipeline.fit(args.arm, args.fold)
    else:
        result = getattr(pipeline, args.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
