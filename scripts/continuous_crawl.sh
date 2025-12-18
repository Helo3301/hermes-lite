#!/bin/bash
# Continuous crawler - runs until target reached
TARGET=${1:-10000}
API="http://localhost:8780"

KEYWORDS=(
    "reinforcement%20learning,RL,policy%20gradient"
    "computer%20vision,image%20recognition,CNN"
    "natural%20language%20processing,NLP,linguistics"
    "speech%20recognition,audio,ASR"
    "generative%20model,GAN,diffusion"
    "graph%20neural%20network,GNN,node%20embedding"
    "time%20series,forecasting,temporal"
    "object%20detection,YOLO,SSD"
    "semantic%20segmentation,instance%20segmentation"
    "machine%20translation,NMT,seq2seq"
    "question%20answering,reading%20comprehension"
    "sentiment%20analysis,opinion%20mining"
    "named%20entity%20recognition,NER,sequence%20labeling"
    "text%20classification,document%20classification"
    "knowledge%20distillation,model%20compression"
    "self-supervised%20learning,contrastive"
    "meta-learning,few-shot,transfer%20learning"
    "federated%20learning,privacy%20preserving"
    "adversarial%20learning,robustness"
    "neural%20architecture%20search,AutoML"
    "attention%20mechanism,transformer,BERT"
    "language%20model,GPT,pretrained"
    "embedding%20learning,representation"
    "multi-task%20learning,joint%20learning"
    "zero-shot%20learning,cross-domain"
)

CATS="cs.LG,cs.CL,cs.CV,cs.AI,cs.NE,stat.ML"
BATCH=50
DAYS=365

echo "Target: $TARGET documents"
idx=0

while true; do
    CURRENT=$(sqlite3 /home/hestiasadmin/hermes-lite/data/hermes.db "SELECT COUNT(*) FROM documents;" 2>/dev/null)

    if [ "$CURRENT" -ge "$TARGET" ]; then
        echo "TARGET REACHED: $CURRENT documents"
        break
    fi

    KW="${KEYWORDS[$idx]}"
    idx=$(( (idx + 1) % ${#KEYWORDS[@]} ))

    echo "[$(date '+%H:%M:%S')] $CURRENT/$TARGET | Keywords: $KW"

    curl -s -X POST "$API/crawler/run?keywords=$KW&categories=$CATS&max_papers=$BATCH&days_back=$DAYS&collection=ai-papers" > /dev/null 2>&1

    sleep 2
done
