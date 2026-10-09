#!/usr/bin/env bash
# CHƯA KIỂM THỬ. Triển khai chaincode ballot lên kênh electionchannel của mạng Fabric 3 tổ chức.
# Yêu cầu: Docker, fabric-samples (có bin/ và config/), mạng đã lên với 3 tổ chức và kênh đã tạo.
# Tên đường dẫn theo cấu trúc organizations/ của fabric-samples/test-network; hãy đối chiếu với phiên bản bạn dùng.
set -euo pipefail

FABRIC_SAMPLES=${FABRIC_SAMPLES:-$HOME/fabric-samples}
CHANNEL=electionchannel
CC=ballot
CC_DIR="$(cd "$(dirname "$0")/../chaincode/ballot" && pwd)"
COLL="$CC_DIR/collections.json"
POLICY="OutOf(2,'Org1MSP.peer','Org2MSP.peer','Org3MSP.peer')"
ORGS="Org1:org1 Org2:org2 Org3:org3"   # MSP (Org1MSP...) -> thư mục org trong test-network

export PATH="$FABRIC_SAMPLES/bin:$PATH"
export FABRIC_CFG_PATH="$FABRIC_SAMPLES/config"

as_org() {  # as_org <tên-org> ; đặt biến môi trường để gọi peer với danh tính tổ chức đó
  local name=$1 dir=$2
  local base="$FABRIC_SAMPLES/test-network/organizations/peerOrganizations/$dir.example.com"
  export CORE_PEER_TLS_ENABLED=true
  export CORE_PEER_LOCALMSPID="${name}MSP"
  export CORE_PEER_TLS_ROOTCERT_FILE="$base/peers/peer0.$dir.example.com/tls/ca.crt"
  export CORE_PEER_MSPCONFIGPATH="$base/users/Admin@$dir.example.com/msp"
  export CORE_PEER_ADDRESS="localhost:$(case $dir in org1) echo 7051;; org2) echo 9051;; org3) echo 11051;; esac)"
}

PKG="$CC.tar.gz"
peer lifecycle chaincode package "$PKG" --path "$CC_DIR" --lang golang --label "${CC}_1.0"
PKG_ID=""

for pair in $ORGS; do
  name=${pair%%:*}; dir=${pair##*:}
  as_org "$name" "$dir"
  peer lifecycle chaincode install "$PKG"
  PKG_ID=$(peer lifecycle chaincode calculatepackageid "$PKG")
done

for pair in $ORGS; do
  name=${pair%%:*}; dir=${pair##*:}
  as_org "$name" "$dir"
  peer lifecycle chaincode approveformyorg -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
    --channelID "$CHANNEL" --name "$CC" --version 1.0 --package-id "$PKG_ID" --sequence 1 \
    --signature-policy "$POLICY" --collections-config "$COLL" --tls --cafile "$FABRIC_SAMPLES/test-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem"
done

as_org Org2 org2
peer lifecycle chaincode commit -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
  --channelID "$CHANNEL" --name "$CC" --version 1.0 --sequence 1 \
  --signature-policy "$POLICY" --collections-config "$COLL" --tls \
  --cafile "$FABRIC_SAMPLES/test-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
  --peerAddresses localhost:7051 --tlsRootCertFiles "$FABRIC_SAMPLES/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" \
  --peerAddresses localhost:9051 --tlsRootCertFiles "$FABRIC_SAMPLES/test-network/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"

echo "Đã commit chaincode $CC. Ví dụ ghi một phiếu (cần đủ 2/3 tổ chức endorse):"
echo "  peer chaincode invoke ... -c '{\"Args\":[\"RecordBallot\",\"ballot_0001.png\",\"valid\",\"0.98\",\"<sha256>\"]}' ..."
