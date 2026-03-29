#!/bin/bash
#
# HR Avatar Analysis Lambda - Cleanup Script
#
# Removes all AWS resources created by deploy.sh:
# - HTTP API Gateway
# - Lambda function
# - IAM role and policies
#
# Usage: ./cleanup.sh
#

set -e

FUNCTION_NAME="hr-avatar-analysis"
API_NAME="hr-avatar-analysis-api"
ROLE_NAME="hr-avatar-analysis-lambda-role"
REGION="${AWS_DEFAULT_REGION:-us-west-2}"

echo "========================================"
echo "HR Avatar Analysis - Cleanup"
echo "========================================"
echo ""
echo "This will delete:"
echo "  - Lambda function: $FUNCTION_NAME"
echo "  - HTTP API Gateway: $API_NAME"
echo "  - IAM role: $ROLE_NAME"
echo ""

read -p "Continue? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""

# =============================================================================
# STEP 1: Delete HTTP API Gateway
# =============================================================================

echo "[1/3] Deleting HTTP API Gateway..."

API_ID=$(aws apigatewayv2 get-apis --region "$REGION" \
    --query "Items[?Name=='$API_NAME'].ApiId" --output text 2>/dev/null || echo "")

if [ -n "$API_ID" ] && [ "$API_ID" != "None" ]; then
    aws apigatewayv2 delete-api --api-id "$API_ID" --region "$REGION" 2>/dev/null || true
    echo "  ✓ API Gateway deleted"
else
    echo "  (API Gateway not found)"
fi

# =============================================================================
# STEP 2: Delete Lambda Function
# =============================================================================

echo ""
echo "[2/3] Deleting Lambda function..."

if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" >/dev/null 2>&1; then
    aws lambda delete-function --function-name "$FUNCTION_NAME" --region "$REGION"
    echo "  ✓ Lambda function deleted"
else
    echo "  (Function not found)"
fi

# =============================================================================
# STEP 3: Delete IAM Role
# =============================================================================

echo ""
echo "[3/3] Deleting IAM role..."

# Delete inline policies
aws iam delete-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name bedrock-invoke 2>/dev/null || true

# Detach managed policies
aws iam detach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true

# Delete role
if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
    aws iam delete-role --role-name "$ROLE_NAME" 2>/dev/null || echo "  (Could not delete role - may be in use)"
    echo "  ✓ IAM role deleted"
else
    echo "  (Role not found)"
fi

# =============================================================================
# Cleanup Local Files
# =============================================================================

rm -f .api-url function.zip 2>/dev/null || true

echo ""
echo "========================================"
echo "✅ CLEANUP COMPLETE"
echo "========================================"
