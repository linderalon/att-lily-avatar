#!/bin/bash
#
# HR Avatar Analysis Lambda - Deployment Script
#
# Deploys the Lambda function with HTTP API Gateway for the HR Avatar
# call analysis feature. Uses AWS Bedrock Claude Haiku for analysis.
#
# Prerequisites:
#   - AWS CLI configured (aws sts get-caller-identity should work)
#   - Bedrock access enabled in your AWS account
#   - Claude models enabled in Bedrock console
#
# Usage: ./deploy.sh
#

set -e  # Exit on error

# =============================================================================
# CONFIGURATION
# =============================================================================

FUNCTION_NAME="hr-avatar-analysis"
API_NAME="hr-avatar-analysis-api"
ROLE_NAME="hr-avatar-analysis-lambda-role"
REGION="${AWS_DEFAULT_REGION:-us-west-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")

if [ -z "$ACCOUNT_ID" ]; then
    echo "ERROR: Could not get AWS account ID. Is AWS CLI configured?"
    exit 1
fi

echo "========================================"
echo "HR Avatar Analysis - Lambda Deployment"
echo "========================================"
echo ""
echo "Region:   $REGION"
echo "Account:  $ACCOUNT_ID"
echo "Function: $FUNCTION_NAME"
echo "API:      $API_NAME"
echo ""

# =============================================================================
# STEP 1: Create or Verify IAM Role
# =============================================================================

echo "[1/5] Setting up IAM role..."

if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
    echo "  ✓ Role already exists"
else
    echo "  Creating new role..."
    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document file://trust-policy.json \
        --description "Role for HR Avatar Analysis Lambda" \
        >/dev/null 2>&1 || {
            echo "  ⚠ Could not create role (may need existing role with Bedrock permissions)"
            echo "  Looking for alternative roles..."
            # Try to find an existing role with Bedrock access
            ROLE_NAME=$(aws iam list-roles --query "Roles[?contains(RoleName, 'Lambda')].RoleName" --output text | head -1)
            if [ -z "$ROLE_NAME" ]; then
                echo "  ERROR: No suitable Lambda role found"
                exit 1
            fi
            echo "  Using existing role: $ROLE_NAME"
        }
    sleep 10
fi

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# Try to attach policies (may fail if no permission, that's OK if role already has them)
aws iam put-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name bedrock-invoke \
    --policy-document file://bedrock-policy.json 2>/dev/null || true

aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true

echo "  ✓ Role ready: $ROLE_ARN"

# =============================================================================
# STEP 2: Package Lambda Function
# =============================================================================

echo ""
echo "[2/5] Packaging Lambda function..."

rm -f function.zip
zip -j function.zip lambda_function.py >/dev/null
echo "  ✓ Created function.zip"

# =============================================================================
# STEP 3: Create or Update Lambda Function
# =============================================================================

echo ""
echo "[3/5] Deploying Lambda function..."

if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "  Updating existing function..."
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file fileb://function.zip \
        --region "$REGION" \
        >/dev/null

    # Wait for update to complete
    aws lambda wait function-updated --function-name "$FUNCTION_NAME" --region "$REGION" 2>/dev/null || sleep 5
    echo "  ✓ Function updated"
else
    echo "  Creating new function..."
    sleep 5  # Wait for role propagation

    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --runtime python3.11 \
        --handler lambda_function.lambda_handler \
        --role "$ROLE_ARN" \
        --zip-file fileb://function.zip \
        --timeout 90 \
        --memory-size 256 \
        --region "$REGION" \
        --description "Analyzes HR Avatar call transcripts using Bedrock Claude" \
        >/dev/null

    aws lambda wait function-active --function-name "$FUNCTION_NAME" --region "$REGION"
    echo "  ✓ Function created"
fi

LAMBDA_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"

# =============================================================================
# STEP 4: Create or Get HTTP API Gateway
# =============================================================================

echo ""
echo "[4/5] Setting up HTTP API Gateway..."

# Check for existing API
API_ID=$(aws apigatewayv2 get-apis --region "$REGION" \
    --query "Items[?Name=='$API_NAME'].ApiId" --output text 2>/dev/null || echo "")

if [ -n "$API_ID" ] && [ "$API_ID" != "None" ]; then
    echo "  ✓ API already exists: $API_ID"
else
    echo "  Creating HTTP API..."
    API_RESULT=$(aws apigatewayv2 create-api \
        --name "$API_NAME" \
        --protocol-type HTTP \
        --cors-configuration '{"AllowOrigins":["*"],"AllowMethods":["POST","OPTIONS"],"AllowHeaders":["content-type"],"MaxAge":86400}' \
        --target "$LAMBDA_ARN" \
        --region "$REGION" \
        2>/dev/null)

    API_ID=$(echo "$API_RESULT" | grep -o '"ApiId":"[^"]*"' | cut -d'"' -f4)

    if [ -z "$API_ID" ]; then
        echo "  ERROR: Failed to create API Gateway"
        exit 1
    fi

    echo "  ✓ API created: $API_ID"
fi

# Get the API endpoint
API_ENDPOINT=$(aws apigatewayv2 get-api --api-id "$API_ID" --region "$REGION" \
    --query 'ApiEndpoint' --output text 2>/dev/null)

# =============================================================================
# STEP 5: Add Lambda Permission for API Gateway
# =============================================================================

echo ""
echo "[5/5] Configuring permissions..."

# Add permission for API Gateway to invoke Lambda (ignore if already exists)
aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "apigateway-${API_ID}" \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*" \
    --region "$REGION" \
    2>/dev/null || echo "  ✓ Permission already configured"

echo "  ✓ Permissions configured"

# =============================================================================
# DONE
# =============================================================================

echo ""
echo "========================================"
echo "✅ DEPLOYMENT COMPLETE"
echo "========================================"
echo ""
echo "API Endpoint:"
echo "  $API_ENDPOINT"
echo ""
echo "Test with:"
echo "  curl -X POST '$API_ENDPOINT' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"transcript\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"dpp\":{\"mode\":\"interview\",\"org\":{\"n\":\"Test\"},\"role\":{\"t\":\"Engineer\"},\"subj\":{\"name\":\"John\"}}}'"
echo ""
echo "Update the ANALYSIS_API_URL in both JS files:"
echo "  hr_avatar/hr-demo.js:     ANALYSIS_API_URL: '$API_ENDPOINT'"
echo "  code_interview/code-interview.js: ANALYSIS_API_URL: '$API_ENDPOINT'"
echo ""

# Save URL to file
echo "$API_ENDPOINT" > .api-url
echo "(Endpoint saved to .api-url)"
