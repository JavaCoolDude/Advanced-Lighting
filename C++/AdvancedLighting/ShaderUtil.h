//--------------------------------------------------------------------------------------
// Globals
//--------------------------------------------------------------------------------------

#define MAX_LINKED_LIGHTS_PER_PIXEL 64
#define MAX_LLL_ELEMENTS            0xFFFFFF
#define MAX_LLL_LIGHTS              256

#define MAX_CASCADES                8

//--------------------------------------------------------------------------------------
// Constant Buffers 
//--------------------------------------------------------------------------------------
#define CB_FRAME             0
#define CB_SIMPLE            1
#define CB_SHADOW_DATA       1

//--------------------------------------------------------------------------------------
// Textures 
//--------------------------------------------------------------------------------------
#define TEX_DIFFUSE          0 

#define TEX_DEPTH            0
#define TEX_NRM              1
#define TEX_COL              2

#define TEX_SHADOW           5

#define SRV_LIGHT_LINKED     10
#define SRV_LIGHT_OFFSET     11
#define SRV_LIGHT_ENV        12

//--------------------------------------------------------------------------------------
// Samplers 
//--------------------------------------------------------------------------------------
#define SAM_LINEAR           0 
#define SAM_POINT            1
#define SAM_SHADOW           5

//--------------------------------------------------------------------------------------
// UAVS 
//--------------------------------------------------------------------------------------
#define UAV_LIGHT_LINKED     3
#define UAV_LIGHT_OFFSET     4
#define UAV_LIGHT_BOUNDS     5
 
// The number of cascades 
#define CASCADE_COUNT_FLAG   3
#define CASCADE_BUFFER_SIZE  1536

// C/C++ side of things
#if !defined(__HLSL_SHADER__)
  typedef DirectX::XMFLOAT4X4  float4x4;
  typedef DirectX::XMFLOAT4    float4;
  typedef DirectX::XMFLOAT3    float3;
  typedef DirectX::XMINT4      int4;
  typedef uint32_t             uint;

  #define cbuffer              struct 

  #define B_REGISTER( reg_ )
  #define T_REGISTER( reg_ )
  #define S_REGISTER( reg_ )
  #define U_REGISTER( reg_ )

// HLSL
#else
  #pragma pack_matrix( row_major )

  #define B_REGISTER( reg_ ) : register(b##reg_)
  #define T_REGISTER( reg_ )   register(t##reg_)
  #define S_REGISTER( reg_ )   register(s##reg_)
  #define U_REGISTER( reg_ )   register(u##reg_)

#endif


//--------------------------------------------------------------------------
struct GPULightEnv
{
  float3    m_WorldPos;
  float     m_Radius;

  float3    m_LinearColor;
  float     m_SpecIntensity; 
};

//-------------------------------------------------------------------------
cbuffer FrameCB     B_REGISTER( CB_FRAME )
{
    float4x4        m_mWorldViewProj;
    float4x4        m_mWorldView; 
    float4x4        m_mWorld;
    float4          m_vLinearDepthConsts;
    float4          m_vViewportToClip;
    float4          m_vCameraPos;
    uint            m_iLLLWidth;
    uint            m_iLLLHeight;
    uint            m_iLLLMaxAlloc;
    uint            m_iUnused;
};

//-------------------------------------------------------------------------
cbuffer ShadowDataCB B_REGISTER( CB_SHADOW_DATA )
{
    float4x4        m_mShadow;
    float4          m_vCascadeOffset[8];
    float4          m_vCascadeScale[8];
    int             m_nCascadeLevels; // Number of Cascades
    int             m_iVisualizeCascades; // 1 is to visualize the cascades in different colors. 0 is to just draw the scene
    int             m_iPCFBlurForLoopStart; // For loop begin value. For a 5x5 kernel this would be -2.
    int             m_iPCFBlurForLoopEnd; // For loop end value. For a 5x5 kernel this would be 3.

    // For Map based selection scheme, this keeps the pixels inside of the the valid range.
    // When there is no boarder, these values are 0 and 1 respectively.
    float           m_fMinBorderPadding;     
    float           m_fMaxBorderPadding;
    float           m_fShadowBiasFromGUI;  // A shadow map offset to deal with self shadow artifacts.  
                                           //These artifacts are aggravated by PCF.
    float           m_fShadowPartitionSize; 

    float           m_fCascadeBlendArea; // Amount to overlap when blending between cascades.
    float           m_fTexelSize; 
    float           m_fNativeTexelSizeInX;
    float           m_fPaddingForCB3; // Padding variables exist because CBs must be a multiple of 16 bytes. 

    float3          m_vLightDir;
    float           m_fPaddingCB4;
};

//-------------------------------------------------------------------------
cbuffer SimpleCB B_REGISTER( CB_SIMPLE )
{
  float4x4          m_mSimpleTransform;
  float             m_mSimpleLightIndex;
  float             m_mSimplePushScale;
  float             m_mSimpleRadius;
  float             m_mSimpleUnused;
};

#if defined(__HLSL_SHADER__)

//--------------------------------------------------------------------------------------
// Textures 
//--------------------------------------------------------------------------------------
Texture2D              g_txDiffuse          : T_REGISTER( TEX_DIFFUSE);

Texture2D              g_txDepth            : T_REGISTER( TEX_DEPTH  );
Texture2D              g_txNormal           : T_REGISTER( TEX_NRM    );
Texture2D              g_txColor            : T_REGISTER( TEX_COL    );

Texture2D              g_txShadow           : T_REGISTER( TEX_SHADOW );

//--------------------------------------------------------------------------------------
// Samplers 
//--------------------------------------------------------------------------------------
SamplerState           g_samLinear          : S_REGISTER( SAM_LINEAR );
SamplerState           g_samPoint           : S_REGISTER( SAM_POINT  );
SamplerComparisonState g_samShadow          : S_REGISTER( SAM_SHADOW );

struct LightFragmentLink
{
  uint m_DepthInfo; // High bits min depth, low bits max depth
  uint m_IndexNext; // Light index and link to the next fragment 
};
globallycoherent RWStructuredBuffer< LightFragmentLink >  g_LightFragmentLinkedBuffer   : U_REGISTER( UAV_LIGHT_LINKED );
globallycoherent RWByteAddressBuffer                      g_LightStartOffsetBuffer      : U_REGISTER( UAV_LIGHT_OFFSET );
globallycoherent RWByteAddressBuffer                      g_LightBoundsBuffer           : U_REGISTER( UAV_LIGHT_BOUNDS );

StructuredBuffer< LightFragmentLink >                     g_LightFragmentLinkedView     : T_REGISTER( SRV_LIGHT_LINKED );
Buffer<uint>                                              g_LightStartOffsetView        : T_REGISTER( SRV_LIGHT_OFFSET ); 

StructuredBuffer<GPULightEnv>                             g_LightEnvs                   : T_REGISTER( SRV_LIGHT_ENV    );

//--------------------------------------------------------------------------------------
// Input / Output structures
//--------------------------------------------------------------------------------------
struct VS_INPUT
{
    float4 vPosition  : POSITION;
    float3 vNormal    : NORMAL;
    float2 vTexcoord  : TEXCOORD0;
};

struct VS_OUTPUT
{
    float4 vPosition  : SV_POSITION; 
    float3 vNormal    : NORMAL;
    float2 vTexcoord  : TEXCOORD0; 
};

struct VS_OUTPUT2D
{
    float4 vPosition    : SV_POSITION;
    float3 vNormal      : NORMAL;
    float2 vTexcoord    : TEXCOORD0;
};

struct VS_OUTPUT_SHADOW
{
  float4 vPosition    : SV_POSITION;
};

struct VS_OUTPUT_SIMPLE
{
  float4 vPosition    : SV_POSITION;
};

//--------------------------------------------------------------------------------------------------
float4  TransformPosition(float3 position, float4x4 transform)
{
  return (position.xxxx * transform[0] + (position.yyyy * transform[1] + (position.zzzz * transform[2] + transform[3])));
}

//--------------------------------------------------------------------------------------------------
uint    ScreenUVsToLLLIndex(float2 screen_uvs)
{
  uint   x_unorm = saturate(screen_uvs.x) * m_iLLLWidth;
  uint   y_unorm = saturate(screen_uvs.y) * m_iLLLHeight;

  return y_unorm * m_iLLLWidth + x_unorm;
}

#endif