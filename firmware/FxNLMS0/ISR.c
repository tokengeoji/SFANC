/*
 * ISR.c
 *
 * Real-Time Active Noise Control (ANC) with Filtered-X Normalized LMS (FxNLMS)
 * Target: TI OMAP-L138 / TMS320C6748
 *
 * Inputs:
 *   - Line-in Left  : Reference Microphone x(n)
 *   - Line-in Right : Error Microphone e(n)
 * Output:
 *   - Line-out Right: Canceling Anti-Noise (-y(n)) [Left muted for TPA3116D2 protection]
 */

#include <xdc/std.h>
#include <xdc/runtime/System.h>
#include <ti/sysbios/BIOS.h>
#include <ti/sysbios/knl/Swi.h>
#include <xdc/cfg/global.h>
#include <math.h>
#include "LED_DIPSW.h"
#include "Timer.h"
#include "interrupt.h"
#include "McASP.h"
#include "define.h"
#include <c6x.h>

/* LED counter */
volatile static int32_t led_cnt = 0;

/* =========================================================================
 * FxNLMS ANC Buffers and Filter Definitions
 * ========================================================================= */

/* Primary adaptive filter W(z) - Length: FILTER_TAP (256) */
float W[FILTER_TAP] = {0.0f};

/* Reference microphone delay line: x(n), x(n-1), ... */
float X[FILTER_TAP] = {0.0f};

/*
 * Secondary path estimated model S_hat(z) coefficients (Length: SEC_FILTER_TAP = 500)
 * Measured from real hardware acoustic duct via NLMS0
 */
float S_hat[SEC_FILTER_TAP] = {
    -0.00060239f,  0.00030271f, -0.00068583f,  0.00035265f, -0.00007957f, -0.00028660f,  0.00017983f, -0.00022319f,
    -0.00012729f, -0.00021978f, -0.00023842f, -0.00016570f, -0.00057563f, -0.00027173f, -0.00032825f, -0.00049965f,
    -0.00053461f, -0.00046409f, -0.00028774f, -0.00060877f,  0.00001679f, -0.00068772f, -0.00059183f, -0.00043459f,
    -0.00032820f, -0.00017097f, -0.00033426f, -0.00024722f, -0.00002806f,  0.00016671f, -0.00019511f,  0.00018162f,
     0.00009498f,  0.00006522f,  0.00028964f,  0.00045890f,  0.00044335f,  0.00076261f, -0.00016234f,  0.00065550f,
    -0.00053015f,  0.00173666f, -0.00260954f,  0.00418844f, -0.00548884f,  0.00777747f, -0.00965053f,  0.02302209f,
     0.07271364f,  0.11690892f,  0.14044167f,  0.09920169f,  0.00777357f,  0.00008289f,  0.12817065f,  0.02283049f,
    -0.06798366f,  0.08920269f, -0.08260290f, -0.19279724f, -0.09689095f, -0.13128787f, -0.11839433f,  0.03405930f,
     0.02600677f, -0.19828938f,  0.13460734f, -0.14187710f,  0.00712803f, -0.07573172f, -0.08976456f, -0.01528165f,
    -0.13270845f,  0.02303303f,  0.01732389f,  0.06596483f,  0.01291696f,  0.01080471f, -0.02925954f,  0.04005418f,
     0.00701792f, -0.05226825f,  0.00076155f, -0.01180140f,  0.01348612f,  0.08282792f,  0.01375842f,  0.04273329f,
    -0.01317611f,  0.04079549f,  0.02030757f,  0.07344957f, -0.00822052f, -0.03353506f, -0.00441715f, -0.02122209f,
     0.03959805f,  0.03296566f,  0.05664966f,  0.04099367f, -0.00768328f,  0.03066016f, -0.00357681f,  0.02030951f,
    -0.01379859f,  0.00923895f, -0.00914364f,  0.02011165f,  0.03280094f,  0.00345212f,  0.04600089f, -0.02384583f,
     0.03059696f, -0.03624043f, -0.01651921f, -0.00749855f, -0.05035834f,  0.01827860f, -0.00713488f,  0.00026733f,
     0.01106505f, -0.00851622f,  0.00408493f, -0.01850331f,  0.00123680f, -0.02287313f, -0.01078510f, -0.02481300f,
    -0.01563931f,  0.00193849f,  0.00237669f,  0.01607019f,  0.00350066f,  0.00414295f, -0.01319427f,  0.00350044f,
    -0.01735576f, -0.01482806f,  0.00914378f, -0.01036587f,  0.01167862f,  0.00668653f,  0.00399623f,  0.02024917f,
    -0.00148209f,  0.03272773f,  0.03473512f,  0.04398391f,  0.06339934f,  0.06617285f,  0.05048681f,  0.00680112f,
    -0.01367974f, -0.04877334f, -0.02877461f, -0.02358247f, -0.01336340f, -0.01259743f, -0.01354328f, -0.01664327f,
    -0.01526234f,  0.00134319f, -0.00137323f, -0.01296893f, -0.05572075f, -0.04589682f, -0.07531876f, -0.07197084f,
    -0.06510184f, -0.03958896f, -0.02823546f, -0.03473081f, -0.00286811f, -0.01289182f, -0.00072469f,  0.00300645f,
     0.00330754f,  0.01144298f, -0.00874309f, -0.00350296f,  0.01350510f,  0.02894651f,  0.05391229f,  0.07108953f,
     0.04667666f,  0.04210439f,  0.03971266f,  0.00926037f,  0.02434109f,  0.01820709f,  0.01875105f,  0.01105986f,
    -0.00694069f,  0.01776977f,  0.01824004f,  0.01721367f,  0.02081647f,  0.01892655f,  0.02926178f, -0.00949989f,
    -0.00341195f, -0.01984870f, -0.02773552f,  0.00456049f,  0.01498889f,  0.00073984f,  0.03509517f,  0.01432661f,
    -0.02401267f,  0.00568118f,  0.00344614f,  0.00088225f, -0.00346270f, -0.00396068f, -0.00065897f,  0.01682646f,
    -0.01716595f, -0.00266736f, -0.01642870f, -0.01245213f, -0.00258032f, -0.02679662f,  0.00147490f, -0.00702728f,
    -0.01671715f, -0.03637103f, -0.02102907f,  0.00155791f,  0.00042541f,  0.00157377f,  0.01117517f,  0.01344880f,
     0.00863583f, -0.01130652f, -0.01566396f,  0.00664791f,  0.00902444f, -0.00013173f,  0.00183832f,  0.03009021f,
     0.00661191f,  0.00543835f, -0.02382748f, -0.00646218f, -0.01040922f, -0.01094830f,  0.00576582f, -0.01078753f,
     0.00939103f,  0.00588706f,  0.00140389f, -0.00369001f, -0.02334586f,  0.00616205f, -0.00850720f, -0.00334472f,
     0.00843959f, -0.02332822f,  0.01473241f, -0.01491189f, -0.00324794f, -0.01013183f, -0.00829157f,  0.01447771f,
     0.00564892f, -0.00232480f, -0.00428420f,  0.00448562f, -0.00612086f,  0.00254716f, -0.01215212f,  0.00054810f,
     0.00876016f, -0.00657960f, -0.00521531f, -0.00840770f, -0.00159380f, -0.00202545f,  0.00910233f,  0.01042587f,
     0.01920931f,  0.02008805f,  0.01335893f,  0.01933513f,  0.02490339f,  0.02224275f,  0.01924303f,  0.00840877f,
     0.00979826f,  0.00273574f,  0.00448863f,  0.01253167f,  0.01590204f,  0.01181159f,  0.00292956f, -0.00396771f,
    -0.01819859f,  0.00384601f, -0.01021606f, -0.00935684f, -0.00542724f, -0.01937309f, -0.00036360f, -0.01600948f,
    -0.00494909f,  0.00103003f, -0.00708691f,  0.00012114f, -0.01373329f, -0.01823646f, -0.01483174f, -0.00927660f,
    -0.00778919f, -0.00122159f, -0.00551730f, -0.00161750f, -0.00240428f,  0.00392353f, -0.00415386f,  0.00175153f,
     0.00106184f, -0.00976828f, -0.00211854f, -0.01786063f, -0.00120276f,  0.00010241f, -0.00891551f, -0.00600357f,
    -0.00925920f,  0.00149771f, -0.00235012f, -0.00363917f, -0.00993195f, -0.00722844f,  0.00672356f,  0.00334604f,
     0.00620635f, -0.00988916f,  0.00036962f,  0.00245073f, -0.00101729f,  0.00855689f, -0.00039428f,  0.00327004f,
    -0.00467626f,  0.00600742f, -0.00452430f,  0.00295484f,  0.01262623f,  0.00366062f,  0.00840154f, -0.00433464f,
     0.00660005f,  0.00235198f,  0.00183392f, -0.00475883f,  0.00702004f, -0.00566907f,  0.00684857f,  0.00707351f,
    -0.00309394f,  0.00111723f, -0.01037060f,  0.00303763f,  0.00205635f,  0.01039862f,  0.00787006f, -0.00279585f,
     0.00201646f, -0.00105945f,  0.00358552f,  0.00565624f, -0.00190588f,  0.00884327f,  0.00338609f,  0.00285340f,
     0.00250997f, -0.00018730f, -0.00625439f, -0.00042172f, -0.00323170f,  0.00767797f,  0.00257763f,  0.00169136f,
     0.00999711f, -0.00697110f,  0.00440776f, -0.00270620f, -0.00238481f,  0.00650997f, -0.00730668f,  0.00920103f,
    -0.00126583f, -0.00105304f,  0.00216514f, -0.00877509f,  0.00480636f, -0.01157910f, -0.00274022f, -0.00376860f,
    -0.00175681f, -0.00417669f, -0.00852259f, -0.01103030f, -0.01356869f,  0.00101006f, -0.01473165f, -0.00602718f,
    -0.00540150f, -0.00341509f,  0.00141385f, -0.00986873f,  0.00436029f, -0.00901807f,  0.00427353f,  0.00090387f,
    -0.00267229f,  0.00901240f, -0.00768443f,  0.00676698f, -0.00289174f,  0.00567328f,  0.00768888f, -0.00101856f,
     0.00196474f,  0.00192379f,  0.00471621f,  0.00398507f,  0.00746759f,  0.00485012f,  0.00235294f, -0.00179803f,
    -0.00408743f, -0.00220740f, -0.00316617f,  0.00255265f, -0.00064964f,  0.00516067f, -0.00240971f,  0.00010497f,
     0.00381397f, -0.00158526f,  0.01250168f, -0.00072969f,  0.01232157f,  0.00422786f,  0.00264470f,  0.00823553f,
     0.00001022f,  0.01055251f,  0.00194026f,  0.00567691f,  0.00337775f,  0.00086418f,  0.01070966f, -0.00161379f,
     0.00050936f,  0.00365852f, -0.00027835f,  0.00339593f, -0.00265626f,  0.00035997f, -0.00224983f, -0.00211107f,
    -0.00259537f, -0.00406008f, -0.00313562f,  0.00192021f,  0.00108178f, -0.00454395f, -0.00015170f, -0.00271501f,
     0.00332324f, -0.00237927f, -0.00248338f, -0.00271963f, -0.00760342f,  0.00603721f,  0.00065653f, -0.00172602f,
     0.00310226f, -0.00657089f,  0.00287797f, -0.00268067f,  0.00079693f,  0.00264430f, -0.00573299f,  0.00060802f,
    -0.00387024f,  0.00012292f,  0.00019024f, -0.00298477f, -0.00219505f, -0.00188531f,  0.00170898f,  0.00193156f,
    -0.00442115f, -0.00351419f, -0.00214427f, -0.00263306f
};


/*
 * Acoustic Feedback Path estimated model F_hat(z) coefficients (Length: FB_FILTER_TAP = 500)
 * Measured from real hardware acoustic duct via NLMS0 (Canceling Speaker -> Reference Mic)
 */
float F_hat[SEC_FILTER_TAP] = {
     0.00011775f,  0.00003599f,  0.00008085f,  0.00013236f,  0.00005192f,  0.00005600f,  0.00012161f,  0.00015830f,
     0.00010869f,  0.00013724f, -0.00009072f,  0.00021858f, -0.00000539f,  0.00010419f,  0.00023266f, -0.00013020f,
     0.00031822f, -0.00012982f,  0.00017441f,  0.00008506f, -0.00009759f,  0.00022978f, -0.00018461f,  0.00005361f,
    -0.00001843f,  0.00001323f,  0.00012103f, -0.00000486f,  0.00009743f, -0.00002094f,  0.00007350f, -0.00002245f,
     0.00016457f,  0.00000829f, -0.00003957f,  0.00018560f, -0.00004233f,  0.00024310f, -0.00006945f,  0.00008233f,
     0.00012415f, -0.00015895f,  0.00014440f, -0.00010686f,  0.00005893f,  0.00004207f, -0.00004634f,  0.00008848f,
     0.00000849f,  0.00006050f, -0.00014060f, -0.00001492f, -0.00016453f,  0.00012593f, -0.00008754f, -0.00011331f,
     0.00026777f, -0.00015887f,  0.00018136f,  0.00011455f,  0.00011351f,  0.00004386f, -0.00000035f,  0.00028964f,
    -0.00001671f,  0.00007067f,  0.00022109f,  0.00004677f,  0.00024041f,  0.00016109f,  0.00011013f,  0.00011850f,
     0.00034084f,  0.00008936f,  0.00017385f, -0.00006952f,  0.00008161f,  0.00024173f,  0.00010161f,  0.00011923f,
    -0.00008049f, -0.00001224f,  0.00001520f,  0.00009527f, -0.00007063f,  0.00223243f,  0.01091162f,  0.02186079f,
     0.02490066f,  0.02466056f,  0.02009664f,  0.01261149f,  0.00664761f,  0.00676710f,  0.00428081f,  0.01041192f,
     0.01109171f,  0.01261799f,  0.01413942f,  0.02009720f,  0.02062620f,  0.00429244f, -0.00268570f, -0.01276454f,
    -0.01289729f, -0.01920139f, -0.01968186f, -0.02483547f, -0.03128100f, -0.03172839f, -0.01788233f, -0.02002583f,
    -0.01587889f, -0.01876442f, -0.03975178f, -0.03240529f, -0.04255311f, -0.02844282f, -0.01817872f, -0.02676224f,
    -0.02342385f, -0.01804121f, -0.00748211f, -0.00089610f,  0.00156367f, -0.00768766f, -0.00362675f,  0.00430866f,
    -0.00586281f,  0.00511840f,  0.00578396f,  0.01174797f,  0.00551932f,  0.01447127f,  0.02987738f,  0.01583964f,
     0.01791683f,  0.01447441f,  0.01062178f,  0.00077455f,  0.01999040f,  0.00965749f,  0.00473094f,  0.01783826f,
     0.00793078f,  0.00787534f,  0.01410227f,  0.01840015f, -0.00119481f,  0.00185921f,  0.01388956f, -0.00639452f,
     0.00182004f,  0.01044317f,  0.00494055f,  0.00378837f,  0.00109018f,  0.00479160f, -0.00093658f,  0.00561917f,
     0.00266781f, -0.00782358f,  0.00111978f, -0.00719054f, -0.00158470f,  0.00024927f, -0.00235777f,  0.00378476f,
    -0.00929739f,  0.00226540f, -0.00686146f, -0.00021256f, -0.00330069f, -0.01694615f, -0.00058865f, -0.00139542f,
    -0.00453258f, -0.00355112f,  0.00166470f, -0.00144720f, -0.00239688f, -0.00144846f,  0.00357447f, -0.01198531f,
    -0.00057138f,  0.00323734f, -0.00882557f,  0.00616988f,  0.00044877f,  0.00978244f, -0.00430341f,  0.00359046f,
     0.00335720f, -0.00643052f,  0.00438204f, -0.00363544f,  0.00014233f, -0.00195475f,  0.00066526f,  0.00426282f,
    -0.00040604f, -0.00511807f,  0.00246629f, -0.00280134f, -0.00774824f,  0.00105216f, -0.00642223f,  0.00099925f,
    -0.00782133f, -0.00010822f, -0.00049867f, -0.00855196f,  0.00254369f, -0.00451518f, -0.00632099f, -0.00581216f,
     0.00053122f,  0.00123689f, -0.00404988f,  0.00426228f,  0.00517962f, -0.00403936f,  0.00719650f,  0.00576313f,
     0.00310632f,  0.00897773f,  0.00105176f,  0.01131391f,  0.00100748f,  0.00750664f,  0.01567539f,  0.00654460f,
     0.01454716f,  0.00943168f,  0.01098377f,  0.00843230f,  0.00665842f,  0.00956414f,  0.00305394f,  0.00230297f,
     0.00464755f,  0.00785945f,  0.00438020f,  0.00704300f,  0.00065315f,  0.00215907f, -0.00225995f, -0.00042777f,
     0.00633633f, -0.00703060f,  0.00251860f, -0.00544310f, -0.00451171f,  0.00057034f, -0.00345282f,  0.00425123f,
    -0.00367034f, -0.00047949f, -0.00098283f, -0.00170086f, -0.00156006f, -0.00510864f, -0.00040882f, -0.00367432f,
     0.00053859f, -0.00358700f, -0.00003623f, -0.00176419f, -0.00396731f,  0.00026147f, -0.00721936f, -0.00000088f,
    -0.00560772f, -0.00376484f, -0.00369559f, -0.00764455f, -0.00014468f, -0.00498227f, -0.00286581f, -0.00173661f,
    -0.00436404f, -0.00491419f, -0.00476705f, -0.00096522f, -0.00447868f, -0.00174404f, -0.00613105f,  0.00023149f,
    -0.00153694f, -0.00205523f,  0.00220405f, -0.00286164f,  0.00214336f, -0.00598923f, -0.00014349f,  0.00113089f,
    -0.00313676f,  0.00182432f, -0.00273224f,  0.00458326f,  0.00020947f, -0.00076456f,  0.00127742f, -0.00198659f,
     0.00240636f, -0.00124786f, -0.00001307f, -0.00179509f,  0.00318269f,  0.00001309f,  0.00012499f,  0.00239237f,
    -0.00014187f,  0.00364116f, -0.00161689f,  0.00173514f,  0.00059810f, -0.00182651f,  0.00201152f, -0.00073208f,
     0.00348455f, -0.00060217f,  0.00152584f,  0.00246791f,  0.00073984f,  0.00270105f, -0.00121419f,  0.00115911f,
    -0.00031111f, -0.00001597f, -0.00116584f,  0.00255901f,  0.00233216f, -0.00211435f,  0.00251149f, -0.00272407f,
     0.00215301f, -0.00079722f, -0.00419331f,  0.00060772f, -0.00427799f,  0.00158818f, -0.00378589f, -0.00321836f,
    -0.00131374f, -0.00558730f, -0.00214734f, -0.00411489f, -0.00317214f, -0.00403383f, -0.00412847f, -0.00395626f,
    -0.00517191f, -0.00424503f, -0.00472210f, -0.00213552f, -0.00670431f, -0.00154937f, -0.00580027f, -0.00454093f,
    -0.00031614f, -0.00571595f,  0.00117865f, -0.00606029f, -0.00183109f, -0.00101189f, -0.00365654f,  0.00243372f,
    -0.00377567f, -0.00074903f, -0.00259412f, -0.00162104f, -0.00012829f, -0.00201647f, -0.00180305f, -0.00178128f,
     0.00082654f, -0.00263262f,  0.00186724f, -0.00032036f, -0.00110382f,  0.00020348f, -0.00437263f,  0.00287088f,
    -0.00149338f,  0.00068371f,  0.00178337f, -0.00289959f,  0.00338069f, -0.00005515f,  0.00305165f,  0.00099389f,
     0.00144296f,  0.00285114f,  0.00211149f,  0.00343393f,  0.00081840f,  0.00360319f,  0.00174923f,  0.00508197f,
     0.00227359f,  0.00270413f,  0.00475521f, -0.00101329f,  0.00571070f, -0.00104766f,  0.00261255f,  0.00343351f,
     0.00009081f,  0.00552632f, -0.00226275f,  0.00266745f,  0.00098618f,  0.00056368f,  0.00165063f, -0.00094366f,
     0.00053144f, -0.00051179f,  0.00162266f, -0.00108780f,  0.00072025f,  0.00020745f, -0.00037375f,  0.00094041f,
    -0.00238168f,  0.00231943f, -0.00190310f, -0.00086867f,  0.00178739f, -0.00316391f,  0.00224101f, -0.00248732f,
     0.00132529f, -0.00105523f, -0.00168293f,  0.00198739f, -0.00186443f, -0.00070202f, -0.00165343f,  0.00017494f,
    -0.00182311f,  0.00030166f, -0.00026531f, -0.00162325f,  0.00007062f, -0.00224074f,  0.00188265f, -0.00232495f,
    -0.00132870f,  0.00129200f, -0.00289642f,  0.00181176f, -0.00247138f, -0.00026729f,  0.00006465f, -0.00219442f,
     0.00203180f, -0.00039570f, -0.00088596f,  0.00136195f,  0.00028981f,  0.00101133f, -0.00049837f,  0.00087251f,
     0.00080017f,  0.00066078f, -0.00039542f,  0.00251864f,  0.00116946f,  0.00085295f,  0.00381444f, -0.00120918f,
     0.00354037f, -0.00025121f,  0.00201838f,  0.00158278f,  0.00003160f,  0.00422331f, -0.00014369f,  0.00255287f,
     0.00225501f,  0.00119054f,  0.00184265f,  0.00040799f,  0.00270225f,  0.00058744f,  0.00182112f,  0.00082472f,
     0.00204735f,  0.00158550f,  0.00056760f,  0.00352285f,  0.00015020f,  0.00315690f,  0.00056045f,  0.00097963f,
     0.00171611f, -0.00005421f,  0.00308821f,  0.00016155f
};

/* Delay line for anti-noise output history (used by Feedback Neutralizer) - Length: FB_FILTER_TAP (500) */
float Y_buf[FB_FILTER_TAP] = {0.0f};

/* Delay line for secondary path filtering - Length: SEC_FILTER_TAP (500) */
float X_hat[SEC_FILTER_TAP] = {0.0f};

/* Filtered-X delay line: x'(n), x'(n-1), ... - Length: FILTER_TAP (256) */
float Fx[FILTER_TAP] = {0.0f};

/* =========================================================================
 * Runtime Tuning and Control Variables (Accessible via CCS Expressions)
 * ========================================================================= */
volatile int   anc_enable             = 1;            /* 1: ANC Active, 0: ANC Bypassed / Muted */
volatile int   speaker_channel        = 2;            /* 0: Both L+R (Matches working NLMS0), 1: Right-Only, 2: Left-Only */
volatile int   test_tone_mode         = 0;            /* 1: Output test sound to verify speaker/amp wiring, 0: Normal ANC */
volatile int   inv_polarity           = 0;            /* 0: Standard inverted phase (-y), 1: Non-inverted (+y) */
volatile int   reset_filter           = 0;            /* Set to 1 in Expressions to instantly reset filter weights to 0 */
volatile float mu                     = 0.001f;       /* Step size for smooth, stable convergence without jitter */
volatile float leakage                = 1.0f;         /* 1.0f = Pure unconstrained LMS for maximum cancellation (23 dB) */
volatile float noise_gate_thresh      = 0.0f;         /* 0.0f: Disabled to prevent gate chattering */

/* Acoustic Feedback Neutralization Controls */
volatile int   feedback_cancel_enable = 1;            /* 1: Feedback Neutralizer ON, 0: Bypassed (original echo) */
volatile float fb_scale               = 1.0f;         /* 1.0 = 100% feedback echo subtraction */

/* Test sound PRNG state */
static uint32_t test_prng = 12345;

/* Real-time channel routing controls (Switchable in CCS Expressions) */
volatile int   ref_channel      = 1;            /* 1: Left is Ref, Right is Err / 2: Right is Ref, Left is Err */

/* Real-time CCS Graph Monitoring Buffers (Matches MIC_TEST & NLMS0) */
int16_t ref_mic_buf[MONITOR_BUFLEN]   = {0};      /* Raw 16-bit Reference mic ADC (Contains feedback echo) */
int16_t ref_clean_buf[MONITOR_BUFLEN] = {0};      /* Cleaned Reference mic ADC (Feedback echo REMOVED!) */
int16_t err_mic_buf[MONITOR_BUFLEN]   = {0};      /* Raw 16-bit Error mic ADC (Plot with: 16 bit signed integer) */
int16_t spk_out_buf[MONITOR_BUFLEN]   = {0};      /* Raw 16-bit Canceling speaker output (Plot with: 16 bit signed integer) */
float   ref_x_buf[MONITOR_BUFLEN]     = {0.0f};   /* Normalized float Reference mic x(n) (Plot with: 32 bit float) */
float   err_e_buf[MONITOR_BUFLEN]     = {0.0f};   /* Normalized float Error mic e(n) (Plot with: 32 bit float) */
float   anti_y_buf[MONITOR_BUFLEN]    = {0.0f};   /* Anti-noise float output y(n) (Plot with: 32 bit float) */
float   fx_buf[MONITOR_BUFLEN]        = {0.0f};   /* Filtered-X signal x'(n) */
static int mon_idx = 0;

/* Real-time debug scalar variables for CCS Expressions watch */
volatile float debug_x        = 0.0f;             /* Raw Reference mic input (x_n) */
volatile float debug_x_clean  = 0.0f;             /* Cleaned Reference mic input without echo */
volatile float debug_y_fb     = 0.0f;             /* Feedback echo signal being subtracted */
volatile float debug_e        = 0.0f;             /* Error mic input */
volatile float debug_y        = 0.0f;             /* Filter anti-noise output */
volatile float debug_fx       = 0.0f;             /* Filtered-X sample */
volatile float debug_err_power= 0.0f;             /* Running energy of error mic */
volatile float debug_ref_pwr  = 0.0f;             /* Running energy of ref mic */
volatile float debug_w_max    = 0.0f;             /* Peak weight magnitude in W(z) */

/* Real-time Performance & XRUN Diagnostics (Watch in CCS Expressions) */
volatile uint32_t isr_cycles     = 0;       /* Last ISR execution time in CPU cycles */
volatile uint32_t isr_max_cycles = 0;       /* Peak ISR execution time (budget: 18,750 @ 300MHz) */
volatile uint32_t xrun_cnt       = 0;       /* McASP Overrun/Underrun counter (0 = perfect real-time) */


/* =========================================================================
 * Data and Buffer Initialization
 * ========================================================================= */
void ClearData( void ) {
    int i;
    for(i = 0; i < FILTER_TAP; i++) {
        W[i]  = 0.0f;
        X[i]  = 0.0f;
        Fx[i] = 0.0f;
    }
    for(i = 0; i < SEC_FILTER_TAP; i++) {
        X_hat[i] = 0.0f;
    }
    for(i = 0; i < FB_FILTER_TAP; i++) {
        Y_buf[i] = 0.0f;
    }
    for(i = 0; i < MONITOR_BUFLEN; i++) {
        ref_mic_buf[i]   = 0;
        ref_clean_buf[i] = 0;
        err_mic_buf[i]   = 0;
        spk_out_buf[i]   = 0;
        ref_x_buf[i]     = 0.0f;
        err_e_buf[i]     = 0.0f;
        anti_y_buf[i]    = 0.0f;
        fx_buf[i]        = 0.0f;
    }
    mon_idx = 0;
    led_cnt = 0;
    debug_err_power = 0.0f;

    /* Turn on LED D6 to show ANC is active by default */
    if(anc_enable) {
        LED_On( LED_D6 );
    } else {
        LED_Off( LED_D6 );
    }
}



/* =========================================================================
 * McASP Audio Interrupt Service Routine (Real-Time FxNLMS)
 * Execution frequency: 16,000 Hz (every 62.5 microseconds)
 * ========================================================================= */
void MCASP_ISR( void ) {
    uint32_t t_start = TSCL;
    uint32_t rx_sample = mcaspRegs->RBUF14;
    int k;

    /* Check McASP Overrun / Underrun flags */
    if (mcaspRegs->RSTAT & 0x01) { xrun_cnt++; mcaspRegs->RSTAT = 0x01; }
    if (mcaspRegs->XSTAT & 0x01) { xrun_cnt++; mcaspRegs->XSTAT = 0x01; }

    /* 1. Receive 2-channel 16-bit audio data from Line-in */
    Int16 ch_left  = (Int16)(rx_sample >> 16);     /* Line-in Left (Tip) */
    Int16 ch_right = (Int16)(rx_sample & 0xFFFF);  /* Line-in Right (Ring) */

    /*
     * 2. Channel mapping: split reference mic x(n) and error mic e(n)
     * ref_channel:
     *   1: Left = Reference Mic, Right = Error Mic (Default)
     *   2: Right = Reference Mic, Left = Error Mic (Swap if wired inversely)
     */
    Int16 raw_ref = (ref_channel == 2) ? ch_right : ch_left;
    Int16 raw_err = (ref_channel == 2) ? ch_left  : ch_right;

    /* 1st-order DC Blocker Filter (removes ADC DC bias < 15 Hz) */
    static float dc_x_in = 0.0f, dc_x_out = 0.0f;
    static float dc_e_in = 0.0f, dc_e_out = 0.0f;

    float raw_x_f = (float)raw_ref / 32768.0f;
    float x_n = raw_x_f - dc_x_in + 0.995f * dc_x_out;
    dc_x_in = raw_x_f;
    dc_x_out = x_n;

    float raw_e_f = (float)raw_err / 32768.0f;
    float e_n = raw_e_f - dc_e_in + 0.995f * dc_e_out;
    dc_e_in = raw_e_f;
    dc_e_out = e_n;

    /*
     * 2-1. Acoustic Feedback Neutralization (Canceling Speaker -> Reference Mic)
     * Estimates speaker sound that arrived at Ref Mic and subtracts it out.
     */
    float y_fb = 0.0f;
    if(feedback_cancel_enable) {
        for(k = 0; k < FB_FILTER_TAP; k++) {
            y_fb += F_hat[k] * Y_buf[k];
        }
    }
    float x_clean = x_n - fb_scale * y_fb;

    /* Reference Signal Power Estimation & Noise Gate Detector (Monitors pure noise) */
    static float ref_pwr_est = 0.0f;
    ref_pwr_est = 0.99f * ref_pwr_est + 0.01f * (x_clean * x_clean);
    int sound_active = (noise_gate_thresh <= 0.0f) || (ref_pwr_est >= noise_gate_thresh);

    /* Record samples into 16-bit graph buffers (Matches MIC_TEST!) */
    ref_mic_buf[mon_idx]   = raw_ref;
    ref_clean_buf[mon_idx] = (Int16)(x_clean * 32767.0f);
    err_mic_buf[mon_idx]   = raw_err;

    /* 3. Update reference signal delay line: X[k] with pure noise x_clean */
    for(k = FILTER_TAP - 1; k > 0; k--) {
        X[k] = X[k - 1];
    }
    X[0] = x_clean;

    /* Check if runtime reset requested via CCS Expressions */
    if(reset_filter) {
        reset_filter = 0;
        for(k = 0; k < FILTER_TAP; k++) {
            W[k]  = 0.0f;
            X[k]  = 0.0f;
            Fx[k] = 0.0f;
        }
        for(k = 0; k < SEC_FILTER_TAP; k++) {
            X_hat[k] = 0.0f;
        }
        for(k = 0; k < FB_FILTER_TAP; k++) {
            Y_buf[k] = 0.0f;
        }
    }

    /* 4. Calculate primary filter output: y(n) = W^T * X */
    float y = 0.0f;
    if(anc_enable && sound_active) {
        for(k = 0; k < FILTER_TAP; k++) {
            y += W[k] * X[k];
        }
    }

    /* 5. Phase inversion for acoustic cancellation (Switchable via inv_polarity) */
    float y_out;
    if(inv_polarity == 0) {
        y_out = -y;   /* Standard inverted anti-noise */
    } else {
        y_out = y;    /* Inverted hardware polarity fallback */
    }

    /* Saturation and clipping protection */
    if (y_out > 1.0f)       y_out = 1.0f;
    else if (y_out < -1.0f) y_out = -1.0f;

    Int16 out_val = 0;
    if(test_tone_mode) {
        /* Direct speaker test mode: generate soft white noise (-12 dBFS) to verify hardware */
        test_prng = test_prng * 1664525 + 1013904223;
        out_val = (int16_t)(test_prng >> 16) / 4;
    } else if(sound_active) {
        out_val = (Int16)(y_out * 32767.0f);
    } else {
        out_val = 0;    /* External sound is OFF -> 100% mute speaker */
    }

    /* 6. Output to secondary speaker via McASP */
    uint16_t out_left  = 0x0000;
    uint16_t out_right = 0x0000;

    if(speaker_channel == 0) {
        out_left  = (uint16_t)out_val;  /* Output to Both Left & Right */
        out_right = (uint16_t)out_val;
    } else if(speaker_channel == 1) {
        out_left  = 0x0000;             /* Left strictly 0x0000 for TPA3116D2 protection */
        out_right = (uint16_t)out_val;  /* Right channel speaker */
    } else if(speaker_channel == 2) {
        out_left  = (uint16_t)out_val;  /* Left channel speaker */
        out_right = 0x0000;
    }

    mcaspRegs->XBUF13 = ((uint32_t)out_left << 16) | (uint32_t)out_right;

    /* Update anti-noise history buffer for feedback neutralizer */
    for(k = FB_FILTER_TAP - 1; k > 0; k--) {
        Y_buf[k] = Y_buf[k - 1];
    }
    Y_buf[0] = (float)out_val / 32768.0f;

    /* 7. Secondary Path Filtering: Calculate Filtered-X signal: x'(n) = S_hat * x_clean(n) */
    /* 7-1. Update secondary path delay buffer */
    for(k = SEC_FILTER_TAP - 1; k > 0; k--) {
        X_hat[k] = X_hat[k - 1];
    }
    X_hat[0] = x_clean;

    /* 7-2. S_hat FIR filtering (500 taps) */
    float fx_n = 0.0f;
    for(k = 0; k < SEC_FILTER_TAP; k++) {
        fx_n += S_hat[k] * X_hat[k];
    }

    /* 7-3. Update Filtered-X delay line: Fx[k] */
    for(k = FILTER_TAP - 1; k > 0; k--) {
        Fx[k] = Fx[k - 1];
    }
    Fx[0] = fx_n;

    /* 8. Normalized LMS (NLMS) Primary Weight Adaptation with Stability Safeguards */
    if(anc_enable && !test_tone_mode) {
        if(sound_active) {
            /* 8-1. Energy normalization with robust regularization (prevents near-zero divergence) */
            float norm_pwr = 0.0005f;
            for(k = 0; k < FILTER_TAP; k++) {
                norm_pwr += Fx[k] * Fx[k];
            }

            /* 8-2. Normalized step size with upper-bound safety limiter */
            float mu_norm = mu / norm_pwr;
            if(mu_norm > 2.0f) mu_norm = 2.0f;

            /* 8-3. Update primary filter weights with leakage, polarity switch, and tap clamping */
            float step = (inv_polarity == 0) ? (mu_norm * e_n) : (-mu_norm * e_n);
            for(k = 0; k < FILTER_TAP; k++) {
                float w_new = W[k] * leakage + step * Fx[k];
                if(w_new > 0.5f)       w_new = 0.5f;
                else if(w_new < -0.5f) w_new = -0.5f;
                W[k] = w_new;
            }
        } else {
            /* External sound is OFF: fast decay weights to silence speaker and stop howling */
            for(k = 0; k < FILTER_TAP; k++) {
                W[k] *= 0.99f;
            }
        }
    }

    /* Record active signals into waveform buffers for CCS Graph monitoring */
    spk_out_buf[mon_idx] = out_val;
    ref_x_buf[mon_idx]   = x_clean;
    err_e_buf[mon_idx]   = e_n;
    anti_y_buf[mon_idx]  = y;
    fx_buf[mon_idx]      = fx_n;
    if(++mon_idx >= MONITOR_BUFLEN) {
        mon_idx = 0;
    }

    /* Update real-time debug scalar variables */
    debug_x       = x_n;
    debug_x_clean = x_clean;
    debug_y_fb    = y_fb;
    debug_e       = e_n;
    debug_y       = y;
    debug_fx      = fx_n;
    debug_err_power = 0.999f * debug_err_power + 0.001f * (e_n * e_n);
    debug_ref_pwr   = ref_pwr_est;

    /* Update real-time performance diagnostics */
    uint32_t elapsed = TSCL - t_start;
    isr_cycles = elapsed;
    if (elapsed > isr_max_cycles) {
        isr_max_cycles = elapsed;
    }

    ++led_cnt;

}


/* =========================================================================
 * SYS/BIOS Hwi and Idle function definitions (matches app.cfg)
 * ========================================================================= */
void TIMER1_TINT12_ISR( void ) {
    StopTimer( (CSL_TmrRegsOvly)CSL_TMR_1_REGS );
    GPIO_ClearInterruptState( GP2 );
    ClearInterrupt( INT_NUM_GPIO_B2 );
    EnableInterrupt( INT_NUM_GPIO_B2 );
}

/* User Push Button ISR: Toggle ANC ON / OFF */
void GPIO_PUSHBUTTON_ISR( void ) {
    DisableInterrupt( INT_NUM_GPIO_B2 );
    StartTimer( (CSL_TmrRegsOvly)CSL_TMR_1_REGS );

    /* Toggle ANC enable state */
    anc_enable = !anc_enable;

    if(anc_enable) {
        LED_On( LED_D6 );   /* LED D6 ON = ANC Active */
    } else {
        LED_Off( LED_D6 );  /* LED D6 OFF = ANC Bypassed */
    }
}

/* Idle LED Task: Heartbeat on LED D4 & background monitoring */
void IdleLED( void ) {
    int k;
    float max_w = 0.0f;
    for(k = 0; k < FILTER_TAP; k++) {
        float abs_w = fabsf(W[k]);
        if(abs_w > max_w) max_w = abs_w;
    }
    debug_w_max = max_w;

    if( led_cnt >= (SAMPLING_FREQ >> 1) ) {
        LED_Toggle( LED_D4 );
        led_cnt = 0;
    }
}
