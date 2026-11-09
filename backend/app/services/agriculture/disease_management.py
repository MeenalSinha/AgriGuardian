"""ICAR-aligned Integrated Pest & Disease Management (IPM) guidelines for tomato foliar conditions."""

from typing import Dict
from backend.app.schemas.disease import ManagementGuidelines

TOMATO_DISEASE_IPM: Dict[str, ManagementGuidelines] = {
    "Tomato___Bacterial_spot": ManagementGuidelines(
        cultural_practices=[
            "Avoid overhead sprinkler irrigation to prevent bacterial splash between plants.",
            "Remove and safely burn or bury heavily infected foliage and plant residue.",
            "Maintain optimal row spacing (60x45 cm) to facilitate rapid canopy drying.",
            "Use certified disease-free seeds and practice 2-year rotation away from solanaceous crops.",
        ],
        biological_controls=[
            "Foliar spray of Pseudomonas fluorescens or Bacillus subtilis @ 5g/L at 15-day intervals.",
            "Apply Neem Seed Kernel Extract (NSKE 5%) as an organic preventative foliar rinse.",
        ],
        chemical_precautions=[
            "Spray Copper Oxychloride 50% WP (2.5 g/L) combined with Streptocycline (1 g/10 L water).",
            "Apply in early morning or late afternoon; avoid application during peak flowering.",
            "Always wear personal protective gear and adhere strictly to CIBRC waiting periods.",
        ],
        kvk_advisory="Consult your local Krishi Vigyan Kendra (KVK) officer for seed treatment protocols and bactericide rotation to prevent resistant strain buildup.",
    ),
    "Tomato___Early_blight": ManagementGuidelines(
        cultural_practices=[
            "Prune infected lower foliage touching the soil to eliminate primary inoculation sources.",
            "Apply straw or plastic mulch to suppress soil splash during rainfall.",
            "Maintain balanced nitrogen fertilization; avoid lush vegetative growth susceptible to Alternaria.",
            "Stake plants upright to improve ventilation through the lower canopy.",
        ],
        biological_controls=[
            "Soil incorporation and foliar spray of Trichoderma harzianum @ 10g/L.",
            "Spray bio-formulations of Ampelomyces quisqualis for fungal mycoparasitism.",
        ],
        chemical_precautions=[
            "Spray Mancozeb 75% WP @ 2.0 g/L or Chlorothalonil 75% WP @ 2.0 g/L at first target-ring appearance.",
            "For active secondary infections, spray Difenoconazole 25% EC @ 1 ml/L.",
            "Alternate systemic fungicides with contact fungicides to delay resistance development.",
        ],
        kvk_advisory="Contact your district KVK agronomist if concentric target spots spread above mid-canopy or during warm, humid monsoon spells.",
    ),
    "Tomato___Late_blight": ManagementGuidelines(
        cultural_practices=[
            "Immediately destroy and bury water-soaked diseased foliage to halt epidemic oospore sporulation.",
            "Ensure rapid soil drainage; avoid fields prone to water stagnation.",
            "Monitor regional weather alerts for temperatures below 22°C paired with relative humidity > 90%.",
        ],
        biological_controls=[
            "Preventative spray of bio-agent Trichoderma viride enriched with well-rotted farmyard manure.",
            "Spray copper soap or fermented butter-milk (chaas) dilution @ 20ml/L in organic setups.",
        ],
        chemical_precautions=[
            "Spray Metalaxyl 8% + Mancozeb 64% WP (Ridomil MZ) @ 2.5 g/L or Cymoxanil 8% + Mancozeb 64% WP @ 2.0 g/L.",
            "Ensure thorough coverage of both upper and lower leaf surfaces during cool humid conditions.",
            "Late blight can destroy entire crops within 48-72 hours; act immediately upon notification.",
        ],
        kvk_advisory="CRITICAL ALERT: Report late blight outbreaks to your district KVK immediately. Synchronized community-level spraying is essential to prevent regional crop collapse.",
    ),
    "Tomato___Leaf_Mold": ManagementGuidelines(
        cultural_practices=[
            "Dramatically improve greenhouse or polytunnel ventilation to reduce ambient humidity below 85%.",
            "Shift exclusively to drip irrigation; eliminate all surface splashing.",
            "Widen plant spacing and prune suckers to promote cross-ventilation.",
        ],
        biological_controls=[
            "Dust with elemental sulfur (wettable sulfur 80% WP @ 2.0 g/L) at early symptom emergence.",
            "Apply bio-fungicidal strains of Bacillus amyloliquefaciens.",
        ],
        chemical_precautions=[
            "Apply Azoxystrobin 23% SC @ 1.0 ml/L or Carbendazim 50% WP @ 1.0 g/L if humidity cannot be lowered.",
            "Target sprays onto lower leaf surfaces where velvety olive-green mold sporulates.",
        ],
        kvk_advisory="Consult KVK horticulture specialists for greenhouse relative-humidity regulation and mold-resistant hybrid recommendations.",
    ),
    "Tomato___Septoria_leaf_spot": ManagementGuidelines(
        cultural_practices=[
            "Remove and safely compost weed reservoirs (e.g., wild solanum, horsenettle) bordering fields.",
            "Clean and sterilize stakes, cages, and pruning equipment before field operations.",
            "Water solely at the base using furrow or drip irrigation to keep leaves dry.",
        ],
        biological_controls=[
            "Apply foliar bio-spray of Trichoderma asperellum @ 5 g/L.",
            "Foliar wash with garlic-chilli extract @ 2% for organic fungal deterrence.",
        ],
        chemical_precautions=[
            "Spray Chlorothalonil 75% WP @ 2.0 g/L or Copper Hydroxide 53.8% DF @ 2.0 g/L.",
            "Begin treatment promptly when small circular dark-bordered spots appear on lower leaflets.",
        ],
        kvk_advisory="Consult KVK officers if lower leaf defoliation exceeds 20% to prevent fruit sunburn and yield collapse.",
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": ManagementGuidelines(
        cultural_practices=[
            "Use a high-pressure water spray to dislodge mite colonies and webbing on leaf undersides.",
            "Avoid excessive nitrogen fertilizer application, which promotes tender, mite-favored foliage.",
            "Maintain adequate soil moisture; drought-stressed plants are especially prone to mite flare-ups.",
        ],
        biological_controls=[
            "Release predatory mites (Phytoseiulus persimilis or Neoseiulus californicus) @ 2-5 mites/plant.",
            "Spray cold-pressed Neem oil (10,000 ppm) @ 3-5 ml/L mixed with soap emulsifier.",
        ],
        chemical_precautions=[
            "Apply acaricides: Spiromesifen 22.9% SC @ 1.0 ml/L or Propargite 57% EC @ 2.0 ml/L.",
            "Ensure complete coverage of leaf undersides where colonies dwell; rotate chemical classes to avoid resistance.",
        ],
        kvk_advisory="Examine leaf undersides with a 10x hand lens at your local KVK diagnostic lab to differentiate mites from nutritional thrips damage.",
    ),
    "Tomato___Target_Spot": ManagementGuidelines(
        cultural_practices=[
            "Remove old diseased crop residues after harvest; deep plow fields to bury overwintering spores.",
            "Ensure proper drainage and avoid continuous mono-cropping of tomato or cucumber.",
            "Prune canopy dense areas to accelerate dew evaporation.",
        ],
        biological_controls=[
            "Soil drench and foliar spray of Trichoderma harzianum enriched compost.",
        ],
        chemical_precautions=[
            "Apply Pyraclostrobin 20% WG @ 1.0 g/L or Azoxystrobin + Difenoconazole @ 1.0 ml/L.",
            "Monitor fruit calyx for sunken lesions and spray before fruit infection progresses.",
        ],
        kvk_advisory="Seek KVK advice on differentiating target spot from early blight through microscopic diagnostic isolation.",
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ManagementGuidelines(
        cultural_practices=[
            "Erect bright yellow sticky traps @ 15-20 traps/acre across field borders to intercept whiteflies.",
            "Raise tomato seedlings in nurseries enclosed by 40-50 mesh insect-proof insect netting.",
            "Rogue out and destroy viral stunted plants with upward-cupped leaves within the first 30 days.",
            "Grow 2-3 border rows of maize, sorghum, or pearl millet as living physical whitefly barriers.",
        ],
        biological_controls=[
            "Spray entomopathogenic fungus Verticillium lecanii or Beauveria bassiana @ 5 g/L against whitefly nymphs.",
            "Spray Neem oil (10,000 ppm) @ 3 ml/L at weekly intervals.",
        ],
        chemical_precautions=[
            "Fungicides CANNOT cure viral infections. Management strictly targets the whitefly insect vector.",
            "Spray systemic insecticide: Imidacloprid 17.8% SL @ 0.3 ml/L or Diafenthiuron 50% WP @ 1.0 g/L.",
            "Rotate with Spiromesifen or Pyriproxyfen to avoid rapid insecticide resistance.",
        ],
        kvk_advisory="Consult KVK urgently for regional whitefly population alerts and recommendations for proven TYLCV-resistant tomato hybrids.",
    ),
    "Tomato___Tomato_mosaic_virus": ManagementGuidelines(
        cultural_practices=[
            "Strict sanitation: Dip hands and pruning tools in 10% trisodium phosphate (TSP) or 20% skim milk before handling plants.",
            "Strictly prohibit smoking, bidi, or tobacco chewing inside tomato fields (tobacco carries mosaic virus).",
            "Immediately rogue, bag, and burn mottled or blistered plants; do not compost infected residue.",
            "Decontaminate trellising wires and stakes before reusing in future seasons.",
        ],
        biological_controls=[
            "Seed soaking in 10% trisodium phosphate solution for 15 minutes followed by running water rinse.",
            "Select ToMV-resistant rootstocks and cultivars (look for 'Tm-2' resistance gene in seed catalogs).",
        ],
        chemical_precautions=[
            "Viruses cannot be cured with chemical bactericides or fungicides. Do not apply fungicides for mosaic virus.",
            "Control chewing insects to prevent secondary viral mechanical transmission across foliage.",
        ],
        kvk_advisory="Submit leaf tissue to the nearest KVK or state agricultural university virology lab for definitive serological (ELISA) confirmation.",
    ),
    "Tomato___healthy": ManagementGuidelines(
        cultural_practices=[
            "Maintain balanced N-P-K fertilization based on a verified soil health card.",
            "Schedule regular drip irrigation during critical flowering and fruit development stages.",
            "Apply organic mulch around root zones to conserve moisture and stabilize soil temperatures.",
        ],
        biological_controls=[
            "Apply prophylactic bio-fertilizers (Azotobacter, Phosphobacteria, and K-solubilizers) @ 5 kg/acre in FYM.",
            "Spray preventative neem extract (NSKE 5%) monthly to ward off sap-sucking pests.",
        ],
        chemical_precautions=[
            "No chemical pesticides or fungicides are necessary or advised for healthy foliage.",
            "Avoid unnecessary prophylactic chemical sprays to protect beneficial pollinator insects and predatory fauna.",
        ],
        kvk_advisory="Continue standard bi-weekly crop scouting; consult KVK for seasonal soil fertility recommendations and micronutrient foliar sprays.",
    ),
}

UNKNOWN_DISEASE_IPM = ManagementGuidelines(
    cultural_practices=[
        "Isolate suspicious plant specimens to prevent potential transmission.",
        "Take high-resolution, well-focused photographs of both the upper and lower leaf surfaces.",
        "Record onset timeline, irrigation frequency, and recent fertilizer applications.",
    ],
    biological_controls=[
        "Apply organic neem seed kernel extract (NSKE 5%) as a gentle broad-spectrum deterrent while awaiting formal diagnosis.",
    ],
    chemical_precautions=[
        "DO NOT spray random or high-potency chemical fungicides/pesticides without definitive pathogen identification.",
        "Inappropriate chemical applications waste farmer capital and trigger chemical phytotoxicity.",
    ],
    kvk_advisory="Take a freshly plucked symptomatic leaf sample sealed in a dry envelope to your nearest Krishi Vigyan Kendra (KVK) plant pathology clinic for laboratory microscopic examination.",
)


def get_management_guidelines(class_id: str) -> ManagementGuidelines:
    """Retrieve IPM guidelines for given disease class identifier."""
    return TOMATO_DISEASE_IPM.get(class_id, UNKNOWN_DISEASE_IPM)
