# Multimodal Deep Learning for London House-Price Prediction

## 🏙️ Overview 
London house prices are notoriously difficult to predict because many intangible and visual factors aren't captured by traditional datasets. Features like a home's curb appeal, interior condition, natural lighting, and neighborhood ambiance can strongly influence price but are often invisible in purely tabular data.

This project addresses that gap by training a multimodal deep learning model that integrates diverse data sources to predict property prices. We combine:

● **Rightmove listings** – Structured data about each property (e.g. size, type, location).

● **Police.UK crime statistics** – Neighborhood safety indicators.

● **Listing descriptions** – Unstructured text from the property listing.

● **Property photographs** – Images of the property's interior and exterior.

● **Satellite imagery** – Aerial view of the property and surrounding area.

● **Google Street View** – 360° street-level panoramas of the property’s vicinity.

**Model Architecture**: Each modality is processed with a dedicated sub-network. We use CNNs (ResNet-18) for images (property photos, satellite, Street View), a BERT-based transformer for textual descriptions, and a feed-forward MLP for tabular features. These outputs are concatenated in a late-fusion layer, enabling the model to learn a joint representation before the final price regression layer.

**Dataset**: We compiled a custom dataset of 41,835 London properties spanning all boroughs, covering listings from April–June 2025. Each property entry includes the sale price and all the above data modalities.

**Preprocessing**:
● Applied a pre-trained image classifier (via Hugging Face) to tag content of property photos and filter out non-informative images.

● Performed kernel density estimation (KDE) on crime data to create a smooth local crime-rate feature for each location.

● Cropped each Street View panorama into directional flat images (capturing consistent street-facing views).

● Truncated listing text descriptions to 512 tokens to fit the BERT model input limit.

## 📊 Key Results
We evaluated performance with cross-validation and a held-out test set. The following improvements in RMSE (error) are measured relative to a baseline model using only tabular listing features (denoted TR):

● **Adding crime data (TRC vs TR)**: RMSE decreased by 6.5%.

● **Adding listing text (TR + TD vs TR)**: RMSE decreased by 18.2%.

● **Adding property images (TR + PI vs TR)**: RMSE decreased by 27.2%.

● **Adding Street View (TR + SV vs TR)**: RMSE decreased by 41.6%.

● **Adding text on top of full visual model**: Increased error by 33% (text hurt performance when images were already included).

**Best Model**: The top-performing model combined all modalities except text (TRC + PI + SI + SV) for the lowest error. Including satellite imagery (SI) provided only a minor gain, but it was part of the best configuration.

## 📌 Implications
● **Images are key**: Visual data proved to be the most impactful. Property photos and Street View images provided the largest boosts in accuracy (whereas adding satellite imagery yielded only a minor benefit). This indicates that a home's appearance and its street context carry crucial price information.

● **Crime data adds value**: Incorporating neighborhood crime statistics consistently improved predictions, confirming that local safety is an important factor in housing prices.

● **Text vs. visuals**: Listing description text is useful as a substitute when images are missing (it can highlight features and condition), but if high-quality images are available, the text becomes redundant or even misleading. In our experiments, adding text on top of a full multimodal model actually worsened performance.

● **What the model sees**: The strong performance from image inputs suggests the model is capturing subtle visual cues of property value – e.g. well-maintained facades, bright interiors, presence of street greenery, and signs of parking or traffic conditions. These visual signals play a big role in how buyers (and the model) perceive a home's value.

## 🙏 Acknowledgements  

This study relies on openly accessible datasets and APIs. I gratefully acknowledge the following sources for making their data available to the public:  

- **Rightmove Sale Housing Listings:** [https://www.rightmove.co.uk/](https://www.rightmove.co.uk/)  
- **UK Police Crime Data (London, May 2024 – April 2025):** [https://data.police.uk/data/](https://data.police.uk/data/)  
- **Ordnance Survey Open Roads Dataset:** [https://www.ordnancesurvey.co.uk/products/os-open-roads](https://www.ordnancesurvey.co.uk/products/os-open-roads)  
- **Google Maps Static API:** [https://developers.google.com/maps/documentation/maps-static/overview](https://developers.google.com/maps/documentation/maps-static/overview)  

Their provision of open and well-documented resources has been critical in enabling reproducible research and supporting this multimodal housing price prediction project. 
