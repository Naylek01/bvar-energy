# Guide d'exploitation des sources et Power Queries - `bvar_energy_raw_data.xlsx`

**Architecture, rafraichissement, garde-fous, risques et diagnostic des erreurs**  
**Version de reference : 7 aout 2026**

## 1. Objectif et principe general

Le projet utilise **un seul classeur raw vivant** :

```text
data/raw/bvar_energy_raw_data.xlsx
```

Ce fichier est mis a jour **en place**. Il n'existe pas un workbook raw par vintage. En revanche, le builder Python produit des sorties datees :

```text
data/raw/bvar_energy_raw_data.xlsx
          |
          v
src/data_pipeline/build_dataset_paper_six_models_v9.py
          |
          v
data/processed/YYYYMMDD/
```

L'objectif du classeur est de reunir les sources dans un format stable pour Python, tout en separant :

- les connexions / donnees live ;
- les tables raw intermediaires ;
- les vues finales chargees dans les onglets lus par Python ;
- les snapshots manuels necessaires pour les sources qui ne doivent pas dependre d'un refresh reseau.

## 2. Inventaire actuel du classeur

Le classeur contient les onglets suivants :

| Onglet | Role | Mise a jour | Lu par Python |
|---|---|---|---|
| `Haver` | Snapshot HICP/PPI | Manuel pour l'instant | Oui |
| `Bloomberg_live` | Formules Bloomberg | Bloomberg Add-in | Non |
| `Bloomberg_copy` | Snapshot values-only | Copie depuis `Bloomberg_live` | Oui |
| `European_Commission` | Vue WOB propre | Power Query | Oui |
| `World_Bank` | Snapshot statique Natural gas Europe | Manuel / rare | Oui |
| `Eurostat` | Vue semi-annuelle gaz/electricite | Power Query | Oui |
| `Metadata` | Contrats et audit | Manuel + futur enrichissement | Audit |

Connexions Power Query presentes :

```text
EC_WOB_RAW_FULL
EC_WOB_VIEW
EUROSTAT_ENERGY_RAW
EUROSTAT_ENERGY_VIEW
WORLD_BANK_GAS_RAW
```

Le pattern recommande est :

```text
source externe
    -> RAW query (connection only)
    -> VIEW query (nettoyage + validations)
    -> onglet final charge dans Excel
    -> builder Python
```

## 3. Politique de chargement des queries

### European Commission

```text
EC_WOB_RAW_FULL   -> Connection only
EC_WOB_VIEW       -> charge dans l'onglet European_Commission
```

`EC_WOB_RAW_FULL` peut contenir beaucoup de cellules et ne doit pas etre charge dans une feuille Excel. `EC_WOB_VIEW` contient seulement la vue exploitable.

### Eurostat

```text
EUROSTAT_ENERGY_RAW   -> Connection only
EUROSTAT_ENERGY_VIEW  -> charge dans l'onglet Eurostat
```

### World Bank

Le flux dynamique n'est plus une dependance du refresh quotidien :

```text
WORLD_BANK_GAS_RAW -> outil de maintenance uniquement
World_Bank         -> snapshot VALUES ONLY lu par Python
```

Si `WORLD_BANK_GAS_RAW` est conserve, il doit etre **Connection only** et **exclu de Refresh All**.

## 4. European Commission - Weekly Oil Bulletin (WOB)

### 4.1 Pourquoi la query utilise une URL statique

Une version dynamique a provoque `Formula.Firewall` : Power Query evaluait une premiere requete web pour trouver un lien, puis utilisait ce resultat comme nouvelle source web. La solution retenue est de pointer directement vers le document historique que la DG ENER met a jour en place.

Le contrat operationnel actuel est :

```powerquery
// URL verifiee le 2026-08-07. Le classeur historique est mis a jour
// EN PLACE par la DG ENER ; le GUID est stable depuis au moins 2024.
// Si ce refresh echoue avec une erreur Excel.Workbook incomprehensible,
// reverifier le lien sur :
//   https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en

Host = "https://energy.ec.europa.eu",
FilePath = "document/download/906e60ca-8b6a-44e7-8589-652854d2fd3f_en"
```

**Important :** la stabilite du GUID est une hypothese operationnelle observee, pas une garantie contractuelle de la Commission.

### 4.2 `EC_WOB_RAW_FULL`

Cette query :

1. telecharge le workbook depuis une racine `Web.Contents` statique ;
2. l'ouvre avec `Excel.Workbook(..., null, true)` afin de ne pas imposer un typage automatique fragile ;
3. exige la presence des feuilles :
   - `Prices with taxes`
   - `Prices wo taxes`
   - `VAT`
   - `Excise duties`
   - `Other Indirect Taxes`
   - `Consumption`
4. retire uniquement les lignes Excel physiquement vides ;
5. ajoute `source_row` et `source_sheet` ;
6. empile les feuilles ;
7. echoue explicitement si le workbook est vide ou si son contrat change.

Le garde-fou de contrat est **reference dans le chemin de dependance** du resultat. C'est important car Power Query utilise une evaluation paresseuse : un check non reference peut ne jamais etre execute.

### 4.3 `EC_WOB_VIEW`

La vue finale ne suppose plus un offset fixe du type `H + 3`.

Pour chacune des feuilles de prix, elle :

1. bufferise la feuille ;
2. cherche la ligne d'en-tete via les trois libelles de series attendus ;
3. identifie la colonne de dates par son contenu ;
4. ne garde que les lignes portant une vraie date depuis 2005 ;
5. convertit les trois prix en valeurs numeriques ;
6. refuse les dates dupliquees ;
7. utilise **`Prices wo taxes` comme calendrier maitre** ;
8. effectue un **LEFT JOIN** de `Prices with taxes` sur `Prices wo taxes` ;
9. trie les dates de la plus recente a la plus ancienne.

Le choix du LEFT JOIN est volontaire : un retard de publication du prix avec taxes ne doit pas supprimer une observation hors taxes disponible, qui est la variable cle pour les modeles hebdomadaires.

### 4.4 Garde-fous WOB

- un seul endpoint historique attendu ;
- six feuilles obligatoires ;
- pas de typage automatique Excel ;
- aucun offset de lignes fixe ;
- trois libelles attendus sur une unique ligne d'en-tete ;
- colonne de dates detectee par donnees valides ;
- dates >= 2005 ;
- aucune date dupliquee ;
- au moins un prix hors taxes sur chaque date conservee ;
- `NoTax LEFT JOIN WithTax` pour proteger le ragged edge.

### 4.5 Performance

La version robuste actuelle score les colonnes sur un echantillon de 200 lignes pour trouver la date. C'est plus lent qu'un reader fonde sur une position fixe, mais beaucoup plus robuste a une modification de mise en page.

Ordre de grandeur pratique :

- quelques dizaines de secondes : acceptable pour un workbook web ;
- plusieurs minutes : investiguer la query ;
- si besoin, reduire `ScoreSampleSize` apres validation, ou figer la premiere colonne comme date avec un garde-fou explicite.

### 4.6 Erreurs WOB et diagnostic

| Erreur / symptome | Cause probable | Ou verifier |
|---|---|---|
| `Formula.Firewall` | Une URL dynamique ou une combinaison de sources a ete reintroduite | `EC_WOB_RAW_FULL`; garder `Host` + `RelativePath` statiques; Data Source Settings |
| `Excel.Workbook` incomprehensible | Endpoint DG ENER change, reponse HTML/PDF au lieu du XLSX | Page officielle WOB puis `FilePath` |
| `Missing sheet(s)` | Structure du workbook modifiee | Liste des feuilles dans le workbook officiel |
| Header introuvable | Libelle de colonne renomme | `EC_WOB_RAW_FULL`, feuille concernee, ligne d'en-tete |
| Aucune colonne de dates | Layout change ou parsing de date incompatible | `EC_WOB_RAW_FULL` et `ScoreSampleSize` |
| Dates dupliquees | Source modifiee ou plusieurs blocs de donnees | Inspecter les dates signalees dans la feuille raw |
| Refresh tres lent | Scoring de colonnes + workbook web | `EC_WOB_VIEW`; verifier les buffers et le sample de 200 lignes |

## 5. Eurostat - prix household gaz et electricite

### 5.1 Architecture

```text
Eurostat API / tables
       -> EUROSTAT_ENERGY_RAW
       -> EUROSTAT_ENERGY_VIEW
       -> onglet Eurostat
```

La query raw expose les dimensions du contrat (`dataset`, `series`, `freq`, `unit`, `band_dimension`, `band`, `currency`, `geo`, `tax`, `date`, etc.). La vue finale ne doit pas choisir silencieusement un autre contrat.

### 5.2 Contrats figes et verifies le 2026-08-07

| Dataset | Serie | Freq | Unit | Dimension bande | Bande | Currency | Geo |
|---|---|---|---|---|---|---|---|
| `nrg_pc_202` | gas household | S | KWH | `nrg_cons` | `GJ20-199` | EUR | EA |
| `nrg_pc_204` | electricity household | S | KWH | `nrg_cons` | `KWH2500-4999` | EUR | EA |
| `nrg_pc_202_h` | gas household historique | S | GJ_GCV | `consom` | `4141100` | EUR | EA |
| `nrg_pc_204_h` | electricity household historique | S | KWH | `consom` | `4161150` | EUR | EA |

**Regle :** ne jamais modifier automatiquement `ExpectedContracts` pour faire passer un refresh. Un changement doit etre investigue comme un changement de contrat amont.

### 5.3 Construction de `EUROSTAT_ENERGY_VIEW`

1. validation des quatre contrats ;
2. selection des codes fiscaux `I_TAX`, `X_TAX`, `X_VAT` ;
3. priorite a la table courante sur la table `_h` dans la zone de recouvrement ;
4. unicite `series x date x tax` ;
5. pivot des codes fiscaux ;
6. calcul :

```text
WTAX = I_TAX
NTAX = X_TAX
VAT  = 100 * (I_TAX / X_VAT - 1)
EXC  = X_VAT - X_TAX
```

7. dates semi-annuelles uniquement en janvier/juillet ;
8. controle de completude ;
9. dernier semestre incomplet autorise comme ragged edge ;
10. semestre incomplet **interieur** interdit ;
11. sortie finale de huit series, date recente en premier.

### 5.4 Pourquoi autoriser le dernier semestre incomplet

`X_VAT` peut etre publie avec retard. Bloquer tout semestre incomplet ferait echouer un refresh normal au bord droit. En revanche, un semestre incomplet suivi plus tard d'un semestre complet est un vrai trou historique et doit lever une erreur.

### 5.5 Risques Eurostat

- passage `EA` -> `EA21`, `EA20`, etc. ;
- changement de bande de consommation ;
- changement d'unite ;
- revision retrospective des series ;
- raccord entre tables historiques et courantes ;
- absence temporaire de `X_VAT` ;
- changement de dimensions (`consom` / `nrg_cons`).

Le risque le plus dangereux est le **changement silencieux de contrat**, car il pourrait reconstruire toute l'histoire de `gas_pre_tax` ou `electricity_pre_tax` sans erreur apparente. C'est pour cette raison que les quatre contrats sont figes.

### 5.6 Erreurs Eurostat et diagnostic

| Erreur | Action |
|---|---|
| `EUROSTAT CONTRACT CHANGED` | Ouvrir `EUROSTAT_ENERGY_RAW`; faire un distinct sur les dimensions de contrat; comparer au contrat documente |
| Dataset absent | Verifier le refresh de la raw query / endpoint Eurostat |
| Plusieurs contrats | Ne pas choisir automatiquement; comprendre pourquoi plusieurs lignes satisfont les filtres |
| `X_VAT` absent au dernier semestre | Ragged edge normal si aucun semestre complet ne vient apres |
| Semestre fiscal interieur incomplet | Inspecter `I_TAX`, `X_TAX`, `X_VAT` a la date signalee |
| Date hors janvier/juillet | Changement de convention temporelle ou parsing incorrect |
| Rupture au raccord historique/courant | Comparer les deux contrats et la conversion `GJ_GCV -> EUR/kWh` sur leur overlap |

## 6. World Bank - snapshot statique

### 6.1 Decision d'architecture

La resolution automatique du lien Pink Sheet a ete retiree du chemin de production. Plusieurs comportements instables de l'API/search World Bank rendaient la query disproportionnee par rapport a son role economique.

Le World Bank sert essentiellement au **backcast pre-TTF**. Une fois TTF disponible, le builder utilise TTF, pas le proxy World Bank pour la partie courante.

Architecture retenue :

```text
World_Bank sheet = snapshot values-only
WORLD_BANK_GAS_RAW = maintenance seulement, si conserve
Refresh All = OFF pour WORLD_BANK_GAS_RAW
```

Le snapshot actuel contient `Natural gas, Europe`, en USD/MMBtu, avec une longue histoire mensuelle.

### 6.2 Mise a jour World Bank

Mise a jour rare / manuelle :

1. aller sur la page Commodity Markets / Pink Sheet de la Banque mondiale ;
2. telecharger `CMO-Historical-Data-Monthly.xlsx` ;
3. verifier la feuille `Monthly Prices` ;
4. extraire `Natural gas, Europe` en `($/mmbtu)` ;
5. remplacer le contenu de `World_Bank` par les valeurs ;
6. conserver le nom de colonne `wb_natural_gas_europe_usd_mmbtu` ;
7. mettre a jour `Metadata` avec date/source.

### 6.3 Risques World Bank

- revision historique du proxy ;
- oubli de mise a jour du snapshot ;
- changement de definition de `Natural gas, Europe` ;
- reactivation accidentelle de la query dynamique dans `Refresh All`.

Dans le workflow actuel, ces risques sont beaucoup moins critiques que les risques Eurostat, car le proxy sert principalement a une portion historique du backcast.

## 7. Bloomberg - live vs snapshot

Bloomberg n'est pas une Power Query.

```text
Bloomberg_live
    -> formules Bloomberg / PX_LAST
    -> necessite Bloomberg connecte

Bloomberg_copy
    -> copy / paste values
    -> source lue par Python
```

Le builder ne doit jamais dependre de `Bloomberg_live`. Les `#NAME?` ou `#N/A` dans la feuille live lorsque Bloomberg n'est pas connecte sont donc acceptables.

### Procedure de mise a jour

1. ouvrir le workbook avec Bloomberg disponible ;
2. rafraichir / attendre la resolution des formules dans `Bloomberg_live` ;
3. verifier les dates les plus recentes ;
4. copier la zone complete ;
5. `Paste Special -> Values` dans `Bloomberg_copy` ;
6. verifier qu'il ne reste aucune formule dans `Bloomberg_copy` ;
7. sauvegarder le workbook avant de lancer Python.

### Garde-fous a conserver

- `Bloomberg_copy` prioritaire dans le builder ;
- `Bloomberg_live` ignore ;
- dates textuelles non valides et erreurs Excel converties en manquants ;
- les autres series de la meme date sont conservees si une serie est manquante ;
- Python retrie les dates chronologiquement meme si Excel affiche recente -> ancienne.

### Risques

- snapshot oublie / stale ;
- mauvais collage de plage ;
- changement de ticker ou d'unite ;
- formule live non resolue au moment de la copie ;
- ajout d'une colonne sans mise a jour du contrat Python.

## 8. Haver

A ce stade, `Haver` est un snapshot dans le workbook, pas une Power Query web. Le reader historique du projet attendait les HICP energy et `PPI Energy`; le ticker PPI Energy documente est `H025PP@G10`.

Tant que l'acces Haver direct depuis ce workbook n'est pas stabilise, la regle est simple : **mettre a jour la feuille sans changer sa structure**, puis sauvegarder avant le builder.

Risques principaux : ticker renomme, historique revise, colonne de date deplacee, nouvelle serie ajoutee sans mise a jour du builder.

## 9. Metadata - ce qu'il faut tracer

La feuille `Metadata` doit servir d'audit leger. Elle ne doit pas etre injectee comme variable numerique dans les panels.

Champs utiles :

```text
raw_workbook_policy          single_workbook_updated_in_place
wob_endpoint_verified       2026-08-07
wob_endpoint_path           document/download/906e60ca-8b6a-44e7-8589-652854d2fd3f_en
world_bank_policy           static_values_only_snapshot
world_bank_last_refresh     <date>
world_bank_series           Natural gas, Europe
world_bank_unit             USD/MMBtu
eurostat_contract_gas       nrg_pc_202 / EA / GJ20-199 / KWH / EUR
eurostat_contract_elec      nrg_pc_204 / EA / KWH2500-4999 / KWH / EUR
eurostat_contract_gas_h     nrg_pc_202_h / EA / 4141100 / GJ_GCV / EUR
eurostat_contract_elec_h    nrg_pc_204_h / EA / 4161150 / KWH / EUR
bloomberg_copy_updated      <date/heure>
haver_updated               <date>
```

## 10. Workflow normal de mise a jour

### Etape A - ouvrir Excel

Ouvrir :

```text
data/raw/bvar_energy_raw_data.xlsx
```

### Etape B - Power Query

`Data -> Refresh All`

Doivent etre rafraichis :

```text
EC_WOB_RAW_FULL -> EC_WOB_VIEW
EUROSTAT_ENERGY_RAW -> EUROSTAT_ENERGY_VIEW
```

`WORLD_BANK_GAS_RAW` ne doit pas etre une dependance de `Refresh All`.

### Etape C - Bloomberg

```text
Bloomberg_live refresh
      -> verifier les dernieres dates
      -> copy
      -> paste values vers Bloomberg_copy
```

### Etape D - Haver

Mettre a jour le snapshot Haver lorsque la source est disponible.

### Etape E - controles visuels rapides

Verifier :

- European Commission : date la plus recente en haut ;
- Eurostat : dernier semestre coherent ;
- Bloomberg_copy : date la plus recente + valeurs numeriques ;
- World_Bank : table toujours presente ;
- Haver : colonnes/tickers attendus ;
- aucune query en erreur dans `Queries & Connections`.

### Etape F - sauvegarder

**Sauvegarder le workbook avant Python.**

### Etape G - builder

Depuis la racine du projet :

```powershell
python src/data_pipeline/build_dataset_paper_six_models_v9.py
```

Si le dossier processed du jour existe deja et doit volontairement etre remplace :

```powershell
python src/data_pipeline/build_dataset_paper_six_models_v9.py --overwrite
```

Les sorties vont dans :

```text
data/processed/YYYYMMDD/
```

## 11. Decision tree en cas d'erreur

```text
REFRESH ALL ECHOUE
|
+-- erreur EC_WOB_RAW_FULL ?
|   +-- Formula.Firewall -> verifier Web.Contents statique + privacy settings
|   +-- Excel.Workbook -> verifier endpoint sur la page DG ENER
|   +-- missing sheets -> comparer structure du workbook officiel
|
+-- erreur EC_WOB_VIEW ?
|   +-- header -> ouvrir EC_WOB_RAW_FULL et inspecter la feuille de prix
|   +-- date -> verifier la colonne contenant les dates
|   +-- duplicate dates -> inspecter les dates signalees
|
+-- erreur EUROSTAT_ENERGY_RAW ?
|   +-- endpoint/dataset -> verifier Eurostat
|
+-- erreur EUROSTAT_ENERGY_VIEW ?
|   +-- CONTRACT CHANGED -> NE PAS modifier ExpectedContracts au hasard
|   +-- verifier distinct dataset/geo/unit/band/currency/freq
|   +-- incomplete interior semester -> inspecter I_TAX/X_TAX/X_VAT
|
+-- erreur WORLD_BANK_GAS_RAW ?
    +-- elle ne doit pas bloquer Refresh All
    +-- verifier qu'elle est Connection only + Refresh All OFF
```

Puis, si le refresh Excel passe mais le builder Python echoue :

```text
BUILDER ECHOUE
|
+-- missing sheet -> verifier noms exacts des onglets
+-- Bloomberg missing/stale -> verifier Bloomberg_copy
+-- Eurostat missing tax -> verifier l'onglet Eurostat et sa query
+-- WOB missing target -> verifier European_Commission
+-- processed folder exists -> --overwrite seulement si voulu
```

## 12. Alternatives si une source devient instable

### WOB

**Actuel :** Power Query sur endpoint historique statique.  
**Alternative :** telecharger manuellement le workbook et garder un snapshot values-only, ou revenir a un fetch Python dedie.

### Eurostat

**Actuel :** Power Query + contrat fige.  
**Alternative :** extraction SDMX Python avec snapshot de raw TSV et contrat ecrit dans un manifest.

### World Bank

**Actuel :** snapshot statique. C'est deja l'alternative la plus robuste pour son role actuel.  
**Alternative dynamique :** query manuelle hors `Refresh All`, uniquement pour maintenance.

### Bloomberg

**Actuel :** live + copy values.  
**Alternative :** automatiser le copy/paste via VBA/Office Script, mais conserver une zone values-only comme interface Python.

### Haver

**Actuel :** snapshot.  
**Alternative :** extraction API/directe lorsque l'acces est stabilise, puis ecriture d'un snapshot dans la meme feuille.

## 13. Regles a ne pas casser

1. Le workbook raw reste **unique** et est mis a jour en place.
2. Les vintages sont les dossiers `processed/YYYYMMDD`, pas des copies datees du workbook raw.
3. Ne jamais charger `EC_WOB_RAW_FULL` dans une feuille si cela cree des volumes inutiles.
4. Ne jamais faire dependre Python de `Bloomberg_live`.
5. Ne jamais modifier un contrat Eurostat automatiquement pour faire passer un refresh.
6. Un trou terminal peut etre ragged ; un trou interieur doit etre investigue.
7. Ne pas reintroduire une URL web dynamique dans WOB sans reevaluer le risque `Formula.Firewall`.
8. `World_Bank` reste un snapshot values-only tant qu'une mise a jour dynamique n'apporte pas de valeur economique suffisante.
9. Sauvegarder Excel apres les refresh/copies et avant le builder.
10. En cas d'erreur, diagnostiquer d'abord la **source ou le contrat amont**, pas le BVAR.

## 14. Points de verification prioritaires

| Source | Check rapide avant build |
|---|---|
| European Commission | derniere date, 3 NTAX presents, aucune query WOB en erreur |
| Eurostat | 4 contrats inchanges, pas de trou fiscal interieur |
| World Bank | feuille values-only presente, colonne et unite intactes |
| Bloomberg | `Bloomberg_copy` recent et sans formules |
| Haver | tickers/structure inchanges, date recente raisonnable |
| Metadata | dates de mise a jour et contrats documentes |

---

### Reference operationnelle WOB

Si l'endpoint historique cesse de fonctionner, ne pas deviner un nouveau GUID. Aller d'abord sur :

`https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en`

Identifier le lien **Price developments 2005 onwards / historical prices**, verifier qu'il pointe vers le workbook attendu, puis mettre a jour `FilePath` seulement apres verification du contenu et des six feuilles obligatoires.
