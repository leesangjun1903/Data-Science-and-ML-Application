# Google Landmarks Dataset v2 A Large-Scale Benchmark for Instance-Level Recognition and Retrieval

## 1. 핵심 주장 및 주요 기여  
Google Landmarks Dataset v2(GLDv2)는  
- **세계 최대 규모**(5백만 장 이상의 이미지, 20만 개 이상의 랜드마크) 이상의 인스턴스 인식·검색 벤치마크를 제공,  
- **실제 응용 시나리오**를 반영하여  
  - 극단적 클래스 불균형(롱테일 분포),  
  - 도메인 외(landmark 아님) 쿼리 99% 혼합,  
  - 높은 인트라 클래스 변이(다양한 각도·조명·계절·매체)  
  등의 현실적 난제를 포함함으로써  
기존 소규모·단일 도시 중심 데이터셋의 한계를 극복한다.

## 2. 해결하려는 문제  
- *인스턴스 인식·검색 연구의 평가 척도가 지나치게 단순·제한적*  
- 기존 Oxford/Paris/Kaggle 공개 데이터셋은  
  - 11~125개 landmark, 수천~수만 장 이미지에 불과  
  - 테스트 쿼리는 대부분 온도메인, false-positive 평가 미흡  
  - 현실적 클래스 롱테일 및 도메인 외 쿼리 반영 부족  

## 3. 데이터셋 구성 및 특징  
- **트레인**: 4.1M 이미지, 203k 클래스  
- **인덱스**: 0.76M 이미지, 101k 클래스  
- **쿼리**: 0.12M 이미지(1.1% 온도메인, 98.9% 오도메인)  
- **장애물**:  
  - 클래스당 이미지 수가 1~수십 장에 분포하는 롱테일  
  - 크롭되지 않은 전체 이미지 쿼리 → 검색 강인성 요구  
  - 잘라낸 small patch 인스턴스는 일부만 공통 영역  

## 4. 평가 지표  
- **Recognition**: Global Average Precision (GAP, µAP)  
  
$$
    \mu AP = \frac{1}{M}\sum_{i=1}^N P(i)\,\mathrm{rel}(i)
  $$  

- **Retrieval**: mAP@100  
  
$$
    mAP@100 = \frac{1}{Q}\sum_{q=1}^Q \frac{1}{\min(m_q,100)}\sum_{k=1}^{\min(n_q,100)}P_q(k)\,\mathrm{rel}_q(k)
  $$

## 5. 데이터 전처리 및 모델링  
1. **클래스 정제**  
   - DELF(글로벌 ResNet-101 + 지역 DELF) 매칭 기반 그래프 클러스터링  
   - 이미지 간 30 inlier 이상 유지 → GLDv2-train-clean (1.6M 이미지, 81k 클래스)  
2. **기본 모델**  
   - 글로벌 임베딩: ResNet-101 + GeM 풀링 + ArcFace 손실  
   - 지역 특징: DELF-ASMK⋆ + spatial verification (SP)  
3. **성능**  
   - GLDv2-train-clean 학습 후, Revisited Oxford/Paris 전이 학습 시 mAP 87.3%/76.2%(Medium) → GLDv1 대비 +10%p↑  
   - Retrieval GLDv2 자체 벤치마크에서 글로벌 임베딩 단일 모델로도 최고 성능 달성  
   - 인식(µAP)에서도 DELF-KD-tree, DELG(Global+SP) 등이 우위  

## 6. 한계 및 개선 방향  
- **얼굴·비랜드마크 false-positive**: 오도메인 쿼리 처리 문턱 설정 연구 필요  
- **세밀한 이미지 레벨 GT 부재**: 클래스 단위 ground truth → 일부 hard-negative/partial-overlap 오차  
- **쿼리–인덱스 뷰 차이**: 소규모 디테일 매칭 성능 저하  
- **라벨 노이즈**: 크라우드소싱·자동 매칭 라벨 오류 여전  

## 7. 일반화 및 향후 연구 고려사항  
- **Transfer Learning**: GLDv2 임베딩을 로고·상품·예술품 인식 등 인스턴스 레벨 과제로 확장 가능  
- **롱테일 클래스 대응**: 소수 샘플 학습(메타러닝, 제로샷) 연구  
- **도메인 외 판단**: false-positive 억제용 불확실도 추정·분류기 동시 학습  
- **Multi-modality**: 텍스트·지리정보 결합으로 라벨링·검색 정교화  
- **효율성**: 대규모 인덱스에서 실시간 검색을 위한 양자화·인덱싱 기법  

이 데이터셋은 인스턴스 레벨 인식·검색 연구의 스케일과 현실적 난제를 한 차원 높였으며, 롱테일·오도메인·높은 변이성 문제 해결을 위한 새로운 모델 개발과 평가 기준을 제시해 향후 커뮤니티 전반에 걸친 발전을 견인할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3ea9d6da-65bc-4ad1-af94-273766d51b0e/2004.01804v2.pdf
