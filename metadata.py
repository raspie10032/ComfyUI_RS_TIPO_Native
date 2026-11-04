"""Optional, validated TIPO v2.1 conditioning fields (not image prompt tags)."""

UNSPECIFIED = "unspecified"
RATINGS = (UNSPECIFIED, "safe", "sensitive", "nsfw", "explicit")
QUALITIES = (UNSPECIFIED, "masterpiece", "best quality", "great quality",
             "good quality", "normal quality", "low quality", "worst quality")
ERAS = (UNSPECIFIED, "newest", "recent", "mid", "early", "old")


def metadata_fields(*, rating=UNSPECIFIED, quality=UNSPECIFIED, quality_era=UNSPECIFIED):
    for name, value, allowed in (("rating", rating, RATINGS), ("quality", quality, QUALITIES),
                                 ("quality_era", quality_era, ERAS)):
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(f"TIPO {name}: select one of {', '.join(allowed)}. / 목록에서 선택해 주세요.")
    if quality == UNSPECIFIED and quality_era != UNSPECIFIED:
        raise ValueError("TIPO: select quality before quality_era. / 시대를 지정하려면 품질도 선택해 주세요.")
    fields = {}
    if quality != UNSPECIFIED:
        fields["quality"] = quality + (f", {quality_era}" if quality_era != UNSPECIFIED else "")
    if rating != UNSPECIFIED:
        # Official Danbooru explicit -> TIPO mapping uses both labels.
        fields["rating"] = "nsfw, explicit" if rating == "explicit" else rating
    return fields


def metadata_header(**settings):
    return "".join(f"{name}: {value}\n" for name, value in metadata_fields(**settings).items())


def metadata_inputs():
    return {
        "rating": (list(RATINGS), {"default": UNSPECIFIED,
            "tooltip": "TIPO content-rating guidance for all outputs. Unspecified omits the field; this is not a tag filter or guarantee. / 등급 유도. 미지정은 기존 동작을 유지합니다."}),
        "quality": (list(QUALITIES), {"default": UNSPECIFIED,
            "tooltip": "TIPO quality metadata, not image model quality tags. Unspecified omits the field. / TIPO 품질 조건이며 이미지 프롬프트에 추가하지 않습니다."}),
        "quality_era": (list(ERAS), {"default": UNSPECIFIED,
            "tooltip": "Optional quality era; select quality first. newest: 2024+, recent: 2020–2024, mid: 2018–2020, early: 2015–2018, old: before 2015. / 품질과 함께 쓰는 시대 조건."}),
    }
