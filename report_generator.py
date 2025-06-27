# report_generator.py
import io
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
import streamlit as st

class PDFReportGenerator:
    """
    수요 예측 결과를 PDF 리포트로 생성하는 클래스
    [수정] 가독성 높은 리포트를 위해 AI 해설 텍스트 파싱 및 서식 적용 기능 강화
    """
    
    def __init__(self, font_path: Optional[str] = None):
        """PDF 리포트 생성기 초기화"""
        self.font_path = font_path
        self._register_fonts()
        self.styles = self._create_styles()
        
    def _register_fonts(self):
        """한글 폰트 등록"""
        if self.font_path and os.path.exists(self.font_path):
            try:
                pdfmetrics.registerFont(TTFont('Korean', self.font_path))
                self.font_name = 'Korean'
            except Exception:
                self.font_name = 'Helvetica'
        else:
            self.font_name = 'Helvetica'
            
    def _create_styles(self):
        """PDF 스타일 정의"""
        styles = getSampleStyleSheet()
        
        styles.add(ParagraphStyle(
            name='CustomTitle', parent=styles['Title'], fontName=self.font_name,
            fontSize=24, textColor=colors.HexColor('#1e3c72'), spaceAfter=30, alignment=TA_CENTER
        ))
        
        styles.add(ParagraphStyle(
            name='CustomH1', parent=styles['h1'], fontName=self.font_name,
            fontSize=18, textColor=colors.HexColor('#1e3c72'), spaceBefore=12, spaceAfter=12, alignment=TA_LEFT,
            borderPadding=5
        ))
        
        styles.add(ParagraphStyle(
            name='CustomH2', parent=styles['h2'], fontName=self.font_name,
            fontSize=14, textColor=colors.HexColor('#2a5298'), spaceBefore=10, spaceAfter=8, alignment=TA_LEFT,
            borderPadding=5
        ))
        
        styles.add(ParagraphStyle(
            name='CustomBody', parent=styles['BodyText'], fontName=self.font_name,
            fontSize=10, leading=16, textColor=colors.HexColor('#333333')
        ))

        styles.add(ParagraphStyle(
            name='CustomBullet', parent=styles['BodyText'], fontName=self.font_name,
            fontSize=10, leading=16, textColor=colors.HexColor('#333333'),
            leftIndent=20, bulletIndent=10
        ))
        
        return styles
    
    def _create_header_footer(self, canvas_obj, doc):
        """페이지 헤더 및 푸터 생성"""
        canvas_obj.saveState()
        # Header
        canvas_obj.setFillColor(colors.HexColor('#1e3c72'))
        canvas_obj.rect(0, A4[1] - 60, A4[0], 60, fill=True, stroke=False)
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFont(self.font_name, 18)
        canvas_obj.drawCentredString(A4[0]/2, A4[1] - 35, "SmartDemand AI 수요예측 리포트")
        # Footer
        canvas_obj.setStrokeColor(colors.HexColor('#e0e0e0'))
        canvas_obj.line(30, 40, A4[0] - 30, 40)
        canvas_obj.setFillColor(colors.HexColor('#666666'))
        canvas_obj.setFont(self.font_name, 9)
        canvas_obj.drawString(30, 25, f"생성일: {datetime.now().strftime('%Y-%m-%d')}")
        canvas_obj.drawRightString(A4[0] - 30, 25, f"Page {doc.page}")
        canvas_obj.restoreState()
    
    def _create_summary_table(self, summary_data: Dict[str, Any]) -> Table:
        """요약 정보 테이블 생성"""
        data = [
            ['분석 기간', summary_data.get('period', 'N/A')],
            ['분석 품목 수', f"{summary_data.get('product_count', 0)}개"],
            ['평균 예측 정확도', f"{summary_data.get('avg_accuracy', 0):.1f}%"],
            ['총 7일 예상 판매량', f"{summary_data.get('total_forecast', 0):,.0f}개"]
        ]
        table = Table(data, colWidths=[2.5*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f2f2f2')),
            ('FONTNAME', (0, 0), (-1, -1), self.font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dddddd')),
            ('PADDING', (0,0), (-1,-1), 12),
        ]))
        return table
    
    def _parse_and_style_explanation(self, text: str) -> List:
        """
        [핵심 수정] AI가 생성한 마크다운 형식의 텍스트를 파싱하고, 깨진 문자를 교정합니다.
        """
        elements = []
        if not text:
            return elements

        # 1. 텍스트 후처리: 깨진 문자(₩)나 불필요한 기호 제거
        text = text.replace('₩~', '~').replace('₩', '')
        text = re.sub(r'\s*`\s*', "'", text) # backticks to single quotes
        
        # 2. **text** -> <b>text</b> 로 변환하여 굵은 글씨 지원
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # 3. 텍스트를 논리적 블록으로 분할 (여러 줄의 개행 처리)
        paragraphs = re.split(r'\n\s*\n', text.strip())
        
        for p_block in paragraphs:
            p_strip = p_block.strip()
            if not p_strip: continue

            # 4. 마크다운 문법에 따라 스타일 적용
            if p_strip.startswith('###'):
                content = p_strip.replace('###', '').strip()
                elements.append(Paragraph(content, self.styles['CustomH2']))
                elements.append(Spacer(1, 0.1*inch))
            elif p_strip.startswith(('* ', '• ')):
                # 여러 줄로 된 목록 항목 처리
                lines = p_strip.split('\n')
                for line in lines:
                    line_strip = line.strip()
                    if line_strip.startswith(('* ', '• ')):
                        content = re.sub(r'^[\*•]\s*', '', line_strip)
                        elements.append(Paragraph(content, self.styles['CustomBullet'], bulletText='•'))
                    else: # 목록 항목의 두 번째 줄 이상
                        elements.append(Paragraph(line_strip, self.styles['CustomBullet']))
                elements.append(Spacer(1, 0.1*inch))
            else: # 일반 텍스트 단락
                elements.append(Paragraph(p_strip, self.styles['CustomBody']))
                elements.append(Spacer(1, 0.1*inch))
        
        return elements

    def generate_report(
        self,
        summary_data: Dict[str, Any],
        products_data: Dict[str, Dict],
        explanations: Optional[Dict[str, Dict[str, str]]] = None
    ) -> bytes:
        """전체 PDF 리포트 생성"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4, topMargin=80, bottomMargin=60,
            leftMargin=inch, rightMargin=inch
        )
        story = []
        
        # 1. 표지
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("SmartDemand AI", self.styles['CustomTitle']))
        story.append(Paragraph("수요 예측 및 경영 분석 리포트", self.styles['CustomH1']))
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph(f"<b>리포트 생성일:</b> {datetime.now().strftime('%Y년 %m월 %d일')}", self.styles['CustomBody']))
        story.append(PageBreak())
        
        # 2. 요약 페이지
        story.append(Paragraph("전체 요약", self.styles['CustomH1']))
        story.append(self._create_summary_table(summary_data))
        story.append(PageBreak())
        
        # 3. 품목별 상세 분석
        story.append(Paragraph("품목별 상세 분석", self.styles['CustomH1']))
        story.append(Spacer(1, 0.2*inch))
        
        for product_name, product_data in products_data.items():
            product_elements = []
            
            product_elements.append(Paragraph(f"📦 {product_name}", self.styles['CustomH1']))
            
            perf_data = [
                [Paragraph('<b>예측 정확도</b>', self.styles['CustomBody']), f"{product_data.get('accuracy', 0):.1f}%"],
                [Paragraph('<b>평균 오차(MAE)</b>', self.styles['CustomBody']), f"약 {product_data.get('mae', 0):.1f}개"],
                [Paragraph('<b>7일 예상 판매량</b>', self.styles['CustomBody']), f"{product_data.get('forecast_7day', 0):,.0f}개"]
            ]
            perf_table = Table(perf_data, colWidths=[2.5*inch, 3*inch])
            perf_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), self.font_name),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]))
            product_elements.append(perf_table)
            product_elements.append(Spacer(1, 0.3*inch))
            
            if explanations and product_name in explanations:
                explanation_text = explanations[product_name].get('detail', '')
                if explanation_text:
                    styled_explanation = self._parse_and_style_explanation(explanation_text)
                    product_elements.extend(styled_explanation)
            
            story.append(KeepTogether(product_elements))
            story.append(PageBreak())
        
        doc.build(story, onFirstPage=self._create_header_footer, onLaterPages=self._create_header_footer)
        
        pdf_data = buffer.getvalue()
        buffer.close()
        return pdf_data

def create_downloadable_report(
    summary_data: Dict[str, Any],
    products_data: Dict[str, Dict],
    explanations: Optional[Dict[str, Dict[str, str]]] = None,
    font_path: Optional[str] = None
) -> bytes:
    try:
        generator = PDFReportGenerator(font_path)
        return generator.generate_report(summary_data, products_data, explanations)
    except Exception as e:
        st.error(f"PDF 생성 중 오류 발생: {str(e)}")
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer)
        p.drawString(100, 750, "PDF Report Generation Error")
        p.drawString(100, 700, f"Error: {str(e)}")
        p.showPage()
        p.save()
        pdf_data = buffer.getvalue()
        buffer.close()
        return pdf_data
