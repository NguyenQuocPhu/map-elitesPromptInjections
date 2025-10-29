from typing import TypeVar, Generic, List, Dict, Tuple, Optional, Union, Any
import json
import ast
import numpy as np

# --- Bổ sung cho hàm visualize (Optional Dependencies) ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    _VIS_LIBS_INSTALLED = True
except ImportError:
    _VIS_LIBS_INSTALLED = False
# ----------------------------------------------------


T = TypeVar("T")


class Archive(Generic[T]):
    """
    Một lớp archive linh hoạt để lưu trữ và thao tác với các bộ sưu tập dữ liệu,
    đặc biệt hữu ích cho các thuật toán như Map-Elites.

    Lớp này hỗ trợ các thao tác như thêm, cập nhật,
    truy xuất và phân tích các bộ sưu tập dữ liệu.
    """

    def __init__(self, name: str):
        """
        Khởi tạo một instance Archive.

        Args:
            name (str): Tên của archive
        """
        self.name = name
        # THAY ĐỔI: Key giờ là Tuple[float, ...]
        self._archive: Dict[Tuple[float, ...], List[T]] = {}

    def add(self, key: Tuple[float, ...], value: List[T]) -> None:
        """
        Thêm một cặp key-value mới vào archive.

        Args:
            key (Tuple[float, ...]): Key duy nhất cho mục nhập, phải là tuple của các số.
            value (List[T]): Danh sách các giá trị cần lưu trữ

        Raises:
            TypeError: Nếu key không phải là tuple hoặc chứa phần tử không phải số.
        """
        if not isinstance(key, tuple):
            raise TypeError("Key phải là một tuple")
        
        # BỔ SUNG: Kiểm tra các phần tử của key phải là số
        if not all(isinstance(k, (int, float)) for k in key):
            raise TypeError("Tất cả các phần tử trong key tuple phải là số (int hoặc float)")

        if not isinstance(value, list):
            value = list(value)
        self._archive[key] = value

    def update(self, key: Tuple[float, ...], value: List[T]) -> None:
        """
        Cập nhật giá trị cho một key đã tồn tại.

        Args:
            key (Tuple[float, ...]): Key để cập nhật
            value (List[T]): Giá trị mới để thiết lập

        Raises:
            KeyError: Nếu key không tồn tại
        """
        if key not in self._archive:
            raise KeyError(f"Key {key} không tồn tại trong archive")

        if not isinstance(value, list):
            value = list(value)
        self._archive[key] = value

    def delete(self, key: Tuple[float, ...]) -> None:
        """
        Xóa một cặp key-value khỏi archive.

        Args:
            key (Tuple[float, ...]): Key để xóa
        """
        self._archive.pop(key, None)

    def flatten_values(self) -> List[T]:
        """
        Trả về một danh sách phẳng của tất cả các giá trị trong archive.

        Returns:
            List[T]: Danh sách phẳng của tất cả các giá trị
        """
        return [item for sublist in self._archive.values() for item in sublist]

    def get(self, key: Tuple[float, ...]) -> Optional[List[T]]:
        """
        Truy xuất giá trị cho một key nhất định.

        Args:
            key (Tuple[float, ...]): Key để truy xuất

        Returns:
            Optional[List[T]]: Các giá trị liên quan đến key, hoặc None
        """
        return self._archive.get(key)

    def keys(self) -> List[Tuple[float, ...]]:
        """
        Trả về một danh sách tất cả các key trong archive.

        Returns:
            List[Tuple[float, ...]]: Danh sách các key
        """
        return list(self._archive.keys())

    def exists(self, key: Tuple[float, ...]) -> bool:
        """
        Kiểm tra xem một key có tồn tại trong archive hay không.

        Args:
            key (Tuple[float, ...]): Key để kiểm tra

        Returns:
            bool: Key có tồn tại hay không
        """
        return key in self._archive

    def extend(self, key: Tuple[float, ...], new_values: List[T]) -> None:
        """
        Mở rộng danh sách giá trị cho một key đã tồn tại.

        Args:
            key (Tuple[float, ...]): Key để mở rộng
            new_values (List[T]): Các giá trị cần thêm

        Raises:
            KeyError: Nếu key không tồn tại
        """
        if key not in self._archive:
            raise KeyError(f"Key {key} không tồn tại trong archive")
        self._archive[key].extend(new_values)

    def values_are_numeric(self) -> bool:
        """
        Kiểm tra xem tất cả các giá trị trong archive có phải là số hay không.

        Returns:
            bool: Tất cả các giá trị có phải là số hay không
        """
        flat_values = self.flatten_values()
        if not flat_values:
            return True  # Archive rỗng được coi là "numeric"
        return all(isinstance(v, (int, float)) for v in flat_values)

    def len_elements(self) -> Dict[Tuple[float, ...], int]:
        """
        Trả về một dictionary về độ dài của các key.

        Returns:
            Dict[Tuple[float, ...], int]: Độ dài của danh sách giá trị của mỗi key
        """
        return {k: len(v) for k, v in self._archive.items()}

    def median_elements(self) -> Dict[Tuple[float, ...], float]:
        """
        Tính giá trị trung vị cho các giá trị số của mỗi key.

        Returns:
            Dict[Tuple[float, ...], float]: Trung vị của các giá trị của mỗi key

        Raises:
            ValueError: Nếu các giá trị không phải là số
        """
        if not self.values_are_numeric():
            raise ValueError("Các giá trị trong archive phải là số")

        return {k: float(np.median(v)) for k, v in self._archive.items() if v}

    def idx_median_elements(self) -> Dict[Tuple[float, ...], int]:
        """
        Tìm chỉ số của giá trị trung vị cho các giá trị số của mỗi key.

        Returns:
            Dict[Tuple[float, ...], int]: Chỉ số của trung vị cho mỗi key

        Raises:
            ValueError: Nếu các giá trị không phải là số
        """
        if not self.values_are_numeric():
            raise ValueError("Các giá trị trong archive phải là số")

        return {k: int(np.argsort(v)[len(v) // 2]) for k, v in self._archive.items() if v}

    def idx_max_elements(self, seed: int = 0) -> Dict[Tuple[float, ...], int]:
        """
        Tìm chỉ số của giá trị lớn nhất cho mỗi key, với việc xử lý ngẫu nhiên khi có nhiều giá trị max.

        Args:
            seed (int, optional): Hạt giống ngẫu nhiên. Mặc định là 0.

        Returns:
            Dict[Tuple[float, ...], int]: Chỉ số của max cho mỗi key

        Raises:
            ValueError: Nếu các giá trị không phải là số
        """
        if not self.values_are_numeric():
            raise ValueError("Các giá trị trong archive phải là số")

        np.random.seed(seed)
        result = {}
        for k, v in self._archive.items():
            if v: # Chỉ xử lý nếu danh sách không rỗng
                max_val = np.max(v)
                indices = np.where(np.array(v) == max_val)[0]
                result[k] = int(np.random.choice(indices))
        return result

    def subtract(self, archive_instance: "Archive[T]") -> "Archive[T]":
        """
        Trừ một archive khác, giữ lại các phần tử từ n đến cuối.

        Args:
            archive_instance (Archive[T]): Archive để trừ

        Returns:
            Archive[T]: Archive kết quả sau khi trừ
        """
        result = Archive[T](self.name)
        for key, value in self._archive.items():
            other_value = archive_instance.get(key)
            if other_value:
                result.add(key, value[len(other_value) :])
            else:
                result.add(key, value)

        # Xóa các key rỗng
        result._archive = {k: v for k, v in result._archive.items() if v}

        return result

    def save(self, filepath: str) -> None:
        """
        Lưu archive vào file JSON. (Không cần thay đổi)

        Args:
            filepath (str): Đường dẫn để lưu file JSON
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self._archive.items()}, f, indent=2)

    def load(self, filepath: str) -> None:
        """
        Tải archive từ file JSON. (Không cần thay đổi)

        Args:
            filepath (str): Đường dẫn đến file JSON
        """
        with open(filepath, "r", encoding="utf-8") as f:
            self._archive = {ast.literal_eval(k): v for k, v in json.load(f).items()}

    def load_from_dict(self, data: Dict[str, List[T]]) -> None:
        """
        Tải archive từ một dictionary. (Không cần thay đổi)

        Args:
            data (Dict[str, List[T]]): Dictionary để tải
        """
        self._archive = {ast.literal_eval(k): v for k, v in data.items()}

    # --- HÀM MỚI ---
    def visualize(
        self,
        metric: str = "max",
        title: Optional[str] = None,
        xlabel: Optional[str] = "Feature 1",
        ylabel: Optional[str] = "Feature 2",
        cmap: str = "viridis",
        ax: Optional["plt.Axes"] = None,
        **heatmap_kwargs,
    ) -> Optional["plt.Axes"]:
        """
        Trực quan hóa archive dưới dạng bản đồ nhiệt (heatmap).
        Hàm này chỉ hỗ trợ archive 2D (key là tuple 2 phần tử).

        Yêu cầu các thư viện: matplotlib, seaborn, pandas.

        Args:
            metric (str, optional): 
                Số liệu để tô màu các ô.
                - 'max': Giá trị lớn nhất trong ô (mặc định)
                - 'median': Giá trị trung vị trong ô
                - 'count': Số lượng phần tử trong ô
            title (Optional[str], optional): Tiêu đề của biểu đồ.
            xlabel (Optional[str], optional): Nhãn trục X.
            ylabel (Optional[str], optional): Nhãn trục Y.
            cmap (str, optional): Colormap của heatmap.
            ax (Optional[plt.Axes], optional): Một trục matplotlib có sẵn để vẽ.
            **heatmap_kwargs: Các tham số khác để truyền cho seaborn.heatmap (ví dụ: annot=True).

        Returns:
            Optional[plt.Axes]: Trục matplotlib đã vẽ, hoặc None nếu archive rỗng.

        Raises:
            ImportError: Nếu thiếu các thư viện bắt buộc.
            ValueError: Nếu archive không phải 2D hoặc metric không hợp lệ.
        """
        if not _VIS_LIBS_INSTALLED:
            raise ImportError(
                "Các thư viện tùy chọn matplotlib, seaborn, và pandas là bắt buộc "
                "cho việc trực quan hóa. Vui lòng cài đặt chúng: "
                "pip install matplotlib seaborn pandas"
            )

        if not self._archive:
            print("Archive rỗng. Không có gì để trực quan hóa.")
            return None

        all_keys = self.keys()
        if len(all_keys[0]) != 2:
            raise ValueError(
                "Trực quan hóa chỉ hỗ trợ archive 2D "
                f"(key phải là tuple 2 phần tử, tìm thấy: {len(all_keys[0])} phần tử)"
            )

        if metric in ("max", "median") and not self.values_are_numeric():
            raise ValueError(f"Metric '{metric}' yêu cầu tất cả giá trị trong archive phải là số.")

        # Chuẩn bị dữ liệu cho DataFrame
        data = []
        for key, values in self._archive.items():
            if not values:  # Bỏ qua các ô rỗng
                continue
                
            x, y = key
            score = 0.0
            if metric == "max":
                score = np.max(values)
            elif metric == "median":
                score = np.median(values)
            elif metric == "count":
                score = len(values)
            else:
                raise ValueError(f"Metric không xác định: '{metric}'. Chỉ hỗ trợ 'max', 'median', 'count'.")
            
            data.append({"x": x, "y": y, "score": score})
        
        if not data:
            print("Archive không có dữ liệu (có thể tất cả các ô đều rỗng). Không có gì để trực quan hóa.")
            return None

        df = pd.DataFrame(data)

        # Pivot dữ liệu thành lưới 2D
        try:
            grid_data = df.pivot(index="y", columns="x", values="score")
        except Exception as e:
            print(f"Lỗi khi pivot dữ liệu: {e}")
            print("Điều này có thể xảy ra nếu các key (tọa độ) không tạo thành một lưới hợp lệ.")
            return None

        # Sắp xếp index (trục y) giảm dần để giá trị Y thấp ở dưới cùng
        grid_data = grid_data.sort_index(ascending=False)
        
        # Sắp xếp cột (trục x) tăng dần
        grid_data = grid_data.sort_index(axis=1, ascending=True)


        # Vẽ
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        if title is None:
            title = f'Archive "{self.name}" - Metric: {metric.capitalize()}'
            
        # Đặt các giá trị mặc định cho heatmap_kwargs nếu chưa có
        heatmap_kwargs.setdefault("annot", False) # Không hiển thị số trên ô
        heatmap_kwargs.setdefault("fmt", ".2f")

        sns.heatmap(
            grid_data,
            cmap=cmap,
            ax=ax,
            cbar_label=f"Score ({metric.capitalize()})",
            **heatmap_kwargs
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax