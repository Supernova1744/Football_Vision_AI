import cv2
import numpy as np
from typing import Tuple, Optional

class pitchDrawer:
    def __init__(self, pitchConfig, config) -> None:
        self.pitchConfig = pitchConfig
        self.config = config
        self.ball_icon = cv2.imread(self.config.BALL_ICON, cv2.IMREAD_UNCHANGED)
        self.player1_icon = cv2.imread(self.config.TEAM1_ICON, cv2.IMREAD_UNCHANGED)
        self.player2_icon = cv2.imread(self.config.TEAM2_ICON, cv2.IMREAD_UNCHANGED)
        self.referee_icon = cv2.imread(self.config.REFEREE_ICON, cv2.IMREAD_UNCHANGED)
        self.keeper_icon = cv2.imread(self.config.KEEPER_ICON, cv2.IMREAD_UNCHANGED)

        
    
    def draw_pitch(
        self,
        background_color: Tuple = (34, 139, 34),
        line_color: Tuple = (255, 255, 255),
        padding: int = 50,
        line_thickness: int = 4,
        point_radius: int = 8,
        scale: float = 0.1
    ) -> np.ndarray:
        """
        Draws a soccer pitch with specified dimensions, colors, and scale.

        Args:
            background_color (sv.Color, optional): Color of the pitch background.
                Defaults to sv.Color(34, 139, 34).
            line_color (sv.Color, optional): Color of the pitch lines.
                Defaults to sv.Color.WHITE.
            padding (int, optional): Padding around the pitch in pixels.
                Defaults to 50.
            line_thickness (int, optional): Thickness of the pitch lines in pixels.
                Defaults to 4.
            point_radius (int, optional): Radius of the penalty spot points in pixels.
                Defaults to 8.
            scale (float, optional): Scaling factor for the pitch dimensions.
                Defaults to 0.1.

        Returns:
            np.ndarray: Image of the soccer pitch.
        """
        scaled_width = int(self.pitchConfig.width * scale)
        scaled_length = int(self.pitchConfig.length * scale)
        scaled_circle_radius = int(self.pitchConfig.centre_circle_radius * scale)
        scaled_penalty_spot_distance = int(self.pitchConfig.penalty_spot_distance * scale)

        pitch_image = np.ones(
            (scaled_width + 2 * padding,
            scaled_length + 2 * padding, 3),
            dtype=np.uint8
        ) * np.array(background_color[::-1], dtype=np.uint8)

        for start, end in self.pitchConfig.edges:
            point1 = (int(self.pitchConfig.vertices[start - 1][0] * scale) + padding,
                    int(self.pitchConfig.vertices[start - 1][1] * scale) + padding)
            point2 = (int(self.pitchConfig.vertices[end - 1][0] * scale) + padding,
                    int(self.pitchConfig.vertices[end - 1][1] * scale) + padding)
            cv2.line(
                img=pitch_image,
                pt1=point1,
                pt2=point2,
                color=line_color[::-1],
                thickness=line_thickness
            )

        centre_circle_center = (
            scaled_length // 2 + padding,
            scaled_width // 2 + padding
        )
        cv2.circle(
            img=pitch_image,
            center=centre_circle_center,
            radius=scaled_circle_radius,
            color=line_color[::-1],
            thickness=line_thickness
        )

        penalty_spots = [
            (
                scaled_penalty_spot_distance + padding,
                scaled_width // 2 + padding
            ),
            (
                scaled_length - scaled_penalty_spot_distance + padding,
                scaled_width // 2 + padding
            )
        ]
        for spot in penalty_spots:
            cv2.circle(
                img=pitch_image,
                center=spot,
                radius=point_radius,
                color=line_color[::-1],
                thickness=-1
            )

        return pitch_image
        
    def draw_points_on_pitch(
        self,
        xy: np.ndarray,
        face_color = (255,0,0),
        edge_color= (0,0,0),
        radius: int = 10,
        thickness: int = 2,
        padding: int = 50,
        scale: float = 0.1,
        pitch: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draws points on a soccer pitch.

        Args:
            xy (np.ndarray): Array of points to be drawn, with each point represented by
                its (x, y) coordinates.
            face_color (sv.Color, optional): Color of the point faces.
                Defaults to sv.Color.RED.
            edge_color (sv.Color, optional): Color of the point edges.
                Defaults to sv.Color.BLACK.
            radius (int, optional): Radius of the points in pixels.
                Defaults to 10.
            thickness (int, optional): Thickness of the point edges in pixels.
                Defaults to 2.
            padding (int, optional): Padding around the pitch in pixels.
                Defaults to 50.
            scale (float, optional): Scaling factor for the pitch dimensions.
                Defaults to 0.1.
            pitch (Optional[np.ndarray], optional): Existing pitch image to draw points on.
                If None, a new pitch will be created. Defaults to None.

        Returns:
            np.ndarray: Image of the soccer pitch with points drawn on it.
        """
        if pitch is None:
            pitch = self.draw_pitch(
                padding=padding,
                scale=scale
            )

        for point in xy:
            scaled_point = (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            cv2.circle(
                img=pitch,
                center=scaled_point,
                radius=radius,
                color=face_color[::-1],
                thickness=-1
            )
            cv2.circle(
                img=pitch,
                center=scaled_point,
                radius=radius,
                color=edge_color[::-1],
                thickness=thickness
            )

        return pitch

    def draw_ball_on_pitch(
        self,
        xy: np.ndarray,
        edge_color= (0,0,0),
        radius: int = 10,
        thickness: int = 2,
        padding: int = 50,
        scale: float = 0.1,
        pitch: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draws points on a soccer pitch.

        Args:
            xy (np.ndarray): Array of points to be drawn, with each point represented by
                its (x, y) coordinates.
            face_color (sv.Color, optional): Color of the point faces.
                Defaults to sv.Color.RED.
            edge_color (sv.Color, optional): Color of the point edges.
                Defaults to sv.Color.BLACK.
            radius (int, optional): Radius of the points in pixels.
                Defaults to 10.
            thickness (int, optional): Thickness of the point edges in pixels.
                Defaults to 2.
            padding (int, optional): Padding around the pitch in pixels.
                Defaults to 50.
            scale (float, optional): Scaling factor for the pitch dimensions.
                Defaults to 0.1.
            pitch (Optional[np.ndarray], optional): Existing pitch image to draw points on.
                If None, a new pitch will be created. Defaults to None.

        Returns:
            np.ndarray: Image of the soccer pitch with points drawn on it.
        """
        if pitch is None:
            pitch = self.draw_pitch(
                padding=padding,
                scale=scale
            )
        
        ball = cv2.resize(self.ball_icon, (2*radius, 2*radius))
        for point in xy:
            scaled_point = (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            h, w = ball.shape[:2]
            h = h // 2
            w = w // 2
            x,y = scaled_point[0], scaled_point[1]
            x = np.clip(x, 0+w, pitch.shape[1]-w)
            y = np.clip(y, 0+h, pitch.shape[0]-h)
        
            # Extract the alpha mask of the overlay image
            alpha_mask = ball[:, :, 3] / 255.0
            
            # Extract the RGB channels of the overlay image
            overlay_rgb = ball[:, :, :3]
            
            
            # Get the region of interest from the background image
            roi = pitch[y-h:y+h, x-w:x+w]
            
            # Blend the overlay image and the region of interest
            blended = (1.0 - alpha_mask[..., np.newaxis]) * roi + alpha_mask[..., np.newaxis] * overlay_rgb
            # Place the blended region back into the background image
            pitch[y-h:y+h, x-w:x+w] = blended
            
            cv2.circle(
                img=pitch,
                center=(x, y),
                radius=radius,
                color=edge_color[::-1],
                thickness=thickness
            )

        return pitch

    def draw_person_on_pitch(
        self,
        player_icon: np.ndarray,
        xy: np.ndarray,
        face_color = (255,0,0),
        radius: int = 10,
        thickness: int = 2,
        padding: int = 50,
        scale: float = 0.1,
        pitch: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draws points on a soccer pitch.

        Args:
            xy (np.ndarray): Array of points to be drawn, with each point represented by
                its (x, y) coordinates.
            face_color (sv.Color, optional): Color of the point faces.
                Defaults to sv.Color.RED.
            edge_color (sv.Color, optional): Color of the point edges.
                Defaults to sv.Color.BLACK.
            radius (int, optional): Radius of the points in pixels.
                Defaults to 10.
            thickness (int, optional): Thickness of the point edges in pixels.
                Defaults to 2.
            padding (int, optional): Padding around the pitch in pixels.
                Defaults to 50.
            scale (float, optional): Scaling factor for the pitch dimensions.
                Defaults to 0.1.
            pitch (Optional[np.ndarray], optional): Existing pitch image to draw points on.
                If None, a new pitch will be created. Defaults to None.

        Returns:
            np.ndarray: Image of the soccer pitch with points drawn on it.
        """
        if pitch is None:
            pitch = self.draw_pitch(
                padding=padding,
                scale=scale
            )
        player = cv2.resize(player_icon, (2*radius, 2*radius))
        for point in xy:
            scaled_point = (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            h, w = player.shape[:2]
            h = h // 2
            w = w // 2
        
            # Extract the alpha mask of the overlay image
            alpha_mask = player[:, :, 3] / 255.0
            
            # Extract the RGB channels of the overlay image
            overlay_rgb = player[:, :, :3]
            
            x,y = scaled_point[0], scaled_point[1]
            
            x = np.clip(x, 0+w, pitch.shape[1]-w)
            y = np.clip(y, 0+h, pitch.shape[0]-h)
            # Get the region of interest from the background image
            roi = pitch[y-h:y+h, x-w:x+w]
            
            # Blend the overlay image and the region of interest
            blended = (1.0 - alpha_mask[..., np.newaxis]) * roi + alpha_mask[..., np.newaxis] * overlay_rgb
            
            # Place the blended region back into the background image
            pitch[y-h:y+h, x-w:x+w] = blended
            
            cv2.circle(
                img=pitch,
                center=(x, y),
                radius=radius,
                color=face_color[::-1],
                thickness=thickness
            )

        return pitch

    def pitch_processing(self, ball_detections, online_players, online_targets, view_transformer):
        pitch = self.draw_pitch()
        pitch_team_1 = view_transformer.transform_points(online_players[online_players[:, 4] == self.config.TEAM_ONE_ID, :2])
        pitch_team_2 = view_transformer.transform_points(online_players[online_players[:, 4] == self.config.TEAM_TWO_ID, :2])
        pitch_goalkeeper = view_transformer.transform_points(online_targets[online_targets[:, 4] == self.config.GOALKEEPER_CLS_ID, :2])
        pitch_referres = view_transformer.transform_points(online_targets[online_targets[:, 4] == self.config.REFERRE_CLS_ID, :2])
        # pitch = draw_pitch_voronoi_diagram(self.pitchConfig, pitch_team_1, pitch_team_2, self.config.COLORS[self.config.TEAM_ONE_ID][::-1], self.config.COLORS[self.config.TEAM_TWO_ID][::-1], pitch=pitch, opacity=0.4)

        pitch = self.draw_person_on_pitch(
                                self.referee_icon,
                                pitch_referres,
                                pitch=pitch,
                                radius=25,
                                face_color=self.config.COLORS[self.config.REFERRE_CLS_ID])
        
        pitch = self.draw_person_on_pitch(
                                self.keeper_icon,
                                pitch_goalkeeper,
                                pitch=pitch,
                                radius=25,
                                face_color=self.config.COLORS[self.config.GOALKEEPER_CLS_ID])
        
        pitch = self.draw_person_on_pitch(
                                self.player1_icon,
                                pitch_team_1,
                                pitch=pitch,
                                radius=25,
                                face_color=self.config.COLORS[self.config.TEAM_ONE_ID][::-1])
        
        pitch = self.draw_person_on_pitch(
                                self.player2_icon,
                                pitch_team_2,
                                pitch=pitch,
                                radius=25,
                                face_color=self.config.COLORS[self.config.TEAM_TWO_ID][::-1])

        if ball_detections.shape[0]:
            pitch_ball = view_transformer.transform_points(ball_detections[:, :2])
            pitch = self.draw_ball_on_pitch(
                                pitch_ball,
                                pitch=pitch,
                                radius=25,
                                )

        return pitch