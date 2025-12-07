import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import os
import os
import pygame


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.screen_width = 640
        self.screen_height = 400

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # Colors
        self.COLOR_BG = (34, 139, 34)  # Forest Green (Grass)
        self.COLOR_TRACK = (105, 105, 105)  # Gray
        self.COLOR_RUMBLE = (255, 255, 255)  # White
        self.COLOR_PLAYER = (255, 0, 0)  # Red
        self.COLOR_OPPONENT = (0, 0, 255)  # Blue
        self.COLOR_PROJECTILE = (255, 255, 0)  # Yellow
        self.COLOR_TEXT = (255, 255, 255)  # White

        # Game constants
        self.CAR_WIDTH = 20
        self.CAR_HEIGHT = 40
        self.MAX_STEPS = 2500

        # Physics constants
        self.ACCELERATION = 0.15
        self.BRAKING = 0.3
        self.FRICTION = 0.04
        self.TURN_SPEED = 3.5
        self.DRIFT_TURN_MOD = 1.5
        self.MAX_SPEED = 6.0
        self.MAX_REVERSE_SPEED = -2.0
        self.PROJECTILE_SPEED = 10.0
        
        # Track definition (center x, center y, width radius, height radius)
        self.track_outer_ellipse = (self.screen_width / 2, self.screen_height / 2, 280, 160)
        self.track_inner_ellipse = (self.screen_width / 2, self.screen_height / 2, 180, 60)

        # Initialize state
        self.player = {}
        self.opponents = []
        self.projectiles = []
        self.steps = 0
        self.score = 0
        self.shoot_cooldown = 0
        
        # The original code called reset() here, which is correct.
        # All attributes needed by reset() and its sub-methods are now defined.
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player = {
            "pos": np.array([self.screen_width / 2, self.screen_height * 0.15]),
            "angle": 90.0,
            "speed": 0.0,
        }

        # Opponent state
        self.opponents = []
        for i in range(2):
            angle_offset = 180 * (i + 1) / 3
            pos = self._get_point_on_track(angle_offset)
            self.opponents.append({
                "pos": pos,
                "angle": 90.0,
                "speed": self.np_random.uniform(2.0, 4.0),
                "target_waypoint": 1,
                "alive": True
            })

        # Projectiles
        self.projectiles = []
        self.shoot_cooldown = 0
        
        # AI Waypoints
        self.waypoints = [self._get_point_on_track(a) for a in range(0, 360, 30)]

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, fire_action, drift_action = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        self._update_player(movement, fire_action, drift_action)
        self._update_opponents()
        self._update_projectiles()

        # Reward for speed
        reward += self.player["speed"] * 0.01

        # Penalty for being off-track
        if not self._is_on_track(self.player["pos"]):
            reward -= 0.1
            self.player["speed"] *= 0.8 # Slow down on grass

        # Handle collisions and score
        reward += self._handle_collisions()

        self.steps += 1
        self.score += reward
        terminated = self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_point_on_track(self, angle_deg):
        """Gets a point on the centerline of the track for a given angle."""
        cx, cy = self.screen_width / 2, self.screen_height / 2
        rx = (self.track_outer_ellipse[2] + self.track_inner_ellipse[2]) / 2
        ry = (self.track_outer_ellipse[3] + self.track_inner_ellipse[3]) / 2
        rad = math.radians(angle_deg)
        x = cx + rx * math.cos(rad)
        y = cy + ry * math.sin(rad)
        return np.array([x, y])

    def _update_player(self, movement, fire_action, drift_action):
        # Handle turning
        turn_multiplier = self.DRIFT_TURN_MOD if drift_action and abs(self.player["speed"]) > 2.0 else 1.0
        if movement == 3:  # Left
            self.player["angle"] += self.TURN_SPEED * turn_multiplier
        if movement == 4:  # Right
            self.player["angle"] -= self.TURN_SPEED * turn_multiplier

        # Handle acceleration/braking
        if movement == 1:  # Up
            self.player["speed"] += self.ACCELERATION
        elif movement == 2:  # Down
            self.player["speed"] -= self.BRAKING
        else: # Apply friction
            self.player["speed"] *= (1 - self.FRICTION)

        # Clamp speed
        self.player["speed"] = np.clip(self.player["speed"], self.MAX_REVERSE_SPEED, self.MAX_SPEED)
        if abs(self.player["speed"]) < self.FRICTION:
            self.player["speed"] = 0

        # Update position
        angle_rad = math.radians(self.player["angle"])
        self.player["pos"][0] += -self.player["speed"] * math.sin(angle_rad)
        self.player["pos"][1] += -self.player["speed"] * math.cos(angle_rad)
        
        # Handle shooting
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if fire_action and self.shoot_cooldown == 0:
            self.shoot_cooldown = 20 # 20 frames cooldown
            proj_pos = self.player["pos"] + np.array([-self.CAR_HEIGHT/2 * math.sin(angle_rad), -self.CAR_HEIGHT/2 * math.cos(angle_rad)])
            self.projectiles.append({"pos": proj_pos, "angle": self.player["angle"], "owner": "player"})

    def _update_opponents(self):
        for opp in self.opponents:
            if not opp["alive"]: continue
            
            target_pos = self.waypoints[opp["target_waypoint"]]
            direction_vec = target_pos - opp["pos"]
            distance = np.linalg.norm(direction_vec)
            
            if distance < 30:
                opp["target_waypoint"] = (opp["target_waypoint"] + 1) % len(self.waypoints)
            
            target_angle = math.degrees(math.atan2(-direction_vec[0], -direction_vec[1]))
            angle_diff = (target_angle - opp["angle"] + 180) % 360 - 180
            opp["angle"] += np.clip(angle_diff, -self.TURN_SPEED, self.TURN_SPEED)
            
            angle_rad = math.radians(opp["angle"])
            opp["pos"][0] += -opp["speed"] * math.sin(angle_rad)
            opp["pos"][1] += -opp["speed"] * math.cos(angle_rad)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            angle_rad = math.radians(proj["angle"])
            proj["pos"][0] += -self.PROJECTILE_SPEED * math.sin(angle_rad)
            proj["pos"][1] += -self.PROJECTILE_SPEED * math.cos(angle_rad)
            if not (0 <= proj["pos"][0] < self.screen_width and 0 <= proj["pos"][1] < self.screen_height):
                self.projectiles.remove(proj)

    def _handle_collisions(self):
        reward = 0
        player_rect = self._get_car_rect(self.player)

        # Projectile collisions
        for proj in self.projectiles[:]:
            if proj["owner"] == "player":
                for opp in self.opponents:
                    if opp["alive"] and self._get_car_rect(opp).collidepoint(proj["pos"]):
                        opp["alive"] = False
                        reward += 10
                        self.projectiles.remove(proj)
                        break
        return reward
    
    def _is_on_track(self, pos):
        x, y = pos
        cx_out, cy_out, rx_out, ry_out = self.track_outer_ellipse
        cx_in, cy_in, rx_in, ry_in = self.track_inner_ellipse
        
        is_in_outer = ((x - cx_out) / rx_out)**2 + ((y - cy_out) / ry_out)**2 <= 1
        is_out_inner = ((x - cx_in) / rx_in)**2 + ((y - cy_in) / ry_in)**2 >= 1
        
        return is_in_outer and is_out_inner

    def _get_car_rect(self, car):
        # This is a simplified AABB for collision, not the rotated one
        return pygame.Rect(car["pos"][0] - self.CAR_WIDTH/2, car["pos"][1] - self.CAR_HEIGHT/2, self.CAR_WIDTH, self.CAR_HEIGHT)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Draw track
        pygame.draw.ellipse(self.screen, self.COLOR_TRACK, [
            self.track_outer_ellipse[0] - self.track_outer_ellipse[2],
            self.track_outer_ellipse[1] - self.track_outer_ellipse[3],
            self.track_outer_ellipse[2] * 2,
            self.track_outer_ellipse[3] * 2
        ])
        pygame.draw.ellipse(self.screen, self.COLOR_BG, [
            self.track_inner_ellipse[0] - self.track_inner_ellipse[2],
            self.track_inner_ellipse[1] - self.track_inner_ellipse[3],
            self.track_inner_ellipse[2] * 2,
            self.track_inner_ellipse[3] * 2
        ])

        # Draw cars
        self._draw_rotated_rect(self.player, self.COLOR_PLAYER)
        for opp in self.opponents:
            if opp["alive"]:
                self._draw_rotated_rect(opp, self.COLOR_OPPONENT)
        
        # Draw projectiles
        for proj in self.projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, proj["pos"].astype(int), 4)

    def _draw_rotated_rect(self, car, color):
        rect_surf = pygame.Surface((self.CAR_WIDTH, self.CAR_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(rect_surf, color, (0, 0, self.CAR_WIDTH, self.CAR_HEIGHT))
        rotated_surf = pygame.transform.rotate(rect_surf, car["angle"])
        new_rect = rotated_surf.get_rect(center=car["pos"])
        self.screen.blit(rotated_surf, new_rect.topleft)

    def _render_ui(self):
        score_text = self.font.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 40))

    def close(self):
        pygame.quit()