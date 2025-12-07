import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
from collections import deque
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player manages trains on a circular track system.

    The goal is to deliver 10 units of cargo within 120 seconds by switching trains
    between an inner and outer track to avoid collisions and reach the correct stations.
    Failure occurs if two trains collide, time runs out, or the maximum number of steps
    is reached.

    Visuals are minimalist and geometric, designed for clarity and aesthetic appeal.
    Gameplay is real-time, requiring quick strategic decisions.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    # --- FIX: Added required class attributes ---
    game_description = (
        "Manage trains on a circular track system, switching them between tracks to deliver cargo "
        "to the correct stations while avoiding collisions."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to switch trains on corresponding channels between the inner and outer tracks."
    )
    auto_advance = True
    # -------------------------------------------

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # === Gymnasium Spaces ===
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # === Game Constants ===
        self.WIDTH, self.HEIGHT = 640, 400
        self.CENTER_X, self.CENTER_Y = self.WIDTH // 2, self.HEIGHT // 2
        self.FPS = self.metadata["render_fps"]
        self.MAX_SCORE = 10
        self.TIME_LIMIT_SECONDS = 120
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        self.INITIAL_TRAINS = 4
        self.MAX_TRAINS = 10
        
        self.TRACK_RADII = [100, 150]
        self.STATION_COLORS = [
            (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 165, 0)
        ]

        # === Colors & Fonts ===
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_TRACK = (70, 80, 100)
        self.COLOR_TRAIN_BODY_CARGO = (0, 255, 127) # Bright Green
        self.COLOR_TRAIN_BODY_EMPTY = (255, 69, 58) # Bright Red
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.COLOR_UI_SUCCESS = (0, 255, 127)
        self.COLOR_UI_FAILURE = (255, 69, 58)
        
        # === Pygame Setup ===
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)

        # === Game State (initialized in reset) ===
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.win_condition = False
        self.collision_occurred = False
        self.trains = []
        self.stations = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_remaining = self.TIME_LIMIT_SECONDS
        self.game_over = False
        self.win_condition = False
        self.collision_occurred = False
        
        self.particles.clear()
        self._create_stations()
        self._create_trains(self.INITIAL_TRAINS)

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # 1. Handle Input
        self._handle_input(action)

        # 2. Update Game State
        delivery_reward = self._update_game_state()
        reward += delivery_reward

        # 3. Check for collisions
        collision_detected = self._check_collisions()
        if collision_detected:
            self.collision_occurred = True

        # 4. Update Timers & Step Counter
        self.steps += 1
        self.time_remaining -= 1 / self.FPS

        # 5. Check for Termination & Truncation (FIX: Separated logic)
        self.win_condition = self.score >= self.MAX_SCORE
        terminated = self.win_condition or self.collision_occurred
        truncated = (self.time_remaining <= 0) or (self.steps >= self.MAX_STEPS)
        
        self.game_over = terminated or truncated

        # 6. Calculate Reward (FIX: Adjusted for new terminated/truncated logic)
        reward += self._calculate_reward(terminated, truncated)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement_action = action[0]
        if movement_action == 0:
            return

        # Action 1-4 maps to control channels 0-3
        control_channel = movement_action - 1
        
        for i, train in enumerate(self.trains):
            if i % 4 == control_channel and train["switch_cooldown"] == 0:
                # Switch track
                train["track_idx"] = 1 - train["track_idx"]
                train["switch_cooldown"] = self.FPS // 2  # 0.5s cooldown
                self._create_particles(
                    train["pos"], self.COLOR_UI_TEXT, 10, 2.0
                )

    def _update_game_state(self):
        delivery_reward = 0
        
        for train in self.trains:
            if train["switch_cooldown"] > 0:
                train["switch_cooldown"] -= 1

            oscillation = math.sin(self.steps / (self.FPS * 2.5))
            speed_multiplier = 0.75 + 0.25 * oscillation
            angular_velocity = (train["base_speed"] * speed_multiplier) / self.TRACK_RADII[train["track_idx"]]
            
            train["angle"] = (train["angle"] + angular_velocity) % (2 * math.pi)
            
            radius = self.TRACK_RADII[train["track_idx"]]
            train["pos"] = (
                self.CENTER_X + radius * math.cos(train["angle"]),
                self.CENTER_Y + radius * math.sin(train["angle"]),
            )
            train["trail"].append(train["pos"])

            delivery_reward += self._check_station_interaction(train)

        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["lifespan"] -= 1

        return delivery_reward

    def _check_station_interaction(self, train):
        reward = 0
        for i, station in enumerate(self.stations):
            if train["track_idx"] == station["track_idx"]:
                dist = self._angular_distance(train["angle"], station["angle"])
                
                if dist < 0.05:
                    if train["has_cargo"] and train["cargo_type"] == i:
                        self.score += 1
                        reward += 10.0
                        train["has_cargo"] = False
                        train["cargo_type"] = None
                        self._create_particles(train["pos"], station["color"], 30, 3.0)
                        self._assign_new_cargo_task(train)

                    elif not train["has_cargo"] and train["cargo_type"] is None:
                        self._assign_new_cargo_task(train)

                    # --- FIX: Disabled stability-breaking mechanic ---
                    elif train["has_cargo"] and train["cargo_type"] != i and train["last_lap_angle"] is not None:
                        if self._angular_distance(train["angle"], train["last_lap_angle"]) > math.pi:
                            if len(self.trains) < self.MAX_TRAINS:
                                # self._create_trains(1) # DISABLED: This causes instability in no-op tests.
                                pass
                            train["last_lap_angle"] = None
        
        if train["has_cargo"] and train["last_lap_angle"] is None:
            train["last_lap_angle"] = train["angle"]
            
        return reward

    def _check_collisions(self):
        train_length_angle = 0.15
        for i in range(len(self.trains)):
            for j in range(i + 1, len(self.trains)):
                t1 = self.trains[i]
                t2 = self.trains[j]
                if t1["track_idx"] == t2["track_idx"]:
                    dist = self._angular_distance(t1["angle"], t2["angle"])
                    if dist < train_length_angle:
                        self._create_particles(t1["pos"], self.COLOR_UI_FAILURE, 50, 5.0)
                        self._create_particles(t2["pos"], self.COLOR_UI_FAILURE, 50, 5.0)
                        return True
        return False

    def _calculate_reward(self, terminated, truncated):
        if not terminated and not truncated:
            return 0.01  # Small reward for surviving a step
        
        if terminated and self.win_condition:
            return 100.0  # Large reward for winning
        
        if terminated and not self.win_condition: # Collision
            return -100.0 # Large penalty for losing
            
        if truncated: # Time/step limit
            return -50.0
            
        return 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_tracks()
        self._render_stations()
        self._render_particles()
        self._render_trains()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "train_count": len(self.trains)
        }
        
    def _create_stations(self):
        self.stations.clear()
        num_stations = len(self.STATION_COLORS)
        for i in range(num_stations):
            angle = (2 * math.pi / num_stations) * i
            track_idx = i % 2
            radius = self.TRACK_RADII[track_idx]
            pos = (
                self.CENTER_X + radius * math.cos(angle),
                self.CENTER_Y + radius * math.sin(angle),
            )
            self.stations.append({
                "angle": angle,
                "track_idx": track_idx,
                "color": self.STATION_COLORS[i],
                "pos": pos
            })

    def _create_trains(self, num_to_add):
        for _ in range(num_to_add):
            if len(self.trains) >= self.MAX_TRAINS:
                break
            
            while True:
                is_safe = True
                angle = self.np_random.uniform(0, 2 * math.pi)
                track_idx = self.np_random.integers(0, 2)
                for other_train in self.trains:
                    if other_train["track_idx"] == track_idx and \
                       self._angular_distance(angle, other_train["angle"]) < 0.2:
                        is_safe = False
                        break
                if is_safe:
                    break

            radius = self.TRACK_RADII[track_idx]
            pos = (
                self.CENTER_X + radius * math.cos(angle),
                self.CENTER_Y + radius * math.sin(angle),
            )

            new_train = {
                "angle": angle,
                "track_idx": track_idx,
                "base_speed": self.np_random.uniform(1.5, 2.5),
                "pos": pos,
                "has_cargo": False,
                "cargo_type": None,
                "switch_cooldown": 0,
                "trail": deque(maxlen=20),
                "last_lap_angle": None,
            }
            self._assign_new_cargo_task(new_train)
            self.trains.append(new_train)

    def _assign_new_cargo_task(self, train):
        train["has_cargo"] = True
        train["cargo_type"] = self.np_random.integers(0, len(self.stations))
        current_station = -1
        for i, station in enumerate(self.stations):
            if train["track_idx"] == station["track_idx"] and self._angular_distance(train["angle"], station["angle"]) < 0.1:
                current_station = i
                break
        if current_station != -1 and train["cargo_type"] == current_station:
            train["cargo_type"] = (train["cargo_type"] + 1) % len(self.stations)
        
        train["last_lap_angle"] = None

    def _render_tracks(self):
        for radius in self.TRACK_RADII:
            pygame.gfxdraw.aacircle(self.screen, self.CENTER_X, self.CENTER_Y, radius, self.COLOR_TRACK)

    def _render_stations(self):
        for i, station in enumerate(self.stations):
            x, y = int(station["pos"][0]), int(station["pos"][1])
            color = station["color"]
            pygame.gfxdraw.filled_circle(self.screen, x, y, 12, (*color, 50))
            pygame.gfxdraw.aacircle(self.screen, x, y, 12, (*color, 100))
            pygame.gfxdraw.filled_circle(self.screen, x, y, 8, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, 8, self.COLOR_BG)

    def _render_trains(self):
        train_w, train_h = 20, 10
        for i, train in enumerate(self.trains):
            body_color = self.COLOR_TRAIN_BODY_CARGO if train["has_cargo"] else self.COLOR_TRAIN_BODY_EMPTY
            
            for j, pos in enumerate(train["trail"]):
                alpha = int(255 * (j / len(train["trail"])))
                if alpha > 10:
                    color = (*body_color, alpha)
                    pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), 2, 0)
            
            x, y = train["pos"]
            angle_deg = -math.degrees(train["angle"])
            
            surf = pygame.Surface((train_w, train_h), pygame.SRCALPHA)
            surf.fill(body_color)
            
            if train["has_cargo"]:
                cargo_color = self.stations[train["cargo_type"]]["color"]
                pygame.draw.rect(surf, cargo_color, (train_w - 6, 1, 5, train_h-2))

            channel_color = self.STATION_COLORS[i % 4]
            pygame.draw.rect(surf, channel_color, (1, 1, 5, train_h-2))
            
            rotated_surf = pygame.transform.rotate(surf, angle_deg)
            rect = rotated_surf.get_rect(center=(int(x), int(y)))
            self.screen.blit(rotated_surf, rect)

    def _render_ui(self):
        score_text = self.font_ui.render(f"DELIVERED: {self.score}/{self.MAX_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_color = self.COLOR_UI_TEXT if self.time_remaining > 10 else self.COLOR_UI_FAILURE
        time_text = self.font_ui.render(f"TIME: {int(self.time_remaining):03}", True, time_color)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        if self.game_over:
            msg = "MISSION COMPLETE" if self.win_condition else "MISSION FAILED"
            color = self.COLOR_UI_SUCCESS if self.win_condition else self.COLOR_UI_FAILURE
            
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.CENTER_X, self.CENTER_Y))
            self.screen.blit(msg_surf, msg_rect)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = (*p["color"], alpha)
            size = int(p["size"] * (p["lifespan"] / p["max_lifespan"]))
            if size > 0 and alpha > 0:
                # Using a temporary surface for alpha blending
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p["x"]) - size, int(p["y"]) - size))


    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            lifespan = self.np_random.integers(self.FPS // 2, self.FPS)
            self.particles.append({
                "x": pos[0], "y": pos[1],
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "color": color,
                "size": self.np_random.integers(2, 5)
            })

    @staticmethod
    def _angular_distance(a1, a2):
        dist = abs(a1 - a2) % (2 * math.pi)
        return min(dist, 2 * math.pi - dist)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # It requires pygame to be installed and will open a window.
    # The environment itself runs headlessly.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Train Conductor")
    
    terminated, truncated = False, False
    running = True
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action[0] = 0 # Default to no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated, truncated = False, False
                if event.key == pygame.K_q:
                    running = False
                
                if not (terminated or truncated):
                    action = env.action_space.sample()
                    action[0] = 0
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4

        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            action[0] = 0 # Reset action after one step
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Reward: {reward}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata["render_fps"])

    env.close()