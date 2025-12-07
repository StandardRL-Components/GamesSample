import gymnasium as gym
import os
import pygame
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a submerged temple, collect hidden artifacts, and reach the exit while avoiding patrol drones and their sonar."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Hold space for a water jet boost. Press shift to teleport between safe zones."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (10, 25, 47)          # Dark Navy
    COLOR_WALL = (31, 58, 95)        # Desaturated Blue
    COLOR_PLAYER = (100, 255, 218)   # Bright Cyan
    COLOR_PLAYER_GLOW = (100, 255, 218, 50)
    COLOR_ENEMY = (255, 70, 85)      # Bright Red
    COLOR_ENEMY_VISION = (255, 70, 85, 50)
    COLOR_ARTIFACT = (255, 215, 0)   # Gold
    COLOR_TELEPORTER = (150, 0, 255) # Purple
    COLOR_EXIT = (0, 255, 127)       # Spring Green
    COLOR_TEXT = (230, 240, 255)     # Off-white
    COLOR_PARTICLE = (220, 240, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)

        self.render_mode = render_mode
        self._initialize_state_variables()

    def _initialize_state_variables(self):
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # Player
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_radius = 10
        self.player_speed = 1.5
        self.player_jet_boost = 4.0
        self.player_drag = 0.85

        # Actions
        self.last_shift_state = 0

        # Level entities
        self.walls = []
        self.artifacts = []
        self.teleporters = []
        self.enemies = []
        self.exit_pos = pygame.Vector2(0, 0)
        self.exit_radius = 15

        # Effects
        self.particles = []
        self.sonar_pings = []

        # RL state
        self.artifacts_collected_count = 0
        self.dist_to_closest_artifact = float('inf')
        self.detection_level = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state_variables()

        # --- Level Design ---
        self.player_pos = pygame.Vector2(60, self.HEIGHT / 2)

        self.walls = [
            pygame.Rect(0, 0, self.WIDTH, 10),
            pygame.Rect(0, self.HEIGHT - 10, self.WIDTH, 10),
            pygame.Rect(0, 0, 10, self.HEIGHT),
            pygame.Rect(self.WIDTH - 10, 0, 10, self.HEIGHT),
            pygame.Rect(150, 100, 20, 200),
            pygame.Rect(300, 0, 20, 150),
            pygame.Rect(300, 250, 20, 150),
            pygame.Rect(450, 100, 20, 200),
        ]

        self.artifacts = [
            {"pos": pygame.Vector2(225, 50), "collected": False, "radius": 8},
            {"pos": pygame.Vector2(380, 200), "collected": False, "radius": 8},
            {"pos": pygame.Vector2(550, 350), "collected": False, "radius": 8},
        ]

        self.teleporters = [
            pygame.Vector2(40, 40),
            pygame.Vector2(600, 40),
            pygame.Vector2(40, 360),
            pygame.Vector2(600, 360),
        ]
        self.current_teleport_index = 0

        self.enemies = [
            self._create_enemy(pygame.Vector2(200, 150), pygame.Vector2(200, 250)),
            self._create_enemy(pygame.Vector2(400, 250), pygame.Vector2(400, 150)),
            self._create_enemy(pygame.Vector2(500, 80), pygame.Vector2(580, 80)),
        ]

        self.exit_pos = pygame.Vector2(self.WIDTH - 40, self.HEIGHT / 2)

        return self._get_observation(), self._get_info()

    def _create_enemy(self, start_pos, end_pos):
        return {
            "pos": pygame.Vector2(start_pos),
            "start": pygame.Vector2(start_pos),
            "end": pygame.Vector2(end_pos),
            "speed": 0.8,
            "radius": 12,
            "vision_angle": 45,
            "vision_range": 100,
            "direction_vec": (end_pos - start_pos).normalize() if start_pos != end_pos else pygame.Vector2(1,0),
            "sonar_timer": self.np_random.integers(0, 150),
            "sonar_cooldown": 150,
            "sonar_range": 75,
        }

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        self.steps += 1

        # --- Handle Actions ---
        move_vec = self._handle_movement(movement, space_held)
        self._handle_teleport(shift_held)

        # --- Update Game State ---
        self._update_player(move_vec)
        self._update_enemies()
        self._update_effects()

        # --- Check Collisions and Events ---
        collected_artifact = self._check_artifact_collection()
        if collected_artifact:
            reward += 5.0
            self.score += 5

        detected = self._check_detection()
        if detected:
            self.game_over = True
            terminated = True
            reward = -100.0
            self.score -= 100

        all_collected = self.artifacts_collected_count == len(self.artifacts)
        if all_collected and self.player_pos.distance_to(self.exit_pos) < self.player_radius + self.exit_radius:
            self.game_over = True
            self.win = True
            terminated = True
            reward = 100.0
            self.score += 100

        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Per Gymnasium docs, terminated should also be true if truncated
            self.game_over = True
            reward -= 10 # Penalty for running out of time

        # --- Continuous Rewards ---
        reward += self._calculate_continuous_reward()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement_action, space_held):
        move_vec = pygame.Vector2(0, 0)
        if movement_action == 1: move_vec.y = -1
        elif movement_action == 2: move_vec.y = 1
        elif movement_action == 3: move_vec.x = -1
        elif movement_action == 4: move_vec.x = 1

        if move_vec.length() > 0:
            move_vec.normalize_ip()
            if space_held:
                self.player_vel += move_vec * self.player_jet_boost
                # Spawn particles
                for _ in range(3):
                    p_vel = -move_vec * self.np_random.uniform(2, 4) + pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
                    p_pos = self.player_pos - move_vec * self.player_radius
                    self.particles.append({"pos": p_pos, "vel": p_vel, "life": 20, "radius": self.np_random.uniform(1, 3)})
        return move_vec

    def _handle_teleport(self, shift_held):
        if shift_held and not self.last_shift_state:
            self.current_teleport_index = (self.current_teleport_index + 1) % len(self.teleporters)
            self.player_pos = pygame.Vector2(self.teleporters[self.current_teleport_index])
            self.player_vel *= 0 # Stop momentum on teleport
            # Teleport visual effect
            for i in range(50):
                angle = i * (360 / 50)
                rad = math.radians(angle)
                p_vel = pygame.Vector2(math.cos(rad), math.sin(rad)) * self.np_random.uniform(2, 5)
                self.particles.append({"pos": pygame.Vector2(self.player_pos), "vel": p_vel, "life": 15, "radius": self.np_random.uniform(1, 2.5)})
        self.last_shift_state = shift_held

    def _update_player(self, move_vec):
        self.player_vel += move_vec * self.player_speed
        self.player_vel *= self.player_drag
        if self.player_vel.length() < 0.1: self.player_vel = pygame.Vector2(0, 0)

        # Collision with walls
        self.player_pos.x += self.player_vel.x
        player_rect = pygame.Rect(self.player_pos.x - self.player_radius, self.player_pos.y - self.player_radius, self.player_radius * 2, self.player_radius * 2)
        for wall in self.walls:
            if wall.colliderect(player_rect):
                if self.player_vel.x > 0: self.player_pos.x = wall.left - self.player_radius
                elif self.player_vel.x < 0: self.player_pos.x = wall.right + self.player_radius
                self.player_vel.x = 0
                player_rect.x = self.player_pos.x - self.player_radius

        self.player_pos.y += self.player_vel.y
        player_rect.y = self.player_pos.y - self.player_radius
        for wall in self.walls:
            if wall.colliderect(player_rect):
                if self.player_vel.y > 0: self.player_pos.y = wall.top - self.player_radius
                elif self.player_vel.y < 0: self.player_pos.y = wall.bottom + self.player_radius
                self.player_vel.y = 0
                player_rect.y = self.player_pos.y - self.player_radius

        # Boundary checks
        self.player_pos.x = max(self.player_radius, min(self.player_pos.x, self.WIDTH - self.player_radius))
        self.player_pos.y = max(self.player_radius, min(self.player_pos.y, self.HEIGHT - self.player_radius))

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy["pos"].distance_to(enemy["end"]) < enemy["speed"]:
                enemy["start"], enemy["end"] = enemy["end"], enemy["start"]
                enemy["direction_vec"] = (enemy["end"] - enemy["start"]).normalize()
            enemy["pos"] += enemy["direction_vec"] * enemy["speed"]

            # Sonar
            enemy["sonar_timer"] += 1
            if enemy["sonar_timer"] >= enemy["sonar_cooldown"]:
                enemy["sonar_timer"] = 0
                self.sonar_pings.append({
                    "pos": pygame.Vector2(enemy["pos"]),
                    "radius": 0,
                    "max_radius": enemy["sonar_range"],
                    "life": 60})

    def _update_effects(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

        self.sonar_pings = [s for s in self.sonar_pings if s["life"] > 0]
        for s in self.sonar_pings:
            s["radius"] += s["max_radius"] / 60
            s["life"] -= 1

    def _check_artifact_collection(self):
        for artifact in self.artifacts:
            if not artifact["collected"] and self.player_pos.distance_to(artifact["pos"]) < self.player_radius + artifact["radius"]:
                artifact["collected"] = True
                self.artifacts_collected_count += 1
                return True
        return False

    def _check_detection(self):
        for enemy in self.enemies:
            # Direct collision
            if self.player_pos.distance_to(enemy["pos"]) < self.player_radius + enemy["radius"]:
                return True
            # Vision cone
            dist = self.player_pos.distance_to(enemy["pos"])
            if dist < enemy["vision_range"]:
                to_player_vec = self.player_pos - enemy["pos"]
                if to_player_vec.length() > 0:
                    to_player = to_player_vec.normalize()
                    angle_to_player = math.degrees(math.acos(enemy["direction_vec"].dot(to_player)))
                    if angle_to_player < enemy["vision_angle"] / 2:
                        return True
        # Sonar pings
        for ping in self.sonar_pings:
            if self.player_pos.distance_to(ping["pos"]) < ping["radius"]:
                return True
        return False

    def _calculate_continuous_reward(self):
        reward = -0.01 # Small penalty for each step to encourage speed

        # Distance to closest artifact reward
        uncollected_artifacts = [a["pos"] for a in self.artifacts if not a["collected"]]
        if uncollected_artifacts:
            new_dist = min(self.player_pos.distance_to(pos) for pos in uncollected_artifacts)
            if new_dist < self.dist_to_closest_artifact:
                reward += 0.1
            self.dist_to_closest_artifact = new_dist

        # Detection level penalty
        min_enemy_dist = float('inf')
        for enemy in self.enemies:
            min_enemy_dist = min(min_enemy_dist, self.player_pos.distance_to(enemy["pos"]))

        new_detection_level = 0
        if min_enemy_dist < 150: # Only calculate if somewhat close
            new_detection_level = max(0, 1 - (min_enemy_dist / 150))

        if new_detection_level > self.detection_level:
            reward -= 0.5
        self.detection_level = new_detection_level

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "artifacts_collected": self.artifacts_collected_count,
            "win": self.win
        }

    def render(self):
        return self._get_observation()

    def _render_game(self):
        # Walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Exit
        if self.artifacts_collected_count == len(self.artifacts):
            self._draw_glow_circle(self.exit_pos, self.exit_radius, self.COLOR_EXIT)

        # Teleporters
        for i, tp_pos in enumerate(self.teleporters):
            color = self.COLOR_PLAYER if i == self.current_teleport_index else self.COLOR_TELEPORTER
            self._draw_glow_circle(tp_pos, 10, color)
            pygame.gfxdraw.filled_circle(self.screen, int(tp_pos.x), int(tp_pos.y), 5, color)

        # Artifacts
        for artifact in self.artifacts:
            if not artifact["collected"]:
                self._draw_glow_circle(artifact["pos"], artifact["radius"] + 5, self.COLOR_ARTIFACT)
                pygame.draw.circle(self.screen, self.COLOR_ARTIFACT, (int(artifact["pos"].x), int(artifact["pos"].y)), artifact["radius"])

        # Sonar Pings
        for ping in self.sonar_pings:
            alpha = int(100 * (ping["life"] / 60))
            if alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, int(ping["pos"].x), int(ping["pos"].y), int(ping["radius"]), self.COLOR_ENEMY_VISION[:3] + (alpha,))

        # Enemies
        for enemy in self.enemies:
            # Vision cone
            self._draw_vision_cone(enemy)
            # Body
            self._draw_glow_circle(enemy["pos"], enemy["radius"] + 5, self.COLOR_ENEMY)
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, (int(enemy["pos"].x), int(enemy["pos"].y)), enemy["radius"])

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20))
            color = self.COLOR_PARTICLE + (alpha,)
            pygame.draw.circle(self.screen, color, (int(p["pos"].x), int(p["pos"].y)), int(p["radius"]))

        # Player
        self._draw_glow_circle(self.player_pos, self.player_radius + 10, self.COLOR_PLAYER)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (int(self.player_pos.x), int(self.player_pos.y)), self.player_radius)

        # Detection Meter
        if self.detection_level > 0.01:
            det_radius = self.player_radius + 5
            det_angle = -90 + 360 * self.detection_level
            pygame.draw.arc(self.screen, self.COLOR_ENEMY, (self.player_pos.x - det_radius, self.player_pos.y - det_radius, det_radius*2, det_radius*2), math.radians(-90), math.radians(det_angle), 2)


    def _draw_vision_cone(self, enemy):
        p1 = enemy["pos"]
        angle_rad = math.radians(enemy["vision_angle"] / 2)
        dir_angle = math.atan2(enemy["direction_vec"].y, enemy["direction_vec"].x)

        p2 = p1 + pygame.Vector2(math.cos(dir_angle - angle_rad), math.sin(dir_angle - angle_rad)) * enemy["vision_range"]
        p3 = p1 + pygame.Vector2(math.cos(dir_angle + angle_rad), math.sin(dir_angle + angle_rad)) * enemy["vision_range"]

        points = [(int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (int(p3.x), int(p3.y))]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY_VISION)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY_VISION)

    def _draw_glow_circle(self, pos, radius, color):
        for i in range(radius, 0, -2):
            alpha = int(color[3] if len(color) == 4 else 50 * (1 - i / radius))
            if alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), i, color[:3] + (alpha,))

    def _render_ui(self):
        # Artifacts collected
        artifact_text = f"Artifacts: {self.artifacts_collected_count} / {len(self.artifacts)}"
        text_surf = self.font_small.render(artifact_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (20, 20))

        # Game Over / Win message
        if self.game_over:
            msg = "MISSION COMPLETE" if self.win else "DETECTED"
            color = self.COLOR_EXIT if self.win else self.COLOR_ENEMY
            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="human_playable")
    obs, info = env.reset()

    # Override Pygame screen for direct display
    env.screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Underwater Temple")

    terminated = False
    total_reward = 0

    # Action state
    movement = 0 # 0: none, 1: up, 2: down, 3: left, 4: right
    space_held = 0 # 0: released, 1: held
    shift_held = 0 # 0: released, 1: held

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset")
    print("Q: Quit")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
                if event.key == pygame.K_SPACE: space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        else: movement = 0

        if not terminated:
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # This part is just for human play, not part of the env
        env.screen.fill(GameEnv.COLOR_BG)
        env._render_game()
        env._render_ui()

        # Display total reward for the human player
        reward_text = f"Total Reward: {total_reward:.2f}"
        reward_surf = env.font_small.render(reward_text, True, GameEnv.COLOR_TEXT)
        env.screen.blit(reward_surf, (env.WIDTH - reward_surf.get_width() - 20, 20))

        pygame.display.flip()
        env.clock.tick(GameEnv.FPS)