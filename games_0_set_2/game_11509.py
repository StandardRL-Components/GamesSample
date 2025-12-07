import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A tower defense game where the agent manages three automated turrets.
    The goal is to defend against waves of enemies by assigning target priorities
    to the turrets to maximize a score multiplier.

    - **Action Space**: MultiDiscrete([5, 2, 2])
      - `actions[0]` (Movement):
        - 1 (Up): Cycle Turret 1's target priority.
        - 2 (Down): Cycle Turret 2's target priority.
        - 3 (Left): Cycle Turret 3's target priority.
        - 0 (None), 4 (Right): No-op.
      - `actions[1]` (Space):
        - 1 (Held): If in "Action Phase", starts the wave.
        - 0 (Released): No-op.
      - `actions[2]` (Shift): No-op.

    - **Observation Space**: A 640x400 RGB image of the game state.

    - **Reward Structure**:
      - +0.01 per enemy hit.
      - +5 for clearing a wave.
      - +100 for winning the game.
      - -100 for losing the game.

    - **Termination**:
      - Win: Maintain a 5x score multiplier for 5 consecutive waves.
      - Loss: Score multiplier drops below 1.0x.
      - Max steps: 20,000 frames (roughly 11 minutes at 30 FPS).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Defend against enemy waves by assigning target priorities to three automated turrets to maximize your score."
    user_guide = "Use ↑ to cycle the left turret's target, ↓ for the middle turret, and ← for the right turret. Press space to start the wave."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 20000

    # Colors
    COLOR_BG = (15, 18, 28)
    COLOR_GRID = (30, 35, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_ACCENT = (100, 120, 255)
    COLOR_PROMPT = (255, 255, 100)

    # Enemy Types
    ENEMY_TYPE_SLOW = 0
    ENEMY_TYPE_FAST = 1
    ENEMY_TYPE_ARMORED = 2
    ENEMY_TYPES = [ENEMY_TYPE_SLOW, ENEMY_TYPE_FAST, ENEMY_TYPE_ARMORED]
    ENEMY_DATA = {
        ENEMY_TYPE_SLOW: {"color": (50, 205, 50), "hp": 10, "speed": 40, "shape": "circle"},
        ENEMY_TYPE_FAST: {"color": (255, 69, 0), "hp": 5, "speed": 80, "shape": "square"},
        ENEMY_TYPE_ARMORED: {"color": (30, 144, 255), "hp": 20, "speed": 30, "shape": "triangle"}
    }
    
    # Turret
    TURRET_FIRE_RATE = 0.5  # seconds
    TURRET_DAMAGE = 1.0
    PROJECTILE_SPEED = 500

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_prompt = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.turrets = []
        self.enemies = []
        self.projectiles = []
        self.explosions = []
        
        # Initialize state variables to be set in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.score_multiplier = 0.0
        self.consecutive_victory_waves = 0
        self.game_phase = "action" # "action" or "wave"
        
        # self.reset() is not called in __init__ as per Gymnasium standard practice

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.score_multiplier = 1.5
        self.consecutive_victory_waves = 0
        self.game_phase = "action"

        self.turrets = [
            self._create_turret(self.SCREEN_WIDTH * 0.2, self.SCREEN_HEIGHT * 0.8),
            self._create_turret(self.SCREEN_WIDTH * 0.5, self.SCREEN_HEIGHT * 0.8),
            self._create_turret(self.SCREEN_WIDTH * 0.8, self.SCREEN_HEIGHT * 0.8),
        ]
        self.enemies.clear()
        self.projectiles.clear()
        self.explosions.clear()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        truncated = False
        
        if self.game_phase == "action":
            self._handle_action_phase_input(movement)
            if space_held:
                self._start_wave()
        else: # self.game_phase == "wave"
            dt = 1.0 / self.FPS
            self._update_turrets(dt)
            reward += self._update_projectiles(dt)
            reward += self._update_enemies(dt)
            self._update_effects(dt)

            if not self.enemies and not self.projectiles:
                self._end_wave()
                reward += 5 # Wave clear bonus
                
                if self.score_multiplier >= 5.0:
                    self.consecutive_victory_waves += 1
                else:
                    self.consecutive_victory_waves = 0

        self.steps += 1
        
        # Check termination conditions
        if self.score_multiplier < 1.0:
            terminated = True
            reward -= 100 # Loss penalty
        elif self.consecutive_victory_waves >= 5:
            terminated = True
            reward += 100 # Victory bonus
        elif self.steps >= self.MAX_STEPS:
            truncated = True
        
        self.game_over = terminated or truncated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_action_phase_input(self, movement_action):
        turret_idx = -1
        if movement_action == 1: turret_idx = 0 # Up
        elif movement_action == 2: turret_idx = 1 # Down
        elif movement_action == 3: turret_idx = 2 # Left
        
        if turret_idx != -1:
            turret = self.turrets[turret_idx]
            turret["target_type"] = (turret["target_type"] + 1) % len(self.ENEMY_TYPES)
            # Sound: "UI click"

    def _start_wave(self):
        self.game_phase = "wave"
        self.wave_number += 1
        self._spawn_enemies()
        # Sound: "Wave start"

    def _end_wave(self):
        self.game_phase = "action"
        self.score_multiplier *= 0.95 # Multiplier decay between waves
        self.score_multiplier = max(1.0, self.score_multiplier)
        # Sound: "Wave complete"

    def _create_turret(self, x, y):
        return {
            "pos": pygame.Vector2(x, y),
            "target_type": self.ENEMY_TYPE_SLOW,
            "fire_cooldown": 0,
            "angle": -90, # Pointing up
        }

    def _spawn_enemies(self):
        wave = self.wave_number
        num_enemies = 20
        
        # Calculate enemy composition
        fast_chance = min(0.25, (wave - 1) * 0.05)
        armored_chance = min(0.25, max(0, wave - 2) * 0.05)
        slow_chance = 1.0 - fast_chance - armored_chance
        
        for _ in range(num_enemies):
            spawn_x = self.np_random.uniform(20, self.SCREEN_WIDTH - 20)
            spawn_y = self.np_random.uniform(-150, -20)
            
            r = self.np_random.random()
            if r < slow_chance:
                enemy_type = self.ENEMY_TYPE_SLOW
            elif r < slow_chance + fast_chance:
                enemy_type = self.ENEMY_TYPE_FAST
            else:
                enemy_type = self.ENEMY_TYPE_ARMORED
            
            data = self.ENEMY_DATA[enemy_type]
            self.enemies.append({
                "pos": pygame.Vector2(spawn_x, spawn_y),
                "hp": data["hp"],
                "max_hp": data["hp"],
                "type": enemy_type,
                "speed": data["speed"],
                "color": data["color"],
                "shape": data["shape"],
            })

    def _update_turrets(self, dt):
        for turret in self.turrets:
            turret["fire_cooldown"] = max(0, turret["fire_cooldown"] - dt)
            
            target = self._find_target_for_turret(turret)
            if target:
                # Aim at target
                direction = target["pos"] - turret["pos"]
                turret["angle"] = direction.angle_to(pygame.Vector2(1, 0))
                
                if turret["fire_cooldown"] <= 0:
                    self._fire_projectile(turret, target)
                    turret["fire_cooldown"] = self.TURRET_FIRE_RATE
                    # Sound: "Turret fire"

    def _find_target_for_turret(self, turret):
        # Prioritize assigned target type
        preferred_targets = [e for e in self.enemies if e["type"] == turret["target_type"]]
        other_targets = [e for e in self.enemies if e["type"] != turret["target_type"]]
        
        # Function to find closest enemy in a list
        def get_closest(target_list):
            closest_enemy = None
            min_dist_sq = float('inf')
            for enemy in target_list:
                dist_sq = turret["pos"].distance_squared_to(enemy["pos"])
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_enemy = enemy
            return closest_enemy
        
        # Find closest preferred, if available. Otherwise, find closest other.
        target = get_closest(preferred_targets)
        if not target:
            target = get_closest(other_targets)
            
        return target

    def _fire_projectile(self, turret, target):
        direction = (target["pos"] - turret["pos"]).normalize()
        self.projectiles.append({
            "pos": pygame.Vector2(turret["pos"]),
            "vel": direction * self.PROJECTILE_SPEED,
            "angle": turret["angle"],
        })

    def _update_projectiles(self, dt):
        hit_reward = 0
        projectiles_to_remove = []
        for proj in self.projectiles:
            proj["pos"] += proj["vel"] * dt
            
            # Boundary check
            if not (0 <= proj["pos"].x <= self.SCREEN_WIDTH and 0 <= proj["pos"].y <= self.SCREEN_HEIGHT):
                projectiles_to_remove.append(proj)
                continue
            
            # Collision check
            for enemy in self.enemies:
                if proj["pos"].distance_to(enemy["pos"]) < 10:
                    damage = self.TURRET_DAMAGE
                    if enemy["type"] == self.ENEMY_TYPE_ARMORED:
                        damage *= 0.5
                    enemy["hp"] -= damage
                    
                    self._create_explosion(proj["pos"], enemy["color"])
                    hit_reward += 0.01
                    projectiles_to_remove.append(proj)
                    # Sound: "Hit impact"
                    break
        
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        return hit_reward

    def _update_enemies(self, dt):
        leaked_penalty = 0
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy["hp"] <= 0:
                enemies_to_remove.append(enemy)
                self.score += 10
                self._create_explosion(enemy["pos"], enemy["color"], scale=1.5)
                # Sound: "Enemy explosion"
                continue

            enemy["pos"].y += enemy["speed"] * dt
            
            if enemy["pos"].y > self.SCREEN_HEIGHT + 10:
                enemies_to_remove.append(enemy)
                self.score_multiplier -= 0.1
                # Sound: "Leak alert"
        
        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        return leaked_penalty

    def _update_effects(self, dt):
        explosions_to_remove = []
        for exp in self.explosions:
            exp["life"] -= dt
            exp["radius"] = exp["max_radius"] * (1 - (exp["life"] / exp["max_life"]))
            if exp["life"] <= 0:
                explosions_to_remove.append(exp)
        self.explosions = [e for e in self.explosions if e not in explosions_to_remove]

    def _create_explosion(self, pos, color, scale=1.0):
        self.explosions.append({
            "pos": pos,
            "color": color,
            "radius": 0,
            "max_radius": 20 * scale,
            "life": 0.25,
            "max_life": 0.25,
        })
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "multiplier": self.score_multiplier,
            "phase": self.game_phase,
            "consecutive_wins": self.consecutive_victory_waves,
        }

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
    def _render_game(self):
        for enemy in self.enemies:
            self._render_enemy(enemy)
        for proj in self.projectiles:
            self._render_projectile(proj)
        for turret in self.turrets:
            self._render_turret(turret)
        for exp in self.explosions:
            self._render_explosion(exp)
        self._render_targeting_ui()

    def _render_turret(self, turret):
        pos = (int(turret["pos"].x), int(turret["pos"].y))
        
        # Base
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, (100, 100, 120))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15, (60, 60, 70))
        
        # Barrel
        barrel_poly = self._create_rotated_rect(turret["pos"], 30, 8, turret["angle"])
        pygame.gfxdraw.aapolygon(self.screen, barrel_poly, self.ENEMY_DATA[turret["target_type"]]["color"])
        pygame.gfxdraw.filled_polygon(self.screen, barrel_poly, self.ENEMY_DATA[turret["target_type"]]["color"])

    def _create_rotated_rect(self, center_pos, width, height, angle):
        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        hw, hh = width / 2, height / 2
        
        points = [
            (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
        ]
        
        rotated_points = []
        for x, y in points:
            rx = x * cos_a - y * sin_a + center_pos.x
            ry = x * sin_a + y * cos_a + center_pos.y
            rotated_points.append((int(rx), int(ry)))
            
        return rotated_points

    def _render_enemy(self, enemy):
        pos = (int(enemy["pos"].x), int(enemy["pos"].y))
        
        # Draw shape
        if enemy["shape"] == "circle":
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, enemy["color"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, enemy["color"])
        elif enemy["shape"] == "square":
            rect = pygame.Rect(pos[0] - 7, pos[1] - 7, 14, 14)
            pygame.draw.rect(self.screen, enemy["color"], rect)
        elif enemy["shape"] == "triangle":
            points = [
                (pos[0], pos[1] - 8),
                (pos[0] - 8, pos[1] + 6),
                (pos[0] + 8, pos[1] + 6)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, enemy["color"])
            pygame.gfxdraw.filled_polygon(self.screen, points, enemy["color"])
            
        # Health bar
        if enemy["hp"] < enemy["max_hp"]:
            bar_w, bar_h = 20, 4
            bar_x, bar_y = pos[0] - bar_w / 2, pos[1] - 20
            health_ratio = enemy["hp"] / enemy["max_hp"]
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (255, 255, 0), (bar_x, bar_y, bar_w * health_ratio, bar_h))

    def _render_projectile(self, proj):
        start_pos = proj["pos"]
        end_pos = start_pos - proj["vel"].normalize() * 10
        pygame.draw.aaline(self.screen, (255, 255, 100), start_pos, end_pos, 2)

    def _render_explosion(self, exp):
        alpha = int(255 * (exp["life"] / exp["max_life"]))
        # Create a temporary surface for alpha blending
        temp_surf = pygame.Surface((exp["max_radius"]*2, exp["max_radius"]*2), pygame.SRCALPHA)
        color = exp["color"]
        pos = (int(exp["max_radius"]), int(exp["max_radius"]))
        radius = int(exp["radius"])
        if radius > 0:
            pygame.gfxdraw.filled_circle(temp_surf, pos[0], pos[1], radius, color + (alpha,))
            pygame.gfxdraw.aacircle(temp_surf, pos[0], pos[1], radius, color + (alpha,))
        
        self.screen.blit(temp_surf, (int(exp["pos"].x - exp["max_radius"]), int(exp["pos"].y - exp["max_radius"])))


    def _render_targeting_ui(self):
        icon_y = 30
        icons = {}
        for e_type in self.ENEMY_TYPES:
            data = self.ENEMY_DATA[e_type]
            x_pos = self.SCREEN_WIDTH * (0.3 + 0.2 * e_type)
            icons[e_type] = pygame.Vector2(x_pos, icon_y)
            
            # Draw icon shape
            if data["shape"] == "circle":
                pygame.gfxdraw.filled_circle(self.screen, int(x_pos), int(icon_y), 10, data["color"])
            elif data["shape"] == "square":
                pygame.draw.rect(self.screen, data["color"], (x_pos - 8, icon_y - 8, 16, 16))
            elif data["shape"] == "triangle":
                points = [(x_pos, icon_y - 8), (x_pos - 8, icon_y + 6), (x_pos + 8, icon_y + 6)]
                pygame.gfxdraw.filled_polygon(self.screen, points, data["color"])

        # Draw lines from turrets to icons
        for turret in self.turrets:
            start_pos = turret["pos"]
            end_pos = icons[turret["target_type"]]
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_pos, end_pos)

    def _render_ui(self):
        # Wave and Multiplier
        wave_text = self.font_ui.render(f"WAVE: {self.wave_number}", True, self.COLOR_UI_TEXT)
        multi_text = self.font_ui.render(f"MULTIPLIER: {self.score_multiplier:.2f}x", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, self.SCREEN_HEIGHT - 30))
        self.screen.blit(multi_text, (self.SCREEN_WIDTH - multi_text.get_width() - 10, self.SCREEN_HEIGHT - 30))
        
        # Victory progress
        if self.consecutive_victory_waves > 0:
            victory_text = self.font_ui.render(f"STREAK: {self.consecutive_victory_waves}/5", True, self.COLOR_UI_ACCENT)
            self.screen.blit(victory_text, (self.SCREEN_WIDTH / 2 - victory_text.get_width()/2, self.SCREEN_HEIGHT - 30))

        # Phase prompt
        if self.game_phase == "action" and self.steps > 0:
            prompt_text = self.font_prompt.render("SET TARGETS - PRESS SPACE TO START WAVE", True, self.COLOR_PROMPT)
            self.screen.blit(prompt_text, (self.SCREEN_WIDTH / 2 - prompt_text.get_width() / 2, self.SCREEN_HEIGHT / 2))
        
        if self.game_over:
            end_text_str = "VICTORY!" if self.consecutive_victory_waves >= 5 else "GAME OVER"
            end_color = (0, 255, 0) if self.consecutive_victory_waves >= 5 else (255, 0, 0)
            end_text = self.font_prompt.render(end_text_str, True, end_color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH / 2 - end_text.get_width() / 2, self.SCREEN_HEIGHT / 2 - 50))


    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping keyboard keys to actions
    # W/S/A for turrets, SPACE to start wave
    key_to_action = {
        pygame.K_w: 1,
        pygame.K_s: 2,
        pygame.K_a: 3,
    }

    # Use a separate Pygame window for human rendering
    human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense AI Environment")
    clock = pygame.time.Clock()

    while not done:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action[0] = key_to_action[event.key]
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
        
        # If not pressing a specific key, check for held keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Multiplier: {info['multiplier']:.2f}, Phase: {info['phase']}")

        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    print("Game Over!")
    print(f"Final Info: {info}")
    env.close()