import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:43:18.360502
# Source Brief: brief_01159.md
# Brief Index: 1159
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A top-down arena shooter with an elemental twist. Annihilate waves of geometric "
        "enemies by exploiting their weaknesses to survive and achieve a high score."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to fire projectiles and "
        "shift to cycle through your unlocked elements."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (13, 2, 33) # #0D0221
    COLOR_GRID = (30, 15, 60)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_TEXT = (240, 240, 240)
    
    ELEMENTS = {
        "fire": {"color": (255, 87, 34), "weakness": "ice"},
        "ice": {"color": (3, 169, 244), "weakness": "electric"},
        "poison": {"color": (76, 175, 80), "weakness": "fire"},
        "electric": {"color": (255, 235, 59), "weakness": "poison"},
    }
    
    # Player
    PLAYER_SIZE = 20
    PLAYER_SPEED = 8.0
    PLAYER_MAX_HEALTH = 100
    
    # Projectiles
    PROJECTILE_SPEED = 15.0
    PROJECTILE_SIZE = 10
    PROJECTILE_COOLDOWN = 6 # frames
    
    # Enemies
    ENEMY_BASE_SPEED = 1.0
    ENEMY_BASE_HEALTH = 30
    ENEMY_SPAWN_BORDER = 50
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Initialize non-state variables
        self.render_mode = render_mode
        self.unlocked_elements = ["fire"]
        
        # Initialize state variables - these are reset in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.wave_clear_timer = 0
        self.level_step_counter = 0
        self.player_pos = pygame.Vector2(0, 0)
        self.player_health = 0
        self.aim_direction = pygame.Vector2(0, -1)
        self.fire_cooldown = 0
        self.current_element_index = 0
        self.last_action = [0, 0, 0]
        self.enemies = []
        self.projectiles = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.wave_clear_timer = 0
        self.level_step_counter = 0

        # Player state
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.aim_direction = pygame.Vector2(0, -1) # Start aiming up
        self.fire_cooldown = 0
        
        # Element selection
        self.unlocked_elements = ["fire"]
        self.current_element_index = 0
        
        # Action state tracking for rising edge detection
        self.last_action = [0, 0, 0]

        # Game objects
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        self.level_step_counter += 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Actions ---
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        fired_this_frame = space_pressed and not (self.last_action[1] == 1)
        cycled_this_frame = shift_pressed and not (self.last_action[2] == 1)
        
        # --- Update Game Logic ---
        self._update_player(movement, fired_this_frame)
        if cycled_this_frame:
            self._cycle_element()
            
        reward += self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- Level Progression ---
        if not self.enemies and self.wave_clear_timer == 0:
            self.wave_clear_timer = self.FPS * 2 # 2 second pause
            reward += 5.0 # Level clear reward
            self.score += 100 * self.level

        if self.wave_clear_timer > 0:
            self.wave_clear_timer -= 1
            if self.wave_clear_timer == 0:
                self.level += 1
                self.level_step_counter = 0
                if self.level % 2 == 0:
                    self._unlock_new_element()
                self._spawn_wave()
        
        # Anti-softlock mechanism
        if self.level_step_counter > self.FPS * 45: # 45 seconds per level
            for enemy in self.enemies:
                self.score += int(enemy.health)
                self._create_explosion(enemy.pos, 30, self.ELEMENTS[enemy.elemental_weakness]["color"])
            self.enemies.clear()
            self.level_step_counter = 0
            reward += 2.0 # Reward for "power-up" use

        # --- Termination ---
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        if self.player_health <= 0 and not self.game_over:
            reward = -100.0
            self.game_over = True
            self._create_explosion(self.player_pos, 100, self.COLOR_PLAYER)
        
        self.last_action = action
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, movement, fire):
        if self.game_over: return

        # Movement and Aiming
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1 # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
            self.aim_direction = move_vec.copy()

        # Clamp player to screen
        self.player_pos.x = max(self.PLAYER_SIZE/2, min(self.player_pos.x, self.SCREEN_WIDTH - self.PLAYER_SIZE/2))
        self.player_pos.y = max(self.PLAYER_SIZE/2, min(self.player_pos.y, self.SCREEN_HEIGHT - self.PLAYER_SIZE/2))

        # Firing
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
        
        if fire and self.fire_cooldown == 0:
            # sfx: player_shoot.wav
            element = self.unlocked_elements[self.current_element_index]
            projectile = {
                "pos": self.player_pos.copy(),
                "vel": self.aim_direction.copy() * self.PROJECTILE_SPEED,
                "element": element,
                "color": self.ELEMENTS[element]["color"],
                "trail": []
            }
            self.projectiles.append(projectile)
            self.fire_cooldown = self.PROJECTILE_COOLDOWN

    def _cycle_element(self):
        # sfx: element_swap.wav
        self.current_element_index = (self.current_element_index + 1) % len(self.unlocked_elements)

    def _unlock_new_element(self):
        all_elements = list(self.ELEMENTS.keys())
        for elem in all_elements:
            if elem not in self.unlocked_elements:
                self.unlocked_elements.append(elem)
                break

    def _update_projectiles(self):
        hit_reward = 0
        for p in self.projectiles[:]:
            p["pos"] += p["vel"]
            
            # Add to trail
            p["trail"].append(p["pos"].copy())
            if len(p["trail"]) > 5:
                p["trail"].pop(0)

            # Remove if off-screen
            if not (0 < p["pos"].x < self.SCREEN_WIDTH and 0 < p["pos"].y < self.SCREEN_HEIGHT):
                self.projectiles.remove(p)
                continue
            
            # Check for enemy collision
            for enemy in self.enemies[:]:
                if enemy.pos.distance_to(p["pos"]) < enemy.size:
                    damage = 20
                    is_weakness = self.ELEMENTS[enemy.elemental_weakness]["weakness"] == p["element"]
                    if is_weakness:
                        damage *= 2
                        hit_reward += 0.2 # Weakness hit reward
                    else:
                        hit_reward += 0.1 # Normal hit reward

                    enemy.health -= damage
                    self.score += damage
                    # sfx: enemy_hit.wav
                    self._create_explosion(p["pos"], 10, p["color"])
                    
                    if p in self.projectiles:
                        self.projectiles.remove(p)
                    
                    if enemy.health <= 0:
                        # sfx: enemy_explode.wav
                        self.score += 50
                        hit_reward += 1.0 # Kill reward
                        self._create_explosion(enemy.pos, 40, self.ELEMENTS[enemy.elemental_weakness]["color"])
                        self.enemies.remove(enemy)
                    break # Projectile can only hit one enemy
        return hit_reward

    def _update_enemies(self):
        for enemy in self.enemies:
            direction = (self.player_pos - enemy.pos)
            if direction.length() > 0:
                direction.normalize_ip()
            
            speed = self.ENEMY_BASE_SPEED + (self.level - 1) * 0.1
            enemy.pos += direction * speed
            enemy.angle += enemy.rotation_speed

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        if self.game_over: return reward
        
        for enemy in self.enemies:
            if self.player_pos.distance_to(enemy.pos) < self.PLAYER_SIZE / 2 + enemy.size / 2:
                # sfx: player_hurt.wav
                self.player_health -= 25
                self.player_health = max(0, self.player_health)
                self._create_explosion(self.player_pos, 20, (255, 0, 0))
                self.enemies.remove(enemy)
                reward -= 1.0 # Penalty for getting hit
                break
        return reward

    def _spawn_wave(self):
        num_enemies = 3 + self.level
        for _ in range(num_enemies):
            # Spawn on border
            side = self.np_random.integers(4)
            if side == 0: # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ENEMY_SPAWN_BORDER)
            elif side == 1: # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ENEMY_SPAWN_BORDER)
            elif side == 2: # Left
                pos = pygame.Vector2(-self.ENEMY_SPAWN_BORDER, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            else: # Right
                pos = pygame.Vector2(self.SCREEN_WIDTH + self.ENEMY_SPAWN_BORDER, self.np_random.uniform(0, self.SCREEN_HEIGHT))

            element_type = self.np_random.choice(list(self.ELEMENTS.keys()))
            max_health = self.ENEMY_BASE_HEALTH * (1 + (self.level - 1) * 0.1)
            
            enemy = type('Enemy', (object,), {
                'pos': pos,
                'health': max_health,
                'max_health': max_health,
                'elemental_weakness': element_type,
                'shape_sides': self.np_random.choice([3, 4, 5, 6]),
                'size': self.np_random.uniform(15, 25),
                'angle': self.np_random.uniform(0, 360),
                'rotation_speed': self.np_random.uniform(-2, 2)
            })()
            self.enemies.append(enemy)

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": color,
                "size": self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_enemies()
        self._render_projectiles()
        if not self.game_over:
            self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30.0))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (int(p["pos"].x - p["size"]), int(p["pos"].y - p["size"])), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_player(self):
        # Glow effect
        for i in range(4, 0, -1):
            alpha = 80 - i * 15
            size = self.PLAYER_SIZE + i * 4
            glow_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER_GLOW, alpha), glow_surf.get_rect(), border_radius=4)
            self.screen.blit(glow_surf, (int(self.player_pos.x - size/2), int(self.player_pos.y - size/2)), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Player body
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Aim indicator
        aim_end = self.player_pos + self.aim_direction * 30
        pygame.draw.line(self.screen, self.COLOR_PLAYER, self.player_pos, aim_end, 2)

    def _render_projectiles(self):
        for p in self.projectiles:
            # Trail
            for i, trail_pos in enumerate(p["trail"]):
                alpha = int(150 * (i / len(p["trail"])))
                size = int(self.PROJECTILE_SIZE * 0.5 * (i / len(p["trail"])))
                if size > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(trail_pos.x), int(trail_pos.y), size, (*p["color"], alpha))

            # Projectile body
            pygame.draw.rect(self.screen, p["color"], (int(p["pos"].x - self.PROJECTILE_SIZE/2), int(p["pos"].y - self.PROJECTILE_SIZE/2), self.PROJECTILE_SIZE, self.PROJECTILE_SIZE), border_radius=2)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (int(p["pos"].x - self.PROJECTILE_SIZE/2), int(p["pos"].y - self.PROJECTILE_SIZE/2), self.PROJECTILE_SIZE, self.PROJECTILE_SIZE), 1, border_radius=2)

    def _render_enemies(self):
        for enemy in self.enemies:
            color = self.ELEMENTS[enemy.elemental_weakness]["color"]
            self._draw_polygon(self.screen, color, enemy.shape_sides, enemy.pos, enemy.size, enemy.angle)
            
            # Health bar
            if enemy.health < enemy.max_health:
                bar_width = enemy.size * 1.5
                bar_height = 5
                bar_x = enemy.pos.x - bar_width / 2
                bar_y = enemy.pos.y - enemy.size - 10
                health_ratio = max(0, enemy.health / enemy.max_health)
                
                pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
                pygame.draw.rect(self.screen, color, (bar_x, bar_y, bar_width * health_ratio, bar_height))

    def _draw_polygon(self, surface, color, num_sides, center_pos, radius, angle_deg):
        points = []
        for i in range(num_sides):
            angle_rad = math.radians(angle_deg) + (i * 2 * math.pi / num_sides)
            x = center_pos.x + radius * math.cos(angle_rad)
            y = center_pos.y + radius * math.sin(angle_rad)
            points.append((int(x), int(y)))
        
        if len(points) > 1:
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Level
        level_text = self.font_small.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))
        
        # Health Bar
        health_bar_width = 200
        health_bar_height = 20
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        
        pygame.draw.rect(self.screen, (50, 0, 0), (10, self.SCREEN_HEIGHT - 30, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, (255, 0, 0), (10, self.SCREEN_HEIGHT - 30, health_bar_width * health_ratio, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, self.SCREEN_HEIGHT - 30, health_bar_width, health_bar_height), 1)

        # Element Display
        elem_x = self.SCREEN_WIDTH - 200
        elem_y = self.SCREEN_HEIGHT - 35
        for i, elem_name in enumerate(self.unlocked_elements):
            color = self.ELEMENTS[elem_name]["color"]
            is_selected = i == self.current_element_index
            
            rect = pygame.Rect(elem_x + i * 35, elem_y, 30, 30)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            if is_selected:
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, 3, border_radius=4)

        # Game Over Text
        if self.game_over:
            text = self.font_large.render("GAME OVER", True, (255, 0, 0))
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "player_health": self.player_health,
            "enemies_remaining": len(self.enemies),
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # Un-dummy the video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    
    # --- Manual Play ---
    obs, info = env.reset()
    
    pygame.display.set_caption("Geometric Annihilation")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Track rising edge for actions
    last_keys = pygame.key.get_pressed()
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- ENV RESET ---")
        
        # --- Action Polling ---
        keys = pygame.key.get_pressed()
        
        movement_action = 0 # No-op
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement_action = 4
        
        # Use rising edge for space and shift as per game logic
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 0

        action = [movement_action, space_action, shift_action]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering for manual play ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            # Wait for 'r' to reset
        
        last_keys = keys
        clock.tick(GameEnv.FPS)
        
    env.close()