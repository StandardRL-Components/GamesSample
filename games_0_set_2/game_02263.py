
# Generated: 2025-08-28T04:17:50.337393
# Source Brief: brief_02263.md
# Brief Index: 2263

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place a block. Shift to cycle block types."
    )

    game_description = (
        "Defend your base from descending enemies by placing defensive blocks in their path. Survive 5 waves to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_s = pygame.font.Font(None, 24)
            self.font_m = pygame.font.Font(None, 36)
            self.font_l = pygame.font.Font(None, 72)
        except pygame.error:
            self.font_s = pygame.font.SysFont("sans", 24)
            self.font_m = pygame.font.SysFont("sans", 36)
            self.font_l = pygame.font.SysFont("sans", 72)


        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_BASE = (0, 200, 100)
        self.COLOR_BASE_BORDER = (0, 255, 120)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_ENEMY_BORDER = (255, 100, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_GREEN = (40, 220, 110)
        self.COLOR_HEALTH_YELLOW = (230, 200, 50)
        self.COLOR_HEALTH_RED = (220, 50, 50)
        self.COLOR_CURSOR = (255, 255, 255)

        # Block types and their properties
        self.BLOCK_TYPES = [
            {"name": "Wall", "color": (80, 120, 220), "max_hp": 200, "cost": 0.01},
            {"name": "Spike", "color": (220, 180, 20), "max_hp": 50, "damage": 5, "cost": 0.02},
            {"name": "Turret", "color": (180, 80, 220), "max_hp": 100, "cost": 0.05, "fire_rate": 60, "range": 150},
        ]
        
        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = 0
        self.max_base_health = 100
        self.current_wave = 0
        self.max_waves = 5
        self.enemies = []
        self.blocks = {}
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_block_type_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.wave_transition_timer = 0
        self.base_hit_timer = 0
        self.base_rect = None
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = self.max_base_health
        self.current_wave = 0
        self.enemies = []
        self.blocks = {}
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.selected_block_type_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.wave_transition_timer = 90  # Start with a 3-second countdown
        self.base_hit_timer = 0
        
        base_grid_y = self.GRID_H - 2
        self.base_rect = pygame.Rect(0, base_grid_y * self.GRID_SIZE, self.WIDTH, self.GRID_SIZE * 2)

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Handle wave transitions
        if self.wave_transition_timer > 0:
            self.wave_transition_timer -= 1
            if self.wave_transition_timer == 0:
                self._spawn_next_wave()
        else:
            self._update_game_logic()

        # Handle player actions
        reward += self._handle_actions(action)
        
        # Update timers
        if self.base_hit_timer > 0:
            self.base_hit_timer -= 1

        # Check for game state changes and calculate rewards
        reward += self._check_game_state()

        terminated = self.game_over or self.steps >= 2500

        if terminated and not self.game_over: # Terminated due to step limit
            self.game_over = True
            reward -= 100 # Penalty for running out of time
            self.score -= 100

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Cursor Movement ---
        cursor_speed = 10
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        if movement == 2: self.cursor_pos[1] += cursor_speed
        if movement == 3: self.cursor_pos[0] -= cursor_speed
        if movement == 4: self.cursor_pos[0] += cursor_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT - self.GRID_SIZE * 3)

        # --- Cycle Block Type (on key press) ---
        if shift_held and not self.last_shift_held:
            self.selected_block_type_idx = (self.selected_block_type_idx + 1) % len(self.BLOCK_TYPES)
            # Sound: UI_Switch

        # --- Place Block (on key press) ---
        if space_held and not self.last_space_held:
            grid_x = self.cursor_pos[0] // self.GRID_SIZE
            grid_y = self.cursor_pos[1] // self.GRID_SIZE
            
            if (grid_x, grid_y) not in self.blocks and grid_y < self.GRID_H - 2:
                block_type = self.BLOCK_TYPES[self.selected_block_type_idx]
                self.blocks[(grid_x, grid_y)] = {
                    "type": block_type,
                    "hp": block_type["max_hp"],
                    "pos": (grid_x, grid_y),
                    "fire_cooldown": 0
                }
                reward -= block_type['cost']
                # Sound: Block_Place

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        return reward

    def _update_game_logic(self):
        # Update Enemies
        for enemy in list(self.enemies):
            self._update_enemy(enemy)
        
        # Update Blocks (Turrets)
        for block in self.blocks.values():
            if block["type"]["name"] == "Turret":
                self._update_turret(block)
        
        # Update Projectiles
        for proj in list(self.projectiles):
            self._update_projectile(proj)

        # Update Particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["radius"] = max(0, p["radius"] * 0.95)

    def _update_enemy(self, enemy):
        speed = 0.5 + self.current_wave * 0.1
        
        # Check for block collision
        next_pos_y = enemy["pos"][1] + speed
        grid_x = int(enemy["pos"][0] / self.GRID_SIZE)
        next_grid_y = int(next_pos_y / self.GRID_SIZE)
        
        block_below = self.blocks.get((grid_x, next_grid_y))

        if block_below:
            # Enemy interacts with block
            if block_below["type"]["name"] == "Spike":
                enemy["hp"] -= block_below["type"]["damage"]
                self._create_particles(enemy["pos"], self.COLOR_ENEMY, 5)
            block_below["hp"] -= 1
            if block_below["hp"] <= 0:
                self._create_particles(
                    [(block_below["pos"][0] + 0.5) * self.GRID_SIZE, (block_below["pos"][1] + 0.5) * self.GRID_SIZE],
                    (100, 100, 100), 15
                )
                del self.blocks[block_below["pos"]]
                # Sound: Block_Destroy
        else:
            # No block, move down
            enemy["pos"][1] = next_pos_y

        # Check if enemy reached base
        if enemy["pos"][1] >= self.base_rect.top:
            self.base_health -= 10
            self.base_hit_timer = 15 # flash for 0.5s
            self.enemies.remove(enemy)
            self._create_particles(enemy["pos"], self.COLOR_BASE, 20)
            # Sound: Base_Hit
            return

        # Check if enemy is dead
        if enemy["hp"] <= 0:
            self.enemies.remove(enemy)
            self._create_particles(enemy["pos"], self.COLOR_ENEMY, 10)
            enemy["reward_given"] = True # Mark as rewarded
            # Sound: Enemy_Explode

    def _update_turret(self, block):
        if block["fire_cooldown"] > 0:
            block["fire_cooldown"] -= 1
            return
        
        turret_pos = pygame.math.Vector2((block["pos"][0] + 0.5) * self.GRID_SIZE, (block["pos"][1] + 0.5) * self.GRID_SIZE)
        
        # Find closest enemy in range
        target = None
        min_dist = block["type"]["range"] ** 2
        for enemy in self.enemies:
            enemy_pos = pygame.math.Vector2(enemy["pos"])
            dist_sq = turret_pos.distance_squared_to(enemy_pos)
            if dist_sq < min_dist:
                min_dist = dist_sq
                target = enemy
        
        if target:
            block["fire_cooldown"] = block["type"]["fire_rate"]
            target_pos = pygame.math.Vector2(target["pos"])
            direction = (target_pos - turret_pos).normalize()
            self.projectiles.append({
                "pos": list(turret_pos),
                "vel": [direction.x * 5, direction.y * 5],
                "color": (255, 200, 255)
            })
            # Sound: Turret_Fire

    def _update_projectile(self, proj):
        proj["pos"][0] += proj["vel"][0]
        proj["pos"][1] += proj["vel"][1]
        
        # Out of bounds check
        if not (0 < proj["pos"][0] < self.WIDTH and 0 < proj["pos"][1] < self.HEIGHT):
            self.projectiles.remove(proj)
            return

        # Hit detection
        for enemy in self.enemies:
            dist_sq = (proj["pos"][0] - enemy["pos"][0])**2 + (proj["pos"][1] - enemy["pos"][1])**2
            if dist_sq < (enemy["radius"] ** 2):
                enemy["hp"] -= 10
                self._create_particles(proj["pos"], proj["color"], 5)
                self.projectiles.remove(proj)
                # Sound: Projectile_Hit
                return

    def _check_game_state(self):
        reward = 0
        # Check for destroyed enemies
        newly_destroyed = [e for e in self.enemies if e["hp"] <= 0 and not e.get("reward_given")]
        if newly_destroyed:
            reward += 0.1 * len(newly_destroyed)
            for e in newly_destroyed:
                e["reward_given"] = True # Prevent double reward

        # Check for wave completion
        if not self.enemies and self.wave_transition_timer == 0 and not self.game_over:
            reward += 1
            if self.current_wave >= self.max_waves:
                self.win = True
                self.game_over = True
                reward += 100
            else:
                self.wave_transition_timer = 150 # 5-second pause

        # Check for loss condition
        if self.base_health <= 0 and not self.game_over:
            self.base_health = 0
            self.game_over = True
            reward -= 100
        
        return reward

    def _spawn_next_wave(self):
        self.current_wave += 1
        num_enemies = 3 + (self.current_wave - 1) * 2
        for _ in range(num_enemies):
            self.enemies.append({
                "pos": [self.np_random.uniform(20, self.WIDTH - 20), self.np_random.uniform(-100, -20)],
                "hp": 10 + self.current_wave * 2,
                "radius": 8,
                "reward_given": False,
            })

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30),
                "radius": self.np_random.uniform(2, 4),
                "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw base
        base_color = self.COLOR_BASE
        if self.base_hit_timer > 0 and self.steps % 4 < 2:
            base_color = (255, 255, 255) # Flash white
        pygame.draw.rect(self.screen, base_color, self.base_rect)
        pygame.draw.rect(self.screen, self.COLOR_BASE_BORDER, self.base_rect, 2)

        # Draw blocks
        for block in self.blocks.values():
            rect = pygame.Rect(block["pos"][0] * self.GRID_SIZE, block["pos"][1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            
            # Draw health indicator inside block
            hp_ratio = block["hp"] / block["type"]["max_hp"]
            inner_color = tuple(c * (0.4 + hp_ratio * 0.6) for c in block["type"]["color"])
            pygame.draw.rect(self.screen, inner_color, rect)
            pygame.draw.rect(self.screen, block["type"]["color"], rect, 2)
            if block["type"]["name"] == "Spike":
                p = block["pos"]
                gs = self.GRID_SIZE
                c = block["type"]["color"]
                pygame.draw.line(self.screen, c, (p[0]*gs, p[1]*gs), ((p[0]+1)*gs, (p[1]+1)*gs))
                pygame.draw.line(self.screen, c, ((p[0]+1)*gs, p[1]*gs), (p[0]*gs, (p[1]+1)*gs))


        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], enemy["radius"], self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], enemy["radius"], self.COLOR_ENEMY_BORDER)

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, proj["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, (255,255,255))


        # Draw particles
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            alpha = int(255 * (p["life"] / 30))
            color = (*p["color"], alpha)
            s = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(s, (pos[0] - p["radius"], pos[1] - p["radius"]))

        # Draw cursor
        grid_x = self.cursor_pos[0] // self.GRID_SIZE
        grid_y = self.cursor_pos[1] // self.GRID_SIZE
        cursor_rect = pygame.Rect(grid_x * self.GRID_SIZE, grid_y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        
        cursor_surface = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        block_color = self.BLOCK_TYPES[self.selected_block_type_idx]["color"]
        is_valid_pos = (grid_x, grid_y) not in self.blocks and grid_y < self.GRID_H - 2
        cursor_fill_color = (*block_color, 100) if is_valid_pos else (255, 0, 0, 100)
        cursor_border_color = (255, 255, 255) if is_valid_pos else (255, 100, 100)
        
        pygame.draw.rect(cursor_surface, cursor_fill_color, cursor_surface.get_rect())
        pygame.draw.rect(cursor_surface, cursor_border_color, cursor_surface.get_rect(), 2)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _render_ui(self):
        # Health bar
        health_ratio = self.base_health / self.max_base_health
        health_color = self.COLOR_HEALTH_GREEN
        if health_ratio < 0.6: health_color = self.COLOR_HEALTH_YELLOW
        if health_ratio < 0.3: health_color = self.COLOR_HEALTH_RED
        bar_width = 200
        health_bar_rect = pygame.Rect(self.WIDTH - bar_width - 10, 10, bar_width * health_ratio, 20)
        pygame.draw.rect(self.screen, health_color, health_bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.WIDTH - bar_width - 10, 10, bar_width, 20), 1)

        # Wave text
        wave_text = f"Wave: {self.current_wave}/{self.max_waves}"
        self._render_text(wave_text, (10, 10), self.font_m)

        # Score text
        score_text = f"Score: {int(self.score)}"
        self._render_text(score_text, (self.WIDTH // 2, self.HEIGHT - 30), self.font_m, align="center")

        # Selected block text
        block_name = self.BLOCK_TYPES[self.selected_block_type_idx]["name"]
        block_color = self.BLOCK_TYPES[self.selected_block_type_idx]["color"]
        self._render_text(f"Selected: {block_name}", (10, self.HEIGHT - 30), self.font_m, color=block_color)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_BASE_BORDER if self.win else self.COLOR_ENEMY
            self._render_text(msg, (self.WIDTH // 2, self.HEIGHT // 2 - 40), self.font_l, color=color, align="center")
            self._render_text(f"Final Score: {int(self.score)}", (self.WIDTH // 2, self.HEIGHT // 2 + 40), self.font_m, align="center")

        # Wave transition countdown
        if self.wave_transition_timer > 0 and self.current_wave < self.max_waves:
            seconds = math.ceil(self.wave_transition_timer / 30)
            text = f"Wave {self.current_wave + 1} starting in {seconds}..."
            self._render_text(text, (self.WIDTH // 2, self.HEIGHT // 2), self.font_m, align="center")


    def _render_text(self, text, pos, font, color=None, align="left"):
        if color is None: color = self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "right":
            text_rect.topright = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "wave": self.current_wave
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame loop for human play
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        # Action mapping for human input
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            # Wait a bit before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        env.clock.tick(30) # Limit to 30 FPS
        
    env.close()