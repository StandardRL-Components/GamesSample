
# Generated: 2025-08-28T06:10:24.983673
# Source Brief: brief_05814.md
# Brief Index: 5814

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place a block. Shift to cycle block types."
    )

    game_description = (
        "Defend your base from enemy waves by strategically placing defensive blocks on the grid."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 20
    CELL_SIZE = SCREEN_HEIGHT // GRID_SIZE
    GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
    GRID_HEIGHT = GRID_SIZE

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 55)
    COLOR_BASE = (0, 150, 255)
    COLOR_WALL = (0, 200, 100)
    COLOR_TURRET = (220, 180, 0)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_PROJECTILE = (255, 200, 0)
    COLOR_CURSOR = (255, 255, 255, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEALTH_BG = (80, 80, 80)
    COLOR_HEALTH_FG = (100, 255, 100)

    # Game settings
    MAX_STEPS = 5000
    TOTAL_WAVES = 20
    INITIAL_RESOURCES = 20
    WAVE_PREP_TIME = 90  # 3 seconds at 30fps
    WAVE_CLEAR_DELAY = 60 # 2 seconds

    # Block Types
    BLOCK_TYPES = {
        0: {"name": "Wall", "cost": 1, "health": 10, "color": COLOR_WALL, "type": "wall"},
        1: {"name": "Turret", "cost": 3, "health": 5, "color": COLOR_TURRET, "type": "turret", "range": 4, "cooldown": 30, "damage": 1}
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_medium = pygame.font.SysFont("sans-serif", 32)
        self.font_large = pygame.font.SysFont("sans-serif", 48)

        self.grid = None
        self.enemies = None
        self.projectiles = None
        self.particles = None
        self.base_pos = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.grid = np.full((self.GRID_WIDTH, self.GRID_HEIGHT), -1, dtype=int)
        self.block_health = {}
        self.turret_cooldowns = {}
        
        self.base_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT - 1)
        self.grid[self.base_pos] = 99 # Special ID for base

        self.enemies = []
        self.projectiles = []
        self.particles = deque()
        
        self.wave = 1
        self.resources = self.INITIAL_RESOURCES
        self.game_phase = "PRE_WAVE"
        self.phase_timer = self.WAVE_PREP_TIME
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_block_type = 0
        
        self.last_space_held = False
        self.last_shift_held = False

        self.pending_rewards = 0.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        step_reward = 0
        self.game_over = self.steps >= self.MAX_STEPS

        if not self.game_over and not self.victory:
            self._handle_input(movement, space_held, shift_held)
            self._update_game_state()
            step_reward = self._calculate_reward()
            self.score += step_reward

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        self.steps += 1
        
        terminated = self.game_over or self.victory
        
        return self._get_observation(), step_reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 2 and self.cursor_pos[1] < self.GRID_HEIGHT - 1: self.cursor_pos[1] += 1
        elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 4 and self.cursor_pos[0] < self.GRID_WIDTH - 1: self.cursor_pos[0] += 1

        # Place block
        if space_held and not self.last_space_held:
            self._place_block()

        # Cycle block type
        if shift_held and not self.last_shift_held:
            self.selected_block_type = (self.selected_block_type + 1) % len(self.BLOCK_TYPES)

    def _place_block(self):
        x, y = self.cursor_pos
        block_info = self.BLOCK_TYPES[self.selected_block_type]
        
        if self.grid[x, y] == -1 and self.resources >= block_info["cost"]:
            self.resources -= block_info["cost"]
            self.grid[x, y] = self.selected_block_type
            self.block_health[(x, y)] = block_info["health"]
            if block_info["type"] == "turret":
                self.turret_cooldowns[(x, y)] = 0
            # sfx: block_place
            self._create_particles((x + 0.5) * self.CELL_SIZE, (y + 0.5) * self.CELL_SIZE, block_info["color"], 15)
            self.pending_rewards -= 0.01

    def _update_game_state(self):
        self.phase_timer -= 1
        if self.game_phase == "PRE_WAVE" and self.phase_timer <= 0:
            self._spawn_wave()
            self.game_phase = "WAVE_ACTIVE"
        
        if self.game_phase == "WAVE_ACTIVE" and not self.enemies:
            self.pending_rewards += 1.0
            if self.wave >= self.TOTAL_WAVES:
                self.victory = True
                self.pending_rewards += 100.0
            else:
                self.wave += 1
                self.resources += 5 + self.wave # Bonus resources
                self.game_phase = "WAVE_CLEAR"
                self.phase_timer = self.WAVE_CLEAR_DELAY
        
        if self.game_phase == "WAVE_CLEAR" and self.phase_timer <= 0:
            self.game_phase = "PRE_WAVE"
            self.phase_timer = self.WAVE_PREP_TIME

        self._update_enemies()
        self._update_turrets()
        self._update_projectiles()
        self._update_particles_list()

    def _spawn_wave(self):
        num_enemies = 1 + self.wave
        enemy_health = 1 + self.wave
        enemy_speed = 0.015 + (self.wave * 0.002)
        
        spawn_points = [
            (0, self.np_random.integers(0, self.GRID_HEIGHT // 2)),
            (self.GRID_WIDTH - 1, self.np_random.integers(0, self.GRID_HEIGHT // 2))
        ]

        for i in range(num_enemies):
            spawn_pos = list(random.choice(spawn_points))
            # Stagger spawn positions slightly to prevent perfect overlap
            spawn_pos[0] += self.np_random.uniform(-0.4, 0.4)
            spawn_pos[1] += self.np_random.uniform(-0.4, 0.4)
            
            self.enemies.append({
                "pos": spawn_pos,
                "health": enemy_health,
                "max_health": enemy_health,
                "speed": enemy_speed
            })

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            grid_x, grid_y = int(enemy["pos"][0]), int(enemy["pos"][1])

            # Check for reaching base
            if grid_x == self.base_pos[0] and grid_y == self.base_pos[1]:
                self.game_over = True
                self.pending_rewards -= 100.0
                self._create_particles((self.base_pos[0] + 0.5) * self.CELL_SIZE, (self.base_pos[1] + 0.5) * self.CELL_SIZE, self.COLOR_BASE, 100)
                # sfx: base_destroyed
                return

            # Pathfinding and movement
            target_pos = [self.base_pos[0] + 0.5, self.base_pos[1] + 0.5]
            direction = [target_pos[0] - enemy["pos"][0], target_pos[1] - enemy["pos"][1]]
            dist = math.hypot(*direction)
            if dist > 0:
                direction = [d / dist for d in direction]
            
            next_pos = [enemy["pos"][0] + direction[0] * enemy["speed"], enemy["pos"][1] + direction[1] * enemy["speed"]]
            next_grid_pos = (int(next_pos[0]), int(next_pos[1]))

            # Collision with blocks
            collided = False
            if 0 <= next_grid_pos[0] < self.GRID_WIDTH and 0 <= next_grid_pos[1] < self.GRID_HEIGHT:
                if self.grid[next_grid_pos] not in [-1, 99]:
                    collided = True
                    self.block_health[next_grid_pos] -= 1
                    # sfx: enemy_hit_block
                    if self.block_health[next_grid_pos] <= 0:
                        self._destroy_block(next_grid_pos)

            if not collided:
                enemy["pos"] = next_pos


    def _destroy_block(self, pos):
        block_type = self.grid[pos]
        if block_type != -1:
            color = self.BLOCK_TYPES[block_type]["color"]
            self._create_particles((pos[0] + 0.5) * self.CELL_SIZE, (pos[1] + 0.5) * self.CELL_SIZE, color, 20)
            self.grid[pos] = -1
            del self.block_health[pos]
            if pos in self.turret_cooldowns:
                del self.turret_cooldowns[pos]
            # sfx: block_destroyed


    def _update_turrets(self):
        for pos, cd in self.turret_cooldowns.items():
            if cd > 0:
                self.turret_cooldowns[pos] -= 1
                continue
            
            turret_info = self.BLOCK_TYPES[self.grid[pos]]
            turret_pixel_pos = [(pos[0] + 0.5) * self.CELL_SIZE, (pos[1] + 0.5) * self.CELL_SIZE]

            target = None
            min_dist = turret_info["range"] * self.CELL_SIZE

            for enemy in self.enemies:
                enemy_pixel_pos = [enemy["pos"][0] * self.CELL_SIZE, enemy["pos"][1] * self.CELL_SIZE]
                dist = math.hypot(turret_pixel_pos[0] - enemy_pixel_pos[0], turret_pixel_pos[1] - enemy_pixel_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    target = enemy

            if target:
                self.turret_cooldowns[pos] = turret_info["cooldown"]
                enemy_pixel_pos = [target["pos"][0] * self.CELL_SIZE, target["pos"][1] * self.CELL_SIZE]
                direction = [enemy_pixel_pos[0] - turret_pixel_pos[0], enemy_pixel_pos[1] - turret_pixel_pos[1]]
                dist = math.hypot(*direction)
                if dist > 0:
                    direction = [d / dist for d in direction]

                self.projectiles.append({
                    "pos": list(turret_pixel_pos),
                    "vel": [d * 8 for d in direction],
                    "damage": turret_info["damage"]
                })
                # sfx: turret_fire

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]

            if not (0 < p["pos"][0] < self.SCREEN_WIDTH and 0 < p["pos"][1] < self.SCREEN_HEIGHT):
                self.projectiles.remove(p)
                continue
            
            hit = False
            for enemy in self.enemies[:]:
                enemy_pixel_pos = [enemy["pos"][0] * self.CELL_SIZE, enemy["pos"][1] * self.CELL_SIZE]
                if math.hypot(p["pos"][0] - enemy_pixel_pos[0], p["pos"][1] - enemy_pixel_pos[1]) < self.CELL_SIZE * 0.6:
                    enemy["health"] -= p["damage"]
                    self._create_particles(p["pos"][0], p["pos"][1], self.COLOR_PROJECTILE, 5)
                    # sfx: projectile_hit
                    if enemy["health"] <= 0:
                        self._create_particles(enemy_pixel_pos[0], enemy_pixel_pos[1], self.COLOR_ENEMY, 30)
                        self.enemies.remove(enemy)
                        self.pending_rewards += 0.1
                        # sfx: enemy_destroyed
                    hit = True
                    break
            if hit:
                self.projectiles.remove(p)

    def _calculate_reward(self):
        reward = self.pending_rewards
        self.pending_rewards = 0.0
        return reward

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave, "resources": self.resources}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_blocks()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        if self.game_over or self.victory or self.game_phase in ["PRE_WAVE", "WAVE_CLEAR"]:
            self._render_overlays()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_blocks(self):
        # Base
        bx, by = self.base_pos
        base_rect = pygame.Rect(bx * self.CELL_SIZE, by * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.gfxdraw.rectangle(self.screen, base_rect, (200, 220, 255))

        # Placed blocks
        for (x, y), block_id in np.ndenumerate(self.grid):
            if block_id in self.BLOCK_TYPES:
                block_info = self.BLOCK_TYPES[block_id]
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, block_info["color"], rect, border_radius=3)
                
                # Health bar
                health_ratio = self.block_health[(x,y)] / block_info["health"]
                if health_ratio < 1.0:
                    hb_bg_rect = pygame.Rect(rect.left + 2, rect.bottom - 5, self.CELL_SIZE - 4, 3)
                    hb_fg_rect = pygame.Rect(rect.left + 2, rect.bottom - 5, (self.CELL_SIZE - 4) * health_ratio, 3)
                    pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, hb_bg_rect)
                    pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, hb_fg_rect)

    def _render_enemies(self):
        for enemy in self.enemies:
            px, py = enemy["pos"][0] * self.CELL_SIZE, enemy["pos"][1] * self.CELL_SIZE
            size = self.CELL_SIZE * 0.8
            rect = pygame.Rect(px - size / 2, py - size / 2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=2)
            
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            hb_bg_rect = pygame.Rect(rect.left, rect.top - 6, size, 4)
            hb_fg_rect = pygame.Rect(rect.left, rect.top - 6, size * health_ratio, 4)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, hb_bg_rect)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, hb_fg_rect)

    def _render_projectiles(self):
        for p in self.projectiles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

    def _render_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        # Pulsing effect for the border
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        border_alpha = 150 + pulse * 105
        
        cursor_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, self.COLOR_CURSOR, cursor_surface.get_rect())
        pygame.gfxdraw.rectangle(cursor_surface, cursor_surface.get_rect(), (*self.COLOR_TEXT, border_alpha))
        self.screen.blit(cursor_surface, rect.topleft)

    def _render_ui(self):
        # Wave
        wave_text = self.font_medium.render(f"Wave: {self.wave}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 5))
        
        # Resources
        resource_text = self.font_medium.render(f"Blocks: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(resource_text, (self.SCREEN_WIDTH - resource_text.get_width() - 10, 5))

        # Selected Block
        block_info = self.BLOCK_TYPES[self.selected_block_type]
        selected_text = self.font_small.render(f"Selected: {block_info['name']} (Cost: {block_info['cost']})", True, self.COLOR_TEXT)
        text_rect = selected_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20))
        self.screen.blit(selected_text, text_rect)
        
        # Preview block
        preview_rect = pygame.Rect(text_rect.left - 25, text_rect.centery - 8, 16, 16)
        pygame.draw.rect(self.screen, block_info["color"], preview_rect, border_radius=2)

    def _render_overlays(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if self.game_over:
            text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY)
        elif self.victory:
            text = self.font_large.render("VICTORY!", True, self.COLOR_BASE)
        elif self.game_phase == "PRE_WAVE":
            text = self.font_large.render(f"WAVE {self.wave}", True, self.COLOR_TEXT)
        elif self.game_phase == "WAVE_CLEAR":
             text = self.font_large.render("WAVE CLEAR", True, self.COLOR_TEXT)
        else:
            return

        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": [x, y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles_list(self):
        for p in list(self.particles):
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95
            p["vel"][1] *= 0.95
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 20))))
            color = (*p["color"], alpha)
            size = max(1, int(p["life"] / 5))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p["pos"][0] - size), int(p["pos"][1] - size)), special_flags=pygame.BLEND_RGBA_ADD)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    game_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Fortress Defense")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Keyboard Controls to Action Mapping ---
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # none
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4

        # Space and Shift
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Display the observation from the environment ---
        # The observation is already a rendered frame
        # Pygame uses (width, height), numpy uses (height, width)
        # So we need to transpose back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Waves Survived: {info['wave']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            action.fill(0)

        clock.tick(30) # Run at 30 FPS

    env.close()