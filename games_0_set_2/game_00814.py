
# Generated: 2025-08-27T14:52:56.963329
# Source Brief: brief_00814.md
# Brief Index: 814

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Arrow keys to move cursor. Space to place selected block. Shift to cycle block types. "
        "Perform a no-op (release all keys) to start the wave."
    )

    game_description = (
        "Defend your fortress from waves of enemies by strategically placing defensive blocks on a grid. "
        "Survive 20 waves to win."
    )

    auto_advance = False

    # --- Constants ---
    # Game Phases
    PHASE_PLACEMENT = 0
    PHASE_ATTACK = 1

    # Grid and Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 20
    GRID_ROWS = 20
    GRID_AREA_WIDTH = 400
    CELL_SIZE = GRID_AREA_WIDTH // GRID_COLS
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    GRID_OFFSET_Y = 0

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (45, 50, 62)
    COLOR_FORTRESS = (66, 179, 104)
    COLOR_FORTRESS_DMG = (255, 100, 100)
    COLOR_ENEMY = (217, 74, 74)
    COLOR_ENEMY_HEALTH = (100, 200, 100)
    COLOR_PROJECTILE = (66, 161, 217)
    COLOR_CURSOR = (230, 230, 80, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BG = (35, 38, 48)
    
    BLOCK_TYPES = {
        1: {"name": "WALL", "color": (130, 140, 160), "max_health": 20},
        2: {"name": "TURRET", "color": (66, 179, 104), "max_health": 10, "fire_rate": 30, "range": 4.5},
        3: {"name": "SLOWER", "color": (66, 161, 217), "max_health": 10, "slow_factor": 0.5, "range": 2.5}
    }

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
        self.font_s = pygame.font.SysFont("Consolas", 16)
        self.font_m = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 40, bold=True)
        
        self.render_mode = render_mode
        self.game_over_message = ""

        # These must be initialized here for the validation check
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game State
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = self.PHASE_PLACEMENT
        
        # Fortress
        self.fortress_health = 100
        self.fortress_max_health = 100
        self.fortress_cells = {(self.GRID_COLS - 1, y) for y in range(6, 14)}

        # Wave
        self.wave = 1
        self.max_waves = 20

        # Player Controls
        self.cursor_pos = [self.GRID_COLS // 2 - 2, self.GRID_ROWS // 2]
        self.selected_block_idx = 0
        self.available_blocks = self._get_blocks_for_wave()
        self.prev_space_held = False
        self.prev_shift_held = False

        # Entities
        self.grid = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=np.int32)
        self.block_healths = {} # (x, y) -> health
        self.turret_cooldowns = {} # (x, y) -> cooldown
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if self.game_phase == self.PHASE_PLACEMENT:
            placement_reward, start_wave = self._handle_placement_phase(action)
            reward += placement_reward
            if start_wave:
                self.game_phase = self.PHASE_ATTACK
                wave_reward, terminated = self._run_attack_phase()
                reward += wave_reward
                if not terminated:
                    self.wave += 1
                    if self.wave > self.max_waves:
                        terminated = True
                        reward += 100 # Win bonus
                        self.game_over = True
                        self.game_over_message = "VICTORY!"
                    else:
                        self.game_phase = self.PHASE_PLACEMENT
                        self.available_blocks = self._get_blocks_for_wave()
        
        self.steps += 1
        if self.steps >= 2000: # Max episode length
            terminated = True
            
        if self.fortress_health <= 0 and not self.game_over:
            terminated = True
            reward -= 100 # Loss penalty
            self.game_over = True
            self.game_over_message = "FORTRESS DESTROYED"

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_placement_phase(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        start_wave = False
        
        # Movement
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 2) # Cannot build on last col
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # Cycle block type
        block_types_available = list(self.BLOCK_TYPES.keys())
        if shift_held and not self.prev_shift_held:
            self.selected_block_idx = (self.selected_block_idx + 1) % len(block_types_available)
        
        # Place block
        if space_held and not self.prev_space_held:
            x, y = self.cursor_pos
            block_id = block_types_available[self.selected_block_idx]
            if self.available_blocks[block_id] > 0 and self.grid[x, y] == 0:
                self.grid[x, y] = block_id
                self.available_blocks[block_id] -= 1
                self.block_healths[(x, y)] = self.BLOCK_TYPES[block_id]["max_health"]
                if block_id == 2: # Turret
                    self.turret_cooldowns[(x, y)] = 0
                self._create_particles(x, y, self.BLOCK_TYPES[block_id]["color"], 10, 0.1)

        # Start wave condition
        total_blocks_left = sum(self.available_blocks.values())
        if (movement == 0 and not space_held and not shift_held) or total_blocks_left == 0:
            start_wave = True

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return 0, start_wave

    def _run_attack_phase(self):
        self._spawn_enemies()
        wave_reward = 0
        terminated = False
        
        attack_steps = 0
        while self.enemies and self.fortress_health > 0:
            attack_steps += 1
            # Update game logic by one tick
            r = self._update_attack_tick()
            wave_reward += r
            if attack_steps > 800: # Failsafe timeout
                break

        if self.fortress_health <= 0:
            terminated = True
        elif not self.enemies: # Wave cleared
            wave_reward += 1.0

        # Clear remaining projectiles
        self.projectiles.clear()
        return wave_reward, terminated

    def _update_attack_tick(self):
        reward = 0
        
        # --- Update Turrets ---
        for (tx, ty), block_id in np.ndenumerate(self.grid):
            if block_id == 2: # Turret
                cooldown = self.turret_cooldowns.get((tx, ty), 0)
                if cooldown > 0:
                    self.turret_cooldowns[(tx, ty)] -= 1
                else:
                    # Find target
                    target = self._find_turret_target(tx, ty)
                    if target:
                        self.turret_cooldowns[(tx, ty)] = self.BLOCK_TYPES[2]["fire_rate"]
                        # Fire projectile
                        start_pos = pygame.Vector2(tx + 0.5, ty + 0.5)
                        target_pos = pygame.Vector2(target['pos'][0], target['pos'][1])
                        direction = (target_pos - start_pos).normalize()
                        self.projectiles.append({
                            'pos': start_pos, 'vel': direction * 0.4, 'damage': 1
                        })
                        # Muzzle flash
                        self._create_particles(tx, ty, self.COLOR_PROJECTILE, 5, 0.2)

        # --- Update Projectiles ---
        for p in self.projectiles[:]:
            p['pos'] += p['vel']
            px, py = int(p['pos'].x), int(p['pos'].y)
            if not (0 <= px < self.GRID_COLS and 0 <= py < self.GRID_ROWS):
                self.projectiles.remove(p)
                continue
            
            hit = False
            for e in self.enemies[:]:
                ex, ey = e['pos']
                if pygame.Rect(px, py, 1, 1).colliderect(pygame.Rect(ex, ey, 1, 1)):
                    e['health'] -= p['damage']
                    self._create_particles(ex, ey, (255, 255, 255), 3, 0.1) # Hit spark
                    hit = True
                    if e['health'] <= 0:
                        reward += 0.1 # Kill reward
                        self._create_particles(ex, ey, self.COLOR_ENEMY, 20, 0.3) # Death explosion
                        self.enemies.remove(e)
                    break # Projectile hits one enemy
            if hit:
                self.projectiles.remove(p)

        # --- Update Enemies ---
        for e in self.enemies[:]:
            # Apply slowing effect
            speed_multiplier = 1.0
            for (sx, sy), block_id in np.ndenumerate(self.grid):
                if block_id == 3: # Slower
                    dist = math.hypot(e['pos'][0] - (sx + 0.5), e['pos'][1] - (sy + 0.5))
                    if dist <= self.BLOCK_TYPES[3]["range"]:
                        speed_multiplier = min(speed_multiplier, self.BLOCK_TYPES[3]["slow_factor"])

            # Move
            move_vec = self._pathfind(e)
            e['pos'] += move_vec * e['speed'] * speed_multiplier

            # Check for fortress collision
            if e['pos'].x >= self.GRID_COLS - 1:
                damage = e['health'] # Enemy deals remaining health as damage
                self.fortress_health -= damage
                reward -= damage * 0.01 # Damage penalty
                self.enemies.remove(e)
                self._create_particles(self.GRID_COLS-1, e['pos'].y, self.COLOR_FORTRESS_DMG, 30, 0.4)
                continue
        
        # --- Update Particles ---
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
                
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "fortress_health": self.fortress_health,
            "enemies_left": len(self.enemies),
            "game_phase": "placement" if self.game_phase == self.PHASE_PLACEMENT else "attack"
        }

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_COLS + 1):
            start = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_ROWS * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_ROWS + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_COLS * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw fortress
        for (fx, fy) in self.fortress_cells:
            rect = pygame.Rect(self.GRID_OFFSET_X + fx * self.CELL_SIZE, self.GRID_OFFSET_Y + fy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_FORTRESS, rect)

        # Draw blocks
        for (x, y), block_id in np.ndenumerate(self.grid):
            if block_id > 0:
                block_info = self.BLOCK_TYPES[block_id]
                rect = pygame.Rect(self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, block_info["color"], rect, border_radius=2)
                if block_id == 2: # Turret circle
                    center = (rect.centerx, rect.centery)
                    pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.CELL_SIZE // 4, (255,255,255))
                if block_id == 3: # Slower radius
                    center_px = (self.GRID_OFFSET_X + int((x + 0.5) * self.CELL_SIZE), self.GRID_OFFSET_Y + int((y + 0.5) * self.CELL_SIZE))
                    radius_px = int(block_info["range"] * self.CELL_SIZE)
                    pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], radius_px, (*block_info["color"], 30))
                    pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], radius_px, (*block_info["color"], 30))

        # Draw enemies
        for e in self.enemies:
            ex, ey = e['pos']
            rect = pygame.Rect(self.GRID_OFFSET_X + ex * self.CELL_SIZE, self.GRID_OFFSET_Y + ey * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=3)
            # Health bar
            health_pct = e['health'] / e['max_health']
            hp_bar_rect = pygame.Rect(rect.left, rect.top - 5, self.CELL_SIZE * health_pct, 3)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_HEALTH, hp_bar_rect)
        
        # Draw projectiles
        for p in self.projectiles:
            start_pos = (self.GRID_OFFSET_X + int(p['pos'].x * self.CELL_SIZE), self.GRID_OFFSET_Y + int(p['pos'].y * self.CELL_SIZE))
            end_pos_vec = p['pos'] - p['vel'] * 2 # Create a tail
            end_pos = (self.GRID_OFFSET_X + int(end_pos_vec.x * self.CELL_SIZE), self.GRID_OFFSET_Y + int(end_pos_vec.y * self.CELL_SIZE))
            pygame.draw.aaline(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 2)

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            pos = (self.GRID_OFFSET_X + int(p['pos'].x * self.CELL_SIZE), self.GRID_OFFSET_Y + int(p['pos'].y * self.CELL_SIZE))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)

        # Draw cursor
        if self.game_phase == self.PHASE_PLACEMENT:
            cursor_rect = pygame.Rect(self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE, self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill(self.COLOR_CURSOR)
            self.screen.blit(s, cursor_rect.topleft)

    def _render_ui(self):
        ui_panel_rect = pygame.Rect(self.GRID_OFFSET_X + self.GRID_AREA_WIDTH, 0, self.SCREEN_WIDTH - (self.GRID_OFFSET_X + self.GRID_AREA_WIDTH), self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_panel_rect)
        
        y_pos = 20
        # Fortress Health
        health_text = self.font_m.render(f"Fortress: {int(self.fortress_health)}%", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (ui_panel_rect.left + 20, y_pos))
        y_pos += 30
        health_bar_rect = pygame.Rect(ui_panel_rect.left + 20, y_pos, ui_panel_rect.width - 40, 15)
        health_pct = max(0, self.fortress_health / self.fortress_max_health)
        pygame.draw.rect(self.screen, self.COLOR_GRID, health_bar_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_FORTRESS, (health_bar_rect.x, health_bar_rect.y, health_bar_rect.width * health_pct, health_bar_rect.height), border_radius=3)
        y_pos += 40

        # Wave Info
        wave_text = self.font_m.render(f"Wave: {self.wave} / {self.max_waves}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (ui_panel_rect.left + 20, y_pos))
        y_pos += 40

        # Score
        score_text = self.font_m.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_panel_rect.left + 20, y_pos))
        y_pos += 60

        # Available Blocks
        if self.game_phase == self.PHASE_PLACEMENT:
            blocks_text = self.font_m.render("Blocks:", True, self.COLOR_TEXT)
            self.screen.blit(blocks_text, (ui_panel_rect.left + 20, y_pos))
            y_pos += 30
            
            block_types_available = list(self.BLOCK_TYPES.keys())
            for i, block_id in enumerate(block_types_available):
                block_info = self.BLOCK_TYPES[block_id]
                is_selected = (i == self.selected_block_idx)
                
                # Selection Highlight
                if is_selected:
                    highlight_rect = pygame.Rect(ui_panel_rect.left + 15, y_pos - 5, ui_panel_rect.width - 30, 30)
                    pygame.draw.rect(self.screen, (255,255,255,20), highlight_rect, border_radius=4)
                
                # Block Icon
                icon_rect = pygame.Rect(ui_panel_rect.left + 25, y_pos, 20, 20)
                pygame.draw.rect(self.screen, block_info["color"], icon_rect, border_radius=2)
                
                # Block Text
                block_ui_text = f"{block_info['name']}: {self.available_blocks[block_id]}"
                text_surf = self.font_s.render(block_ui_text, True, self.COLOR_TEXT)
                self.screen.blit(text_surf, (icon_rect.right + 10, y_pos))
                y_pos += 35

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text_surf = self.font_l.render(self.game_over_message, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        overlay.blit(text_surf, text_rect)
        
        self.screen.blit(overlay, (0, 0))
    
    # --- Helper Functions ---
    def _get_blocks_for_wave(self):
        return {1: 5, 2: 2, 3: 1} # Wall, Turret, Slower

    def _spawn_enemies(self):
        num_enemies = 5 + (self.wave - 1)
        speed = 0.02 + (self.wave - 1) * 0.002
        health = 5 + (self.wave - 1) * 2

        for _ in range(num_enemies):
            self.enemies.append({
                'pos': pygame.Vector2(0, self.np_random.integers(0, self.GRID_ROWS)),
                'health': health,
                'max_health': health,
                'speed': speed
            })
            
    def _pathfind(self, enemy):
        # Simple pathfinding: move towards the closest fortress cell, avoid walls
        ex, ey = enemy['pos']
        
        # Find closest fortress y
        closest_fortress_y = min([fy for _, fy in self.fortress_cells], key=lambda y_pos: abs(y_pos - ey))
        target = pygame.Vector2(self.GRID_COLS - 0.5, closest_fortress_y + 0.5)
        
        direction = (target - enemy['pos']).normalize() if (target - enemy['pos']).length() > 0 else pygame.Vector2(0,0)
        
        # Check for collisions
        next_pos_grid = (int(ex + direction.x * 0.6), int(ey + direction.y * 0.6))
        if 0 <= next_pos_grid[0] < self.GRID_COLS and 0 <= next_pos_grid[1] < self.GRID_ROWS:
            if self.grid[next_pos_grid] == 1: # Wall
                # Try moving just horizontally
                alt_dir = pygame.Vector2(direction.x, 0)
                alt_next_pos = (int(ex + alt_dir.x * 0.6), int(ey))
                if not (0 <= alt_next_pos[0] < self.GRID_COLS and self.grid[alt_next_pos] == 1):
                    return alt_dir
                # Try moving just vertically
                alt_dir = pygame.Vector2(0, direction.y)
                alt_next_pos = (int(ex), int(ey + alt_dir.y * 0.6))
                if not (0 <= alt_next_pos[1] < self.GRID_ROWS and self.grid[alt_next_pos] == 1):
                    return alt_dir
                return pygame.Vector2(0, 0) # Blocked
        
        return direction

    def _find_turret_target(self, tx, ty):
        turret_range = self.BLOCK_TYPES[2]["range"]
        best_target = None
        min_dist = float('inf')
        for e in self.enemies:
            dist = math.hypot(e['pos'][0] - tx, e['pos'][1] - ty)
            if dist <= turret_range and dist < min_dist:
                min_dist = dist
                best_target = e
        return best_target
        
    def _create_particles(self, x, y, color, count, speed_factor):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * speed_factor
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pygame.Vector2(x + 0.5, y + 0.5),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'radius': self.np_random.uniform(1, 3),
                'color': color,
                'lifespan': lifespan,
                'max_lifespan': lifespan
            })

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Fortress")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # --- Manual Control Mapping ---
    # This is separate from the MultiDiscrete action space and is just for human play
    action = [0, 0, 0] # no-op, release, release
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # Get key presses for manual control
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
            
        # Space and Shift
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # If in attack phase, the step function runs the whole wave, so we just need to render
        # If in placement phase, we can control the speed of actions
        if info['game_phase'] == 'placement':
            clock.tick(15) # Limit placement speed
        else:
            # During attack phase visualization, we can just let it render as fast as possible
            # since the simulation is already complete.
            pass

    pygame.quit()