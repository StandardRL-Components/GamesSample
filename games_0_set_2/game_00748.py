
# Generated: 2025-08-27T14:39:56.407094
# Source Brief: brief_00748.md
# Brief Index: 748

        
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
        "Controls: Use arrow keys to move the placement cursor. "
        "Hold Shift to cycle through block types (Basic, Wall, Cannon). "
        "Press Space to place the selected block. Survive the waves!"
    )

    game_description = (
        "A top-down tower defense game. Strategically place blocks to build a fortress "
        "and defend your core against waves of enemies."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- System Setup ---
        self.screen_width = 640
        self.screen_height = 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.rng = np.random.default_rng()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Game Configuration ---
        self.grid_size = 20
        self.build_area_rect = pygame.Rect(140, 60, 360, 280)
        self.fortress_core_grid_pos = (
            self.build_area_rect.centerx // self.grid_size,
            self.build_area_rect.centery // self.grid_size
        )
        self.spawn_point = (20, self.screen_height // 2)

        self.block_config = {
            'basic': {'cost': 10, 'health': 50, 'color': (150, 150, 150)},
            'wall': {'cost': 25, 'health': 200, 'color': (100, 100, 120)},
            'cannon': {'cost': 50, 'health': 75, 'color': (200, 100, 100), 'range': 120, 'fire_rate': 45, 'damage': 10}
        }
        self.block_types = list(self.block_config.keys())

        # --- Colors ---
        self.COLOR_BG = (30, 35, 40)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_BUILD_AREA = (35, 40, 45)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_PROJECTILE = (255, 200, 0)
        self.COLOR_CORE = (0, 150, 255)
        self.COLOR_TEXT = (230, 230, 230)
        
        # --- Fonts ---
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_large = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_huge = pygame.font.SysFont("Consolas", 48, bold=True)

        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.last_shift_state = 0
        self.last_space_state = 0

        # --- Run initial reset and validation ---
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Game State
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = 'build' # 'build' or 'wave'
        self.wave_number = 1
        self.resources = 100
        self.fortress_max_health = 100
        self.fortress_health = self.fortress_max_health
        
        # Phase Timers
        self.build_phase_timer = 450 # 15 seconds at 30fps
        self.wave_enemies_to_spawn = 0
        self.wave_spawn_timer = 0

        # Player State
        self.cursor_grid_pos = [self.fortress_core_grid_pos[0], self.fortress_core_grid_pos[1]]
        self.selected_block_idx = 0
        self.last_shift_state = 0
        self.last_space_state = 0

        # Entities
        self.blocks = {}
        self.enemies = []
        self.projectiles = []
        self.particles = []

        # Add fortress core block
        core_block = {
            'type': 'core',
            'grid_pos': self.fortress_core_grid_pos,
            'health': self.fortress_max_health,
            'max_health': self.fortress_max_health
        }
        self.blocks[self.fortress_core_grid_pos] = core_block

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        self.game_over = self._check_termination()

        if not self.game_over:
            self._handle_input(action)
            
            if self.game_phase == 'build':
                self._update_build_phase()
            elif self.game_phase == 'wave':
                self._update_wave_phase()

            self._update_particles()
        
        self.steps += 1
        reward = self.reward_this_step
        terminated = self.game_over

        if terminated:
            if self.wave_number > 20:
                reward += 100 # Win bonus
                self.score += 10000
            elif self.fortress_health <= 0:
                reward -= 100 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_action, shift_action = action
        
        # --- Cursor Movement ---
        if movement == 1: self.cursor_grid_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_grid_pos[1] += 1 # Down
        elif movement == 3: self.cursor_grid_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_grid_pos[0] += 1 # Right

        self.cursor_grid_pos[0] = np.clip(self.cursor_grid_pos[0], self.build_area_rect.left // self.grid_size, (self.build_area_rect.right // self.grid_size) - 1)
        self.cursor_grid_pos[1] = np.clip(self.cursor_grid_pos[1], self.build_area_rect.top // self.grid_size, (self.build_area_rect.bottom // self.grid_size) - 1)

        # --- Cycle Block Type (on press) ---
        if shift_action == 1 and self.last_shift_state == 0:
            self.selected_block_idx = (self.selected_block_idx + 1) % len(self.block_types)
            # sfx: UI_CYCLE_SOUND

        # --- Place Block (on press) ---
        if space_action == 1 and self.last_space_state == 0 and self.game_phase == 'build':
            pos_tuple = tuple(self.cursor_grid_pos)
            if pos_tuple not in self.blocks:
                block_type_name = self.block_types[self.selected_block_idx]
                config = self.block_config[block_type_name]
                if self.resources >= config['cost']:
                    self.resources -= config['cost']
                    self.reward_this_step -= config['cost'] * 0.01

                    new_block = {
                        'type': block_type_name,
                        'grid_pos': pos_tuple,
                        'health': config['health'],
                        'max_health': config['health']
                    }
                    if block_type_name == 'cannon':
                        new_block['fire_cooldown'] = 0
                        new_block['target'] = None
                    
                    self.blocks[pos_tuple] = new_block
                    self._create_particles(self._grid_to_pixel(pos_tuple), config['color'], 10)
                    # sfx: BLOCK_PLACE_SOUND

        self.last_space_state = space_action
        self.last_shift_state = shift_action

    def _update_build_phase(self):
        self.build_phase_timer -= 1
        if self.build_phase_timer <= 0:
            self.game_phase = 'wave'
            self._prepare_next_wave()
            
    def _prepare_next_wave(self):
        num_enemies = 3 + (self.wave_number - 1)
        self.wave_enemies_to_spawn = num_enemies
        self.wave_spawn_timer = 90 # 3 seconds

    def _update_wave_phase(self):
        # Spawn enemies
        if self.wave_enemies_to_spawn > 0:
            self.wave_spawn_timer -= 1
            if self.wave_spawn_timer <= 0:
                self._spawn_enemy()
                self.wave_enemies_to_spawn -= 1
                self.wave_spawn_timer = self.rng.integers(60, 120) # 2-4 seconds

        # Update entities
        self._update_cannons()
        self._move_projectiles()
        self._move_enemies()
        
        # Check for wave completion
        if self.wave_enemies_to_spawn == 0 and not self.enemies:
            self.game_phase = 'build'
            self.build_phase_timer = 450
            self.wave_number += 1
            self.reward_this_step += 1.0 # Wave survival reward
            self.score += 100 * self.wave_number
            self.resources += 50 + 10 * self.wave_number
            # sfx: WAVE_COMPLETE_SOUND

    def _spawn_enemy(self):
        health = 20 * (1.1 ** (self.wave_number - 1))
        speed = 1.0 * (1.05 ** (self.wave_number - 1))
        enemy = {
            'pos': list(self.spawn_point),
            'health': health,
            'max_health': health,
            'speed': speed,
            'target_block': None,
        }
        self.enemies.append(enemy)

    def _update_cannons(self):
        for block in self.blocks.values():
            if block['type'] == 'cannon':
                block['fire_cooldown'] = max(0, block['fire_cooldown'] - 1)
                
                # Find new target if needed
                if block.get('target') is None or block['target'] not in self.enemies:
                    block['target'] = None
                    closest_enemy = None
                    min_dist = block['range'] ** 2
                    cannon_pos = self._grid_to_pixel(block['grid_pos'])
                    for enemy in self.enemies:
                        dist_sq = (enemy['pos'][0] - cannon_pos[0])**2 + (enemy['pos'][1] - cannon_pos[1])**2
                        if dist_sq < min_dist:
                            min_dist = dist_sq
                            closest_enemy = enemy
                    block['target'] = closest_enemy

                # Fire if ready and has target
                if block['target'] and block['fire_cooldown'] == 0:
                    block['fire_cooldown'] = self.block_config['cannon']['fire_rate']
                    cannon_pos = self._grid_to_pixel(block['grid_pos'])
                    
                    # sfx: CANNON_FIRE_SOUND
                    self._create_particles(cannon_pos, self.COLOR_PROJECTILE, 5, speed_mult=2)

                    projectile = {
                        'pos': list(cannon_pos),
                        'target': block['target'],
                        'speed': 5,
                        'damage': self.block_config['cannon']['damage']
                    }
                    self.projectiles.append(projectile)

    def _move_projectiles(self):
        projectiles_to_remove = []
        for proj in self.projectiles:
            if proj['target'] not in self.enemies:
                projectiles_to_remove.append(proj)
                continue

            target_pos = proj['target']['pos']
            proj_pos = proj['pos']
            
            direction = [target_pos[0] - proj_pos[0], target_pos[1] - proj_pos[1]]
            dist = math.hypot(*direction)

            if dist < self.grid_size / 2: # Hit
                proj['target']['health'] -= proj['damage']
                self.reward_this_step += 0.1
                self.score += 10
                self._create_particles(proj_pos, self.COLOR_PROJECTILE, 15)
                projectiles_to_remove.append(proj)
                # sfx: ENEMY_HIT_SOUND
            else:
                direction[0] /= dist
                direction[1] /= dist
                proj['pos'][0] += direction[0] * proj['speed']
                proj['pos'][1] += direction[1] * proj['speed']

        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]

    def _move_enemies(self):
        enemies_to_remove = []
        blocks_to_remove = []
        
        for enemy in self.enemies:
            if enemy['health'] <= 0:
                enemies_to_remove.append(enemy)
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 30, 0.95)
                # sfx: ENEMY_DEATH_SOUND
                continue

            # Find target block if none or destroyed
            if enemy['target_block'] is None or enemy['target_block']['grid_pos'] not in self.blocks:
                enemy['target_block'] = self._find_closest_block(enemy['pos'])

            if enemy['target_block']:
                target_pos = self._grid_to_pixel(enemy['target_block']['grid_pos'])
                direction = [target_pos[0] - enemy['pos'][0], target_pos[1] - enemy['pos'][1]]
                dist = math.hypot(*direction)

                if dist < self.grid_size: # Close enough to attack
                    enemy['target_block']['health'] -= 1 # Attack damage
                    if enemy['target_block']['type'] == 'core':
                        self.fortress_health = enemy['target_block']['health']
                    
                    if enemy['target_block']['health'] <= 0:
                        if enemy['target_block']['type'] != 'core':
                             blocks_to_remove.append(enemy['target_block']['grid_pos'])
                        enemy['target_block'] = None # Find new target next frame
                else: # Move towards target
                    direction[0] /= dist
                    direction[1] /= dist
                    enemy['pos'][0] += direction[0] * enemy['speed']
                    enemy['pos'][1] += direction[1] * enemy['speed']
            else: # No blocks left, move to core
                target_pos = self._grid_to_pixel(self.fortress_core_grid_pos)
                direction = [target_pos[0] - enemy['pos'][0], target_pos[1] - enemy['pos'][1]]
                dist = math.hypot(*direction)
                if dist > 1:
                    direction[0] /= dist
                    direction[1] /= dist
                    enemy['pos'][0] += direction[0] * enemy['speed']
                    enemy['pos'][1] += direction[1] * enemy['speed']

        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        for pos in set(blocks_to_remove):
            if pos in self.blocks:
                self._create_particles(self._grid_to_pixel(pos), (128,128,128), 20)
                del self.blocks[pos]

    def _find_closest_block(self, pos):
        closest_block = None
        min_dist_sq = float('inf')
        for block in self.blocks.values():
            block_pos = self._grid_to_pixel(block['grid_pos'])
            dist_sq = (pos[0] - block_pos[0])**2 + (pos[1] - block_pos[1])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_block = block
        return closest_block

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['vel'][0] *= p.get('drag', 0.98)
            p['vel'][1] *= p.get('drag', 0.98)

    def _check_termination(self):
        return self.fortress_health <= 0 or self.wave_number > 20 or self.steps >= 10000

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
            "wave": self.wave_number,
            "resources": self.resources,
            "fortress_health": self.fortress_health,
            "phase": self.game_phase
        }
        
    def _grid_to_pixel(self, grid_pos):
        return (
            grid_pos[0] * self.grid_size + self.grid_size // 2,
            grid_pos[1] * self.grid_size + self.grid_size // 2
        )

    def _render_game(self):
        # Draw build area and grid
        pygame.draw.rect(self.screen, self.COLOR_BUILD_AREA, self.build_area_rect)
        for x in range(self.build_area_rect.left, self.build_area_rect.right, self.grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.build_area_rect.top), (x, self.build_area_rect.bottom))
        for y in range(self.build_area_rect.top, self.build_area_rect.bottom, self.grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.build_area_rect.left, y), (self.build_area_rect.right, y))
        pygame.draw.rect(self.screen, self.COLOR_GRID, self.build_area_rect, 1)

        # Draw blocks
        for pos, block in self.blocks.items():
            px, py = self._grid_to_pixel(pos)
            if block['type'] == 'core':
                pygame.gfxdraw.filled_circle(self.screen, px, py, self.grid_size // 2, self.COLOR_CORE)
                pygame.gfxdraw.aacircle(self.screen, px, py, self.grid_size // 2, self.COLOR_CORE)
            else:
                config = self.block_config[block['type']]
                rect = pygame.Rect(px - self.grid_size//2, py - self.grid_size//2, self.grid_size, self.grid_size)
                pygame.draw.rect(self.screen, config['color'], rect, border_radius=2)
                if block['type'] == 'cannon':
                    pygame.gfxdraw.filled_circle(self.screen, px, py, 4, (50,50,50))
                    if block.get('target'):
                        target_pos = block['target']['pos']
                        angle = math.atan2(target_pos[1] - py, target_pos[0] - px)
                        end_x = px + (self.grid_size//2) * math.cos(angle)
                        end_y = py + (self.grid_size//2) * math.sin(angle)
                        pygame.draw.line(self.screen, (50,50,50), (px,py), (int(end_x), int(end_y)), 3)

            # Health bar for blocks
            if block['health'] < block['max_health']:
                health_pct = block['health'] / block['max_health']
                bar_w = self.grid_size * health_pct
                bar_rect = pygame.Rect(px - self.grid_size//2, py + self.grid_size//2 + 2, bar_w, 3)
                pygame.draw.rect(self.screen, (0,255,0) if block['type'] != 'core' else self.COLOR_CORE, bar_rect)

        # Draw enemies
        for enemy in self.enemies:
            px, py = int(enemy['pos'][0]), int(enemy['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, px, py, self.grid_size//3, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, px, py, self.grid_size//3, tuple(c*0.8 for c in self.COLOR_ENEMY))
            # Health bar for enemies
            health_pct = enemy['health'] / enemy['max_health']
            bar_w = self.grid_size * 0.8 * health_pct
            bar_rect = pygame.Rect(px - self.grid_size * 0.4, py - self.grid_size * 0.6, bar_w, 3)
            pygame.draw.rect(self.screen, (255,0,0), bar_rect)
            
        # Draw projectiles
        for proj in self.projectiles:
            px, py = int(proj['pos'][0]), int(proj['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, px, py, 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, px, py, 3, self.COLOR_PROJECTILE)
            
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                # Basic particle rendering, no need for surface alpha
                 pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

        # Draw cursor
        if self.game_phase == 'build':
            cursor_px, cursor_py = self._grid_to_pixel(self.cursor_grid_pos)
            rect = pygame.Rect(cursor_px - self.grid_size//2, cursor_py - self.grid_size//2, self.grid_size, self.grid_size)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=3)

    def _render_ui(self):
        # Top Bar
        top_bar_rect = pygame.Rect(0, 0, self.screen_width, 40)
        pygame.draw.rect(self.screen, (20,22,25), top_bar_rect)
        
        # Info text
        info_texts = [
            f"WAVE: {self.wave_number}/20",
            f"HEALTH: {max(0, int(self.fortress_health))}",
            f"RESOURCES: {self.resources}",
            f"SCORE: {self.score}"
        ]
        for i, text in enumerate(info_texts):
            self._draw_text(text, (10 + i * 150, 20), self.font_large, self.COLOR_TEXT)

        # Phase indicator
        if self.game_phase == 'build':
            time_left = self.build_phase_timer / 30
            phase_text = f"BUILD PHASE: {time_left:.1f}s"
            color = (100, 200, 255)
        else: # wave
            phase_text = "WAVE IN PROGRESS"
            color = self.COLOR_ENEMY
        self._draw_text(phase_text, (self.screen_width // 2, self.screen_height - 20), self.font_large, color)
        
        # Selected block display
        if self.game_phase == 'build':
            block_name = self.block_types[self.selected_block_idx].upper()
            cost = self.block_config[self.block_types[self.selected_block_idx]]['cost']
            select_text = f"SELECTED: {block_name} (Cost: {cost})"
            self._draw_text(select_text, (self.screen_width - 10, 20), self.font_large, self.COLOR_TEXT, align="right")
        
        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.wave_number > 20 else "GAME OVER"
            color = (0, 255, 0) if self.wave_number > 20 else (255, 0, 0)
            self._draw_text(msg, (self.screen_width//2, self.screen_height//2 - 20), self.font_huge, color)
            self._draw_text(f"Final Score: {self.score}", (self.screen_width//2, self.screen_height//2 + 30), self.font_large, self.COLOR_TEXT)

    def _draw_text(self, text, pos, font, color, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "left":
            text_rect.midleft = pos
        elif align == "right":
            text_rect.midright = pos
        self.screen.blit(text_surface, text_rect)
        
    def _create_particles(self, pos, color, count, speed_mult=1.0, drag=0.95):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = (self.rng.random() + 0.1) * 3 * speed_mult
            life = self.rng.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.rng.integers(2, 5),
                'drag': drag,
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    terminated = False
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Tower Defense Gym Environment")
    
    # Action state
    movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space_held = 0
    shift_held = 0

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
            # Key presses
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
                elif event.key == pygame.K_r: # Reset key
                    obs, info = env.reset()

            # Key releases
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    movement = 0
                elif event.key == pygame.K_SPACE: space_held = 0
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 0

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Run at 30 FPS

    env.close()