
# Generated: 2025-08-28T00:34:25.412519
# Source Brief: brief_03832.md
# Brief Index: 3832

        
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
        "Controls: Arrows to move cursor. Space to place a Standard Block. Shift to place a Reinforced Block. Do nothing to start the wave."
    )

    game_description = (
        "Build a block fortress to withstand increasingly difficult waves of enemy attacks in an isometric 2D strategy game."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF = 20, 10
        self.GRID_ORIGIN_X = self.WIDTH // 2
        self.GRID_ORIGIN_Y = 100
        self.MAX_STEPS = 2000
        self.MAX_WAVES = 20

        # Colors
        self.COLOR_BG = (40, 40, 50)
        self.COLOR_GRID = (60, 60, 70)
        self.COLOR_CURSOR = (0, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_ENEMY = (255, 80, 50)
        self.COLOR_PROJECTILE = (255, 255, 0)
        
        self.BLOCK_COLORS = {
            'standard': ((50, 150, 200), (40, 120, 160), (30, 90, 120)),
            'reinforced': ((150, 50, 200), (120, 40, 160), (90, 30, 120)),
            'core': ((200, 200, 50), (160, 160, 40), (120, 120, 30)),
        }
        
        self.BLOCK_SPECS = {
            'standard': {'cost': 10, 'hp': 50},
            'reinforced': {'cost': 25, 'hp': 150},
        }

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = "BUILD"
        self.cursor_pos = [0, 0]
        self.blocks = {}
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.wave = 1
        self.fortress_health = 100
        self.max_fortress_health = 100
        self.resources = 0
        self.fortress_core_pos = (0,0)
        self.phase_transition_timer = 0
        self.game_result = ""

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = "BUILD"
        self.wave = 1
        self.fortress_health = 100
        self.resources = 30
        self.game_result = ""

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self.blocks = {}
        self.fortress_core_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT - 2)
        self.blocks[self.fortress_core_pos] = {
            'type': 'core', 'hp': 9999, 'max_hp': 9999, 'damage_flash': 0
        }

        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        if self.game_phase == "BUILD":
            reward += self._handle_build_phase(action)
        elif self.game_phase == "COMBAT":
            reward += self._handle_combat_phase()

        self.score += reward
        terminated = self._check_termination()
        
        if terminated and not self.game_result:
            if self.fortress_health <= 0:
                self.game_result = "DEFEAT"
                reward = -100
            elif self.wave > self.MAX_WAVES:
                self.game_result = "VICTORY!"
                reward = 100
            else: # Max steps reached
                self.game_result = "TIME UP"
                
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_build_phase(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # 2. Handle block placement
        pos_tuple = tuple(self.cursor_pos)
        if pos_tuple not in self.blocks:
            block_type_to_place = None
            if shift_held: block_type_to_place = 'reinforced'
            elif space_held: block_type_to_place = 'standard'

            if block_type_to_place:
                spec = self.BLOCK_SPECS[block_type_to_place]
                if self.resources >= spec['cost']:
                    self.resources -= spec['cost']
                    self.blocks[pos_tuple] = {
                        'type': block_type_to_place, 
                        'hp': spec['hp'], 
                        'max_hp': spec['hp'],
                        'damage_flash': 0
                    }
                    # sfx: place_block
                    self._create_particles(self._iso_to_screen(*pos_tuple), 20, self.BLOCK_COLORS[block_type_to_place][0])

        # 3. Handle phase transition
        if movement == 0 and not space_held and not shift_held:
            self.game_phase = "COMBAT"
            self.phase_transition_timer = 60 # 2 seconds at 30fps
            self._spawn_wave()
            # sfx: wave_start
        
        return 0

    def _handle_combat_phase(self):
        reward = 0
        if self.phase_transition_timer > 0:
            self.phase_transition_timer -= 1
            return 0

        self._update_enemies()
        reward += self._update_projectiles()
        self._update_particles()
        
        if not self.enemies and self.phase_transition_timer <= 0:
            # Wave cleared
            reward += 1.0 
            self.wave += 1
            if self.wave > self.MAX_WAVES:
                self.game_over = True
            else:
                self.game_phase = "BUILD"
                self.resources += 15 + self.wave * 2
                # sfx: wave_cleared
        return reward

    def _spawn_wave(self):
        num_enemies = 3 + self.wave
        for _ in range(num_enemies):
            side = self.np_random.choice(['left', 'right', 'top'])
            if side == 'left':
                start_pos = [-1, self.np_random.integers(0, self.GRID_HEIGHT)]
            elif side == 'right':
                start_pos = [self.GRID_WIDTH, self.np_random.integers(0, self.GRID_HEIGHT)]
            else: # top
                start_pos = [self.np_random.integers(0, self.GRID_WIDTH), -1]
            
            base_hp = 20
            base_damage = 5
            hp = int(base_hp * (1 + 0.1 * (self.wave - 1)))
            damage = int(base_damage * (1 + 0.1 * (self.wave - 1)))

            self.enemies.append({
                'grid_pos': start_pos,
                'screen_pos': np.array(self._iso_to_screen(*start_pos), dtype=float),
                'hp': hp,
                'max_hp': hp,
                'damage': damage,
                'attack_cooldown': self.np_random.integers(30, 90),
                'speed': 0.5 + self.np_random.uniform(-0.1, 0.1)
            })

    def _update_enemies(self):
        for enemy in self.enemies:
            # Pathfinding towards the core
            target_pos = self.fortress_core_pos
            current_pos = np.round(enemy['grid_pos']).astype(int)
            
            # Simple greedy movement
            dx = np.sign(target_pos[0] - current_pos[0])
            dy = np.sign(target_pos[1] - current_pos[1])
            
            move_x, move_y = 0, 0
            if dx != 0 and dy != 0: # Move diagonally if possible
                if self.np_random.random() < 0.5: move_x = dx
                else: move_y = dy
            elif dx != 0: move_x = dx
            elif dy != 0: move_y = dy
                
            next_grid_pos = (current_pos[0] + move_x, current_pos[1] + move_y)

            # Attack or move
            enemy['attack_cooldown'] -= 1
            if next_grid_pos in self.blocks and enemy['attack_cooldown'] <= 0:
                # Attack block
                target_screen_pos = self._iso_to_screen(*next_grid_pos)
                self.projectiles.append({
                    'start_pos': enemy['screen_pos'].copy(),
                    'end_pos': np.array(target_screen_pos, dtype=float),
                    'progress': 0.0,
                    'damage': enemy['damage'],
                    'target_block': next_grid_pos
                })
                enemy['attack_cooldown'] = 120 # Reset cooldown
                # sfx: enemy_fire
            else:
                # Move
                direction = np.array([
                    (move_x - move_y) * self.TILE_WIDTH_HALF,
                    (move_x + move_y) * self.TILE_HEIGHT_HALF
                ])
                enemy['screen_pos'] += direction * enemy['speed'] / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else np.array([0,0])
                enemy['grid_pos'] = self._screen_to_iso(*enemy['screen_pos'])


    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        for i, p in enumerate(self.projectiles):
            p['progress'] += 0.1 # Speed of projectile
            if p['progress'] >= 1.0:
                projectiles_to_remove.append(i)
                pos = tuple(np.round(p['target_block']).astype(int))
                
                if pos in self.blocks:
                    block = self.blocks[pos]
                    block['hp'] -= p['damage']
                    block['damage_flash'] = 10
                    # sfx: impact
                    self._create_particles(p['end_pos'], 15, self.COLOR_PROJECTILE)
                    if block['hp'] <= 0 and block['type'] != 'core':
                        del self.blocks[pos]
                        # sfx: block_destroy
                        self._create_particles(p['end_pos'], 40, (100, 100, 100))
                    elif block['hp'] <= 0 and block['type'] == 'core':
                        self.fortress_health = 0 # Game over
                else: # Projectile hit empty space behind fortress
                    if p['target_block'][1] >= self.GRID_HEIGHT -1:
                        self.fortress_health -= p['damage']
                        self.fortress_health = max(0, self.fortress_health)
                        # sfx: fortress_damage
                        # Visual feedback for fortress damage can be a screen shake or flash
        
        # Remove projectiles that have hit their target
        for i in sorted(projectiles_to_remove, reverse=True):
            del self.projectiles[i]
            
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _check_termination(self):
        if self.fortress_health <= 0: self.game_over = True
        if self.wave > self.MAX_WAVES: self.game_over = True
        if self.steps >= self.MAX_STEPS: self.game_over = True
        return self.game_over

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "health": self.fortress_health,
            "resources": self.resources,
        }
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                sx, sy = self._iso_to_screen(x, y)
                points = [
                    (sx, sy),
                    (sx + self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
                    (sx, sy + self.TILE_HEIGHT_HALF * 2),
                    (sx - self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Render entities layer by layer
        entities = []
        for pos, block in self.blocks.items():
            entities.append({'type': 'block', 'pos': pos, 'data': block})
        for enemy in self.enemies:
            entities.append({'type': 'enemy', 'pos': enemy['grid_pos'], 'data': enemy})
        
        entities.sort(key=lambda e: e['pos'][0] + e['pos'][1])

        for entity in entities:
            if entity['type'] == 'block':
                self._render_block(entity['pos'], entity['data'])
            elif entity['type'] == 'enemy':
                self._render_enemy(entity['data'])

        if self.game_phase == "BUILD":
            self._render_cursor()
        
        self._render_projectiles()
        self._render_particles()

    def _iso_to_screen(self, x, y):
        sx = self.GRID_ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        sy = self.GRID_ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(sx), int(sy)

    def _screen_to_iso(self, sx, sy):
        sx_rel = sx - self.GRID_ORIGIN_X
        sy_rel = sy - self.GRID_ORIGIN_Y
        x = (sx_rel / self.TILE_WIDTH_HALF + sy_rel / self.TILE_HEIGHT_HALF) / 2
        y = (sy_rel / self.TILE_HEIGHT_HALF - sx_rel / self.TILE_WIDTH_HALF) / 2
        return x, y

    def _render_block(self, pos, block):
        x, y = pos
        sx, sy = self._iso_to_screen(x, y)
        colors = self.BLOCK_COLORS[block['type']]
        block_height = 20
        
        top_poly = [
            (sx, sy - block_height),
            (sx + self.TILE_WIDTH_HALF, sy - block_height + self.TILE_HEIGHT_HALF),
            (sx, sy - block_height + self.TILE_HEIGHT_HALF * 2),
            (sx - self.TILE_WIDTH_HALF, sy - block_height + self.TILE_HEIGHT_HALF),
        ]
        left_poly = [
            (sx - self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
            (sx, sy + self.TILE_HEIGHT_HALF * 2),
            (sx, sy - block_height + self.TILE_HEIGHT_HALF * 2),
            (sx - self.TILE_WIDTH_HALF, sy - block_height + self.TILE_HEIGHT_HALF),
        ]
        right_poly = [
            (sx + self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
            (sx, sy + self.TILE_HEIGHT_HALF * 2),
            (sx, sy - block_height + self.TILE_HEIGHT_HALF * 2),
            (sx + self.TILE_WIDTH_HALF, sy - block_height + self.TILE_HEIGHT_HALF),
        ]

        # Damage flash
        if block.get('damage_flash', 0) > 0:
            flash_color = (255, 0, 0)
            pygame.gfxdraw.filled_polygon(self.screen, top_poly, flash_color)
            pygame.gfxdraw.filled_polygon(self.screen, left_poly, flash_color)
            pygame.gfxdraw.filled_polygon(self.screen, right_poly, flash_color)
            block['damage_flash'] -= 1
        else:
            pygame.gfxdraw.filled_polygon(self.screen, top_poly, colors[0])
            pygame.gfxdraw.filled_polygon(self.screen, left_poly, colors[1])
            pygame.gfxdraw.filled_polygon(self.screen, right_poly, colors[2])
        
        pygame.gfxdraw.aapolygon(self.screen, top_poly, colors[2])
        pygame.gfxdraw.aapolygon(self.screen, left_poly, colors[2])
        pygame.gfxdraw.aapolygon(self.screen, right_poly, colors[2])
        
        # Health bar for blocks
        if block['hp'] < block['max_hp']:
            health_pct = block['hp'] / block['max_hp']
            bar_width = 30
            bar_height = 4
            bar_x = sx - bar_width // 2
            bar_y = sy - block_height - 10
            pygame.draw.rect(self.screen, (80,0,0), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, (0,200,0), (bar_x, bar_y, bar_width * health_pct, bar_height))

    def _render_cursor(self):
        sx, sy = self._iso_to_screen(*self.cursor_pos)
        points = [
            (sx, sy),
            (sx + self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
            (sx, sy + self.TILE_HEIGHT_HALF * 2),
            (sx - self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
        ]
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, points, 2)

    def _render_enemy(self, enemy):
        sx, sy = enemy['screen_pos']
        # Simple pyramid shape
        size = 12
        color = self.COLOR_ENEMY
        points = [(sx, sy - size), (sx - size, sy + size), (sx + size, sy + size)]
        pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), color)
        pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), color)

        # Health bar
        health_pct = enemy['hp'] / enemy['max_hp']
        bar_width = 25
        bar_height = 4
        bar_x = sx - bar_width / 2
        bar_y = sy - size - 10
        pygame.draw.rect(self.screen, (80,0,0), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, (200,0,0), (bar_x, bar_y, bar_width * health_pct, bar_height))

    def _render_projectiles(self):
        for p in self.projectiles:
            current_pos = p['start_pos'] + (p['end_pos'] - p['start_pos']) * p['progress']
            pygame.gfxdraw.filled_circle(self.screen, int(current_pos[0]), int(current_pos[1]), 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, int(current_pos[0]), int(current_pos[1]), 3, self.COLOR_PROJECTILE)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

    def _render_ui(self):
        # Health Bar
        pygame.draw.rect(self.screen, (80,0,0), (10, 10, 200, 20))
        health_width = max(0, 200 * (self.fortress_health / self.max_fortress_health))
        pygame.draw.rect(self.screen, (0,200,0), (10, 10, health_width, 20))
        health_text = self.font_small.render(f"Core HP: {self.fortress_health}/{self.max_fortress_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Wave and Resources
        wave_text = self.font_small.render(f"Wave: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - 120, 10))
        resources_text = self.font_small.render(f"Resources: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(resources_text, (self.WIDTH - 120, 30))

        # Phase indicator
        if self.game_phase == "COMBAT" and self.phase_transition_timer > 0:
            alpha = min(255, int(255 * (self.phase_transition_timer / 30))) if self.phase_transition_timer < 30 else 255
            text_surf = self.font_large.render(f"WAVE {self.wave}", True, self.COLOR_TEXT)
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (self.WIDTH//2 - text_surf.get_width()//2, self.HEIGHT//2 - text_surf.get_height()//2))
        elif self.game_phase == "BUILD":
            text_surf = self.font_large.render("BUILD PHASE", True, self.COLOR_TEXT)
            text_surf.set_alpha(100)
            self.screen.blit(text_surf, (self.WIDTH//2 - text_surf.get_width()//2, 20))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            result_text = self.font_large.render(self.game_result, True, self.COLOR_TEXT)
            self.screen.blit(result_text, (self.WIDTH//2 - result_text.get_width()//2, self.HEIGHT//2 - 50))
            score_text = self.font_small.render(f"Final Score: {self.score:.1f}", True, self.COLOR_TEXT)
            self.screen.blit(score_text, (self.WIDTH//2 - score_text.get_width()//2, self.HEIGHT//2 + 10))

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a dictionary to track held keys for smoother controls
    keys_held = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False,
        pygame.K_RSHIFT: False,
    }

    # Pygame window for human play
    pygame.display.set_caption("Block Fortress")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    print(env.user_guide)

    while not done:
        action = [0, 0, 0] # Default no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False
        
        # Map held keys to the MultiDiscrete action
        if keys_held[pygame.K_UP]: action[0] = 1
        elif keys_held[pygame.K_DOWN]: action[0] = 2
        elif keys_held[pygame.K_LEFT]: action[0] = 3
        elif keys_held[pygame.K_RIGHT]: action[0] = 4
        
        if keys_held[pygame.K_SPACE]: action[1] = 1
        if keys_held[pygame.K_LSHIFT] or keys_held[pygame.K_RSHIFT]: action[2] = 1
        
        # In manual play, we want to step only on an explicit action
        # The environment logic handles the turn-based nature correctly
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Wave: {info['wave']}, Health: {info['health']}")

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit frame rate

    pygame.quit()
    print("Game Over!")