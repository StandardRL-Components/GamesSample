
# Generated: 2025-08-27T16:30:50.938592
# Source Brief: brief_01248.md
# Brief Index: 1248

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to select a build site. Space to build a basic tower, Shift for an advanced tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in this minimalist TD."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_PATH = (45, 45, 55)
    COLOR_BASE = (60, 180, 75)
    COLOR_BUILD_SITE = (70, 70, 80)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_ENEMY = (230, 25, 75)
    COLOR_TOWER_BASIC = (67, 133, 245)
    COLOR_TOWER_ADVANCED = (255, 191, 0)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_HEALTH_BG = (128, 0, 0)
    COLOR_HEALTH_FG = (0, 255, 0)
    COLOR_TEXT = (240, 240, 240)

    # Game Parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 3000  # Longer to allow for 10 waves
    MAX_WAVES = 10
    INITIAL_RESOURCES = 250
    RESOURCES_PER_KILL = 25

    # Tower Specs
    TOWER_BASIC_COST = 100
    TOWER_BASIC_RANGE = 80
    TOWER_BASIC_DAMAGE = 10
    TOWER_BASIC_COOLDOWN = 20  # frames

    TOWER_ADVANCED_COST = 225
    TOWER_ADVANCED_RANGE = 120
    TOWER_ADVANCED_DAMAGE = 25
    TOWER_ADVANCED_COOLDOWN = 40  # frames
    TOWER_ADVANCED_UNLOCK_WAVE = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        self.font_gameover = pygame.font.Font(None, 64)

        # Game path and build sites (defined once)
        self.path = [
            (-20, 100), (100, 100), (100, 300), (450, 300),
            (450, 150), (self.SCREEN_WIDTH + 20, 150)
        ]
        self.base_pos = (self.SCREEN_WIDTH - 40, 150)
        self.build_sites = [
            {'pos': (220, 180), 'tower': None},
            {'pos': (330, 180), 'tower': None},
            {'pos': (220, 250), 'tower': None},
            {'pos': (330, 250), 'tower': None},
        ]

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize other state variables
        self.reset()
        
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.reward_this_step = 0

        self.resources = self.INITIAL_RESOURCES
        self.enemies_killed = 0
        self.cursor_index = 0
        
        for site in self.build_sites:
            site['tower'] = None

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.wave_number = 0
        self._start_new_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = -0.01  # Time penalty

        if not self.game_over:
            self._handle_input(action)
            self._update_towers()
            self._update_projectiles()
            self._update_enemies()
            self._update_particles()
            self._check_wave_completion()

        self.steps += 1
        self.score += self.reward_this_step
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.game_over:
             self.reward_this_step += 10 if self.win else -10
        
        reward = self.reward_this_step

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Cursor movement
        if movement != 0:
            last_index = self.cursor_index
            if movement == 1: # Up
                self.cursor_index = max(0, self.cursor_index - 2)
            elif movement == 2: # Down
                self.cursor_index = min(3, self.cursor_index + 2)
            elif movement == 3: # Left
                if self.cursor_index in [1, 3]: self.cursor_index -= 1
            elif movement == 4: # Right
                if self.cursor_index in [0, 2]: self.cursor_index += 1
            # Play sound on move
            # if last_index != self.cursor_index: # sfx: cursor_move

        # Place tower
        site = self.build_sites[self.cursor_index]
        if site['tower'] is None:
            if space_held:
                self._create_tower('basic', site)
            elif shift_held and self.wave_number >= self.TOWER_ADVANCED_UNLOCK_WAVE:
                self._create_tower('advanced', site)

    def _create_tower(self, tower_type, site):
        cost = self.TOWER_BASIC_COST if tower_type == 'basic' else self.TOWER_ADVANCED_COST
        if self.resources >= cost:
            self.resources -= cost
            # sfx: build_tower
            if tower_type == 'basic':
                tower = {
                    'pos': site['pos'], 'type': 'basic', 'range': self.TOWER_BASIC_RANGE,
                    'damage': self.TOWER_BASIC_DAMAGE, 'cooldown': 0, 'max_cooldown': self.TOWER_BASIC_COOLDOWN
                }
            else:
                tower = {
                    'pos': site['pos'], 'type': 'advanced', 'range': self.TOWER_ADVANCED_RANGE,
                    'damage': self.TOWER_ADVANCED_DAMAGE, 'cooldown': 0, 'max_cooldown': self.TOWER_ADVANCED_COOLDOWN
                }
            self.towers.append(tower)
            site['tower'] = tower
            self._create_particles(site['pos'], 20, self.COLOR_CURSOR)


    def _start_new_wave(self):
        self.wave_number += 1
        if self.wave_number > 1:
            self.reward_this_step += 5 # Wave completion bonus
            # sfx: wave_complete

        num_enemies = self.wave_number
        health = 50 + self.wave_number * 10
        speed = 1.0 + self.wave_number * 0.1
        
        for i in range(num_enemies):
            self.enemies.append({
                'pos': [self.path[0][0] - i * 30, self.path[0][1]],
                'health': health,
                'max_health': health,
                'speed': speed,
                'waypoint_index': 1,
                'id': self.np_random.integers(1, 1_000_000)
            })
        # sfx: wave_start

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                target = self._find_target(tower)
                if target:
                    # sfx: tower_shoot
                    self.projectiles.append({
                        'start_pos': tower['pos'],
                        'pos': list(tower['pos']),
                        'target': target,
                        'damage': tower['damage'],
                        'speed': 10
                    })
                    tower['cooldown'] = tower['max_cooldown']

    def _find_target(self, tower):
        closest_enemy = None
        min_dist = tower['range']
        for enemy in self.enemies:
            dist = math.hypot(enemy['pos'][0] - tower['pos'][0], enemy['pos'][1] - tower['pos'][1])
            if dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
        return closest_enemy

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_pos = proj['target']['pos']
            dx = target_pos[0] - proj['pos'][0]
            dy = target_pos[1] - proj['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < proj['speed']:
                # Hit
                # sfx: projectile_hit
                self.reward_this_step += 0.1
                proj['target']['health'] -= proj['damage']
                self._create_particles(proj['pos'], 10, self.COLOR_PROJECTILE)
                if proj['target']['health'] <= 0:
                    # sfx: enemy_die
                    self.reward_this_step += 1.0
                    self.resources += self.RESOURCES_PER_KILL
                    self.enemies_killed += 1
                    self._create_particles(proj['target']['pos'], 30, self.COLOR_ENEMY)
                    self.enemies.remove(proj['target'])
                self.projectiles.remove(proj)
            else:
                # Move
                proj['pos'][0] += (dx / dist) * proj['speed']
                proj['pos'][1] += (dy / dist) * proj['speed']

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy['waypoint_index'] >= len(self.path):
                # Reached base
                # sfx: base_hit
                self.game_over = True
                self.win = False
                self.enemies.remove(enemy)
                continue

            target_waypoint = self.path[enemy['waypoint_index']]
            dx = target_waypoint[0] - enemy['pos'][0]
            dy = target_waypoint[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < enemy['speed']:
                enemy['pos'] = list(target_waypoint)
                enemy['waypoint_index'] += 1
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'color': color
            })

    def _check_wave_completion(self):
        if not self.enemies:
            if self.wave_number >= self.MAX_WAVES:
                if not self.game_over:
                    self.game_over = True
                    self.win = True
            else:
                self._start_new_wave()

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "enemies_killed": self.enemies_killed,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 35)
        
        # Base
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.base_pos[0]-10, self.base_pos[1]-10, 20, 20))
        pygame.gfxdraw.rectangle(self.screen, (self.base_pos[0]-10, self.base_pos[1]-10, 20, 20), self.COLOR_BG)

        # Build sites and cursor
        for i, site in enumerate(self.build_sites):
            pygame.gfxdraw.aacircle(self.screen, int(site['pos'][0]), int(site['pos'][1]), 15, self.COLOR_BUILD_SITE)
            if i == self.cursor_index:
                pulse = int(1 + abs(math.sin(self.steps * 0.2)) * 3)
                pygame.gfxdraw.aacircle(self.screen, int(site['pos'][0]), int(site['pos'][1]), 15 + pulse, self.COLOR_CURSOR)

        # Towers
        for tower in self.towers:
            pos = (int(tower['pos'][0]), int(tower['pos'][1]))
            color = self.COLOR_TOWER_BASIC if tower['type'] == 'basic' else self.COLOR_TOWER_ADVANCED
            if tower['type'] == 'basic':
                pygame.draw.rect(self.screen, color, (pos[0]-8, pos[1]-8, 16, 16))
            else: # advanced is a triangle
                points = [(pos[0], pos[1]-9), (pos[0]-9, pos[1]+9), (pos[0]+9, pos[1]+9)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            
            # Cooldown indicator
            cooldown_ratio = tower['cooldown'] / tower['max_cooldown']
            if cooldown_ratio > 0:
                pygame.draw.arc(self.screen, self.COLOR_BG, (pos[0]-10, pos[1]-10, 20, 20), 0, cooldown_ratio * 2 * math.pi, 3)

        # Projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            radius = 8 + int(self.wave_number * 0.2)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            
            # Health bar
            health_ratio = max(0, enemy['health'] / enemy['max_health'])
            bar_width = 20
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (pos[0] - bar_width//2, pos[1] - radius - 8, bar_width, 5))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (pos[0] - bar_width//2, pos[1] - radius - 8, int(bar_width * health_ratio), 5))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*p['color'], alpha)
            size = int(p['life'] * 0.2)
            if size > 0:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

    def _render_ui(self):
        # Wave
        wave_text = self.font_large.render(f"WAVE {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Resources
        res_text = self.font_large.render(f"R: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (self.SCREEN_WIDTH - res_text.get_width() - 10, 10))

        # Kills
        kills_text = self.font_small.render(f"KILLS: {self.enemies_killed}", True, self.COLOR_TEXT)
        self.screen.blit(kills_text, (self.SCREEN_WIDTH - kills_text.get_width() - 10, self.SCREEN_HEIGHT - kills_text.get_height() - 10))

        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, self.SCREEN_HEIGHT - score_text.get_height() - 10))
        
    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        message = "VICTORY" if self.win else "GAME OVER"
        color = self.COLOR_BASE if self.win else self.COLOR_ENEMY
        
        text = self.font_gameover.render(message, True, color)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11", "dummy" or "windows"

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0

    print(f"\n{env.game_description}")
    print(f"\n{env.user_guide}\n")

    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        move_action = 0 # none
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        # --- End Human Controls ---

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {total_reward:.2f}")
    print(f"Info: {info}")
    
    # Keep the final screen visible for a moment
    pygame.time.wait(3000)

    env.close()