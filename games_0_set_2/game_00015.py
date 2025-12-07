
# Generated: 2025-08-27T16:22:12.951615
# Source Brief: brief_00015.md
# Brief Index: 15

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑/↓ to aim your swing. Press space to attack. "
        "Defeat all the monsters before they overwhelm you."
    )

    game_description = (
        "Swing your sword to defeat waves of procedurally generated monsters in this side-view arcade game. "
        "Time your attacks to cleave through enemies, but be careful—they will damage you on contact!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and rendering constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.FONT_UI = pygame.font.Font(None, 24)
        self.FONT_TITLE = pygame.font.Font(None, 48)
        self.FONT_SUBTITLE = pygame.font.Font(None, 32)
        
        # Colors
        self.COLOR_BG_TOP = (30, 30, 40)
        self.COLOR_BG_BOTTOM = (50, 50, 60)
        self.COLOR_PLAYER = (100, 220, 100)
        self.COLOR_PLAYER_DMG = (255, 100, 100)
        self.COLOR_ENEMY = (220, 80, 80)
        self.COLOR_ENEMY_DARK = (100, 40, 40)
        self.COLOR_SWORD = (255, 255, 100)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_UI_BG = (60, 60, 70)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_HP_HIGH = (100, 200, 100)
        self.COLOR_HP_MID = (200, 200, 100)
        self.COLOR_HP_LOW = (200, 100, 100)
        
        # Game constants
        self.MAX_STEPS = 5000
        self.WIN_CONDITION_KILLS = 20
        self.PLAYER_MAX_HEALTH = 100.0
        self.PLAYER_POS = (100, self.HEIGHT // 2)
        self.PLAYER_SIZE = (20, 40)
        
        # State variables must be defined in __init__ before reset() is called
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_health = 0
        self.player_hit_timer = 0
        self.monsters = []
        self.particles = []
        self.monsters_defeated = 0
        self.monster_spawn_timer = 0
        self.monster_spawn_interval = 0
        self.base_monster_speed = 0
        self.swing_aim_angle = 0
        self.swing_state = {}
        self.prev_space_held = False
        self.np_random = None

        # Initialize state
        self.reset()
        
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        # Player state
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_hit_timer = 0
        
        # Sword state
        self.swing_aim_angle = 0  # 0 is horizontal right, positive is up
        self.swing_state = {
            'active': False, 'timer': 0, 'duration': 12,  # in frames
            'start_angle': 0, 'end_angle': 0, 'radius': 70, 'width': 6,
            'damage': 0, 'hit_monsters': set()
        }
        self.prev_space_held = False

        # Monster state
        self.monsters = []
        self.monsters_defeated = 0
        self.monster_spawn_interval = 50  # Initial spawn rate
        self.monster_spawn_timer = self.monster_spawn_interval
        self.base_monster_speed = 1.0
        
        # VFX
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # 1. Handle Input
        self._handle_input(action)
        
        # 2. Update Game State
        self._update_swing()
        self._update_monsters()
        reward += self._check_collisions()
        self._update_particles()
        self._spawn_monsters()
        
        self.player_hit_timer = max(0, self.player_hit_timer - 1)
        
        # 3. Calculate rewards from swing
        if self.swing_state['active']:
            swing_reward = self._process_swing_hits()
            reward += swing_reward

        # 4. Check Termination
        self.steps += 1
        terminated = False
        if self.player_health <= 0:
            terminated = True
            self.game_over = True
            reward -= 100
        elif self.monsters_defeated >= self.WIN_CONDITION_KILLS:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1:  # Up
            self.swing_aim_angle = min(45, self.swing_aim_angle + 5)
        elif movement == 2:  # Down
            self.swing_aim_angle = max(-45, self.swing_aim_angle - 5)
            
        space_pressed = space_held and not self.prev_space_held
        if space_pressed and not self.swing_state['active']:
            # SFX: Sword woosh
            self.swing_state.update({
                'active': True,
                'timer': 0,
                'start_angle': self.swing_aim_angle - 60,
                'end_angle': self.swing_aim_angle + 60,
                'damage': self.np_random.integers(5, 16),
                'hit_monsters': set()
            })
        self.prev_space_held = space_held

    def _update_swing(self):
        if not self.swing_state['active']:
            return
        
        self.swing_state['timer'] += 1
        if self.swing_state['timer'] >= self.swing_state['duration']:
            self.swing_state['active'] = False

    def _process_swing_hits(self):
        reward = 0
        swing_rad = self.swing_state['radius']
        start_angle_rad = math.radians(self.swing_state['start_angle'])
        end_angle_rad = math.radians(self.swing_state['end_angle'])

        for i, monster in enumerate(self.monsters):
            if i in self.swing_state['hit_monsters']:
                continue

            dist_to_player = math.hypot(monster['pos'][0] - self.PLAYER_POS[0], monster['pos'][1] - self.PLAYER_POS[1])
            
            if dist_to_player < swing_rad + monster['size']:
                angle_to_monster = math.atan2(self.PLAYER_POS[1] - monster['pos'][1], monster['pos'][0] - self.PLAYER_POS[0])
                
                # Normalize angle to be within the same range as swing angles
                angle_to_monster = math.degrees(angle_to_monster)

                if self.swing_state['start_angle'] <= angle_to_monster <= self.swing_state['end_angle']:
                    damage = self.swing_state['damage']
                    monster['health'] -= damage
                    monster['hit_timer'] = 5
                    self.swing_state['hit_monsters'].add(i)
                    reward += damage * 0.1
                    # SFX: Monster hit
                    for _ in range(int(damage / 2)):
                        self._create_particle(monster['pos'], self.COLOR_SWORD, 3, 10, 0.95)
        return reward

    def _update_monsters(self):
        for monster in self.monsters:
            dx = self.PLAYER_POS[0] - monster['pos'][0]
            dy = self.PLAYER_POS[1] - monster['pos'][1]
            dist = math.hypot(dx, dy)
            if dist > 1:
                monster['pos'] = (
                    monster['pos'][0] + (dx / dist) * monster['speed'],
                    monster['pos'][1] + (dy / dist) * monster['speed']
                )
            monster['hit_timer'] = max(0, monster['hit_timer'] - 1)

    def _check_collisions(self):
        reward = 0
        surviving_monsters = []
        player_rect = pygame.Rect(self.PLAYER_POS[0] - self.PLAYER_SIZE[0] / 2,
                                  self.PLAYER_POS[1] - self.PLAYER_SIZE[1] / 2,
                                  *self.PLAYER_SIZE)

        for monster in self.monsters:
            if monster['health'] <= 0:
                self.monsters_defeated += 1
                self.score += 10
                reward += 10
                # SFX: Monster death
                for _ in range(20):
                    self._create_particle(monster['pos'], monster['color'], 5, 20, 0.92)
                
                if self.monsters_defeated > 0 and self.monsters_defeated % 5 == 0:
                    self.monster_spawn_interval = max(20, self.monster_spawn_interval - 1)
                    self.base_monster_speed += 0.05
                continue

            monster_rect = pygame.Rect(monster['pos'][0] - monster['size'], monster['pos'][1] - monster['size'],
                                       monster['size'] * 2, monster['size'] * 2)
            if player_rect.colliderect(monster_rect) and self.player_hit_timer == 0:
                damage = 5
                self.player_health -= damage
                self.player_hit_timer = 30  # 1s invincibility
                # SFX: Player hit
                for _ in range(15):
                    self._create_particle(self.PLAYER_POS, self.COLOR_PLAYER_DMG, 4, 15, 0.9)
            
            surviving_monsters.append(monster)
        
        self.monsters = surviving_monsters
        return reward

    def _spawn_monsters(self):
        self.monster_spawn_timer -= 1
        if self.monster_spawn_timer <= 0:
            self.monster_spawn_timer = self.monster_spawn_interval
            
            spawn_x = self.WIDTH + 30
            spawn_y = self.np_random.integers(50, self.HEIGHT - 50)
            
            health = self.np_random.integers(10, 31)
            new_monster = {
                'pos': (float(spawn_x), float(spawn_y)),
                'health': float(health),
                'max_health': float(health),
                'speed': self.base_monster_speed + self.np_random.uniform(-0.2, 0.2),
                'size': self.np_random.integers(8, 15),
                'color': self.COLOR_ENEMY,
                'hit_timer': 0
            }
            self.monsters.append(new_monster)
    
    def _create_particle(self, pos, color, max_speed, max_life, drag):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, max_speed)
        particle = {
            'pos': list(pos),
            'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
            'lifespan': self.np_random.integers(max_life // 2, max_life),
            'max_lifespan': max_life,
            'color': color,
            'size': self.np_random.integers(2, 5),
            'drag': drag
        }
        self.particles.append(particle)

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= p['drag']
            p['vel'][1] *= p['drag']
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        # --- RENDER ---
        # Background
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = [int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio) for i in range(3)]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        # Particles
        for p in self.particles:
            size = int(p['size'] * (p['lifespan'] / p['max_lifespan']))
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

        # Monsters
        for m in self.monsters:
            pos = (int(m['pos'][0]), int(m['pos'][1]))
            color = self.COLOR_WHITE if m['hit_timer'] > 0 else m['color']
            pygame.draw.rect(self.screen, color, (pos[0] - m['size'], pos[1] - m['size'], m['size'] * 2, m['size'] * 2))
            if m['health'] < m['max_health']:
                hp_ratio = m['health'] / m['max_health']
                bar_width = m['size'] * 2
                pygame.draw.rect(self.screen, self.COLOR_ENEMY_DARK, (pos[0] - m['size'], pos[1] - m['size'] - 8, bar_width, 4))
                pygame.draw.rect(self.screen, self.COLOR_HP_HIGH, (pos[0] - m['size'], pos[1] - m['size'] - 8, bar_width * hp_ratio, 4))
        
        # Player
        player_color = self.COLOR_PLAYER if self.player_hit_timer % 6 < 3 else self.COLOR_PLAYER_DMG
        player_rect = pygame.Rect(self.PLAYER_POS[0] - self.PLAYER_SIZE[0]/2, self.PLAYER_POS[1] - self.PLAYER_SIZE[1]/2, *self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, player_color, player_rect, border_radius=4)
        
        # Aim indicator
        aim_end_x = self.PLAYER_POS[0] + 30 * math.cos(math.radians(self.swing_aim_angle))
        aim_end_y = self.PLAYER_POS[1] - 30 * math.sin(math.radians(self.swing_aim_angle))
        pygame.draw.line(self.screen, self.COLOR_UI, self.PLAYER_POS, (int(aim_end_x), int(aim_end_y)), 1)
        
        # Sword swing
        if self.swing_state['active']:
            rect = pygame.Rect(self.PLAYER_POS[0] - self.swing_state['radius'], self.PLAYER_POS[1] - self.swing_state['radius'],
                               self.swing_state['radius'] * 2, self.swing_state['radius'] * 2)
            start_rad = math.radians(self.swing_state['start_angle'])
            end_rad = math.radians(self.swing_state['end_angle'])
            pygame.draw.arc(self.screen, self.COLOR_SWORD, rect, start_rad, end_rad, self.swing_state['width'])

        # UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Health bar
        hp_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        hp_color = self.COLOR_HP_HIGH if hp_ratio > 0.5 else (self.COLOR_HP_MID if hp_ratio > 0.2 else self.COLOR_HP_LOW)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 10, 204, 24))
        pygame.draw.rect(self.screen, hp_color, (12, 12, 200 * hp_ratio, 20))
        
        # Score
        score_text = self.FONT_UI.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 15))
        
        # Monster count
        count_text = self.FONT_SUBTITLE.render(f"{self.monsters_defeated} / {self.WIN_CONDITION_KILLS}", True, self.COLOR_UI)
        text_rect = count_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 30))
        self.screen.blit(count_text, text_rect)
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY" if self.win else "GAME OVER"
            end_text = self.FONT_TITLE.render(msg, True, self.COLOR_WHITE)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def validate_implementation(self):
        print("Running implementation validation...")
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()