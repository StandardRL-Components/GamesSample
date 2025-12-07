
# Generated: 2025-08-28T02:15:50.335547
# Source Brief: brief_01651.md
# Brief Index: 1651

        
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
        "Controls: Arrow keys to move. Hold Space to fire. Shift does nothing (yet!)."
    )

    game_description = (
        "A fast-paced, top-down arcade space shooter. Survive five waves of increasingly difficult alien attackers."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 72)
        self.font_subtitle = pygame.font.Font(None, 36)

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_PLAYER_PROJECTILE = (128, 255, 255)
        self.COLOR_EXPLOSION = [(255, 255, 100), (255, 150, 50), (200, 50, 50)]
        self.COLOR_WAVE_TEXT = (255, 255, 255)
        self.ENEMY_COLORS = [
            (255, 80, 80), (100, 150, 255), (200, 100, 255),
            (255, 180, 80), (255, 255, 255)
        ]

        # Game constants
        self.PLAYER_SPEED = 5
        self.PLAYER_FIRE_COOLDOWN = 6  # frames
        self.PROJECTILE_SPEED = 10
        self.MAX_WAVES = 5
        self.WAVE_TRANSITION_TIME = 90  # 3 seconds at 30fps
        self.MAX_EPISODE_STEPS = 30 * 120 # 2 minutes

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_projectiles = None
        self.last_player_fire_step = None
        self.enemies = None
        self.enemy_projectiles = None
        self.explosions = None
        self.stars = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.current_wave = None
        self.wave_transition_timer = None
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.player_pos = pygame.Vector2(self.width / 2, self.height - 50)
        self.player_health = 100
        self.player_projectiles = []
        self.last_player_fire_step = 0
        self.enemies = []
        self.enemy_projectiles = []
        self.explosions = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.current_wave = 0
        self.wave_transition_timer = self.WAVE_TRANSITION_TIME

        self._generate_stars()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Store pre-move distances for reward calculation
        prev_min_enemy_dist = self._get_min_dist_to(self.enemies)
        prev_min_proj_dist = self._get_min_dist_to(self.enemy_projectiles)

        # Handle player actions
        self._handle_input(movement, space_held)

        # Update game state
        reward_events = self._update_game_state()
        
        # Calculate rewards from events
        reward += reward_events.get("enemy_destroyed", 0) * 10
        if reward_events.get("player_hit"):
            reward -= 5
        if reward_events.get("wave_cleared"):
            reward += 100
            
        # Calculate continuous rewards
        new_min_enemy_dist = self._get_min_dist_to(self.enemies)
        new_min_proj_dist = self._get_min_dist_to(self.enemy_projectiles)

        if new_min_enemy_dist < prev_min_enemy_dist:
            reward += 0.1
        if new_min_proj_dist > prev_min_proj_dist:
            reward += 0.2 # Rewarded for moving away from projectiles

        self.score += reward_events.get("enemy_destroyed", 0) * 10 # Update score based on kills

        # Check termination conditions
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 500
            else: # Loss
                reward -= 100

        self.steps += 1
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_projectiles(self.enemy_projectiles, self.ENEMY_COLORS[self.current_wave-1 if self.current_wave > 0 else 0])
        self._render_projectiles(self.player_projectiles, self.COLOR_PLAYER_PROJECTILE)
        self._render_enemies()
        self._render_player()
        self._render_explosions()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave}

    def _handle_input(self, movement, space_held):
        # Movement
        if movement == 1: self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3: self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos.x += self.PLAYER_SPEED
        
        # Clamp player position
        self.player_pos.x = np.clip(self.player_pos.x, 20, self.width - 20)
        self.player_pos.y = np.clip(self.player_pos.y, 20, self.height - 20)

        # Firing
        if space_held and (self.steps - self.last_player_fire_step) > self.PLAYER_FIRE_COOLDOWN:
            # Sfx: Player shoot
            self.player_projectiles.append(pygame.Vector2(self.player_pos))
            self.last_player_fire_step = self.steps

    def _update_game_state(self):
        reward_events = {}
        
        # Wave management
        if not self.enemies and not self.game_over:
            if self.wave_transition_timer > 0:
                self.wave_transition_timer -= 1
            else:
                if self.current_wave >= self.MAX_WAVES:
                    self.win = True
                else:
                    self.current_wave += 1
                    self._spawn_wave()
                    reward_events["wave_cleared"] = 1
                    self.wave_transition_timer = self.WAVE_TRANSITION_TIME

        # Update player projectiles
        for proj in self.player_projectiles[:]:
            proj.y -= self.PROJECTILE_SPEED
            if proj.y < 0:
                self.player_projectiles.remove(proj)

        # Update enemy projectiles
        for proj in self.enemy_projectiles[:]:
            proj.y += self.PROJECTILE_SPEED
            if proj.y > self.height:
                self.enemy_projectiles.remove(proj)

        # Update enemies
        for enemy in self.enemies[:]:
            self._move_enemy(enemy)
            if self.np_random.random() < enemy['fire_rate']:
                # Sfx: Enemy shoot
                self.enemy_projectiles.append(pygame.Vector2(enemy['pos']))

        # Update explosions
        for explosion in self.explosions[:]:
            explosion['radius'] += explosion['speed']
            explosion['alpha'] -= 15
            if explosion['alpha'] <= 0:
                self.explosions.remove(explosion)

        # Check collisions
        reward_events.update(self._check_collisions())
        
        return reward_events

    def _check_collisions(self):
        events = {}
        # Player projectiles vs enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if proj.distance_to(enemy['pos']) < enemy['size']:
                    # Sfx: Enemy hit
                    enemy['health'] -= 10
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    if enemy['health'] <= 0:
                        # Sfx: Explosion
                        self._create_explosion(enemy['pos'], enemy['size'] * 2)
                        self.enemies.remove(enemy)
                        events["enemy_destroyed"] = events.get("enemy_destroyed", 0) + 1
                    break

        # Enemy projectiles vs player
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 10, 20, 20)
        for proj in self.enemy_projectiles[:]:
            if player_rect.collidepoint(proj):
                # Sfx: Player hit
                self.enemy_projectiles.remove(proj)
                self.player_health -= 10
                self.player_health = max(0, self.player_health)
                events["player_hit"] = True
                self._create_explosion(self.player_pos, 20, speed=1)
                break
        
        return events

    def _check_termination(self):
        if self.player_health <= 0:
            if not self.game_over:
                self._create_explosion(self.player_pos, 40, speed=2)
            self.game_over = True
            return True
        if self.win:
            self.game_over = True
            return True
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            return True
        return False

    def _spawn_wave(self):
        num_enemies = 5 + self.current_wave * 2
        wave_data = {
            1: {'pattern': 'horizontal', 'health': 10, 'size': 12, 'fire_rate': 0.005},
            2: {'pattern': 'diagonal', 'health': 20, 'size': 14, 'fire_rate': 0.01},
            3: {'pattern': 'circular', 'health': 30, 'size': 16, 'fire_rate': 0.015},
            4: {'pattern': 'sine', 'health': 40, 'size': 18, 'fire_rate': 0.02},
            5: {'pattern': 'boss', 'health': 200, 'size': 40, 'fire_rate': 0.03, 'num': 1},
        }
        data = wave_data[self.current_wave]
        num_enemies = data.get('num', num_enemies)

        for i in range(num_enemies):
            if data['pattern'] == 'boss':
                x = self.width / 2
                y = 100
            else:
                x = (self.width / (num_enemies + 1)) * (i + 1)
                y = 80 + self.np_random.integers(-20, 20)
            
            enemy = {
                'pos': pygame.Vector2(x, y),
                'health': data['health'],
                'size': data['size'],
                'fire_rate': data['fire_rate'] + (self.current_wave - 1) * 0.005,
                'pattern': data['pattern'],
                'speed': 1.0 + (self.current_wave - 1) * 0.2,
                'move_counter': self.np_random.random() * 2 * math.pi, # For sine/circle patterns
                'direction': 1 if self.np_random.random() > 0.5 else -1
            }
            self.enemies.append(enemy)

    def _move_enemy(self, enemy):
        pattern = enemy['pattern']
        speed = enemy['speed']
        enemy['move_counter'] += 0.03 * speed
        
        if pattern == 'horizontal':
            enemy['pos'].x += speed * enemy['direction']
            if enemy['pos'].x < enemy['size'] or enemy['pos'].x > self.width - enemy['size']:
                enemy['direction'] *= -1
        elif pattern == 'diagonal':
            enemy['pos'].x += speed * enemy['direction']
            enemy['pos'].y += speed * 0.5
            if enemy['pos'].x < enemy['size'] or enemy['pos'].x > self.width - enemy['size']:
                enemy['direction'] *= -1
            if enemy['pos'].y > self.height + enemy['size']: # Reset if off-screen
                enemy['pos'].y = -enemy['size']
        elif pattern == 'circular':
            center_x = self.width / 2 + enemy['direction'] * 150
            enemy['pos'].x = center_x + math.cos(enemy['move_counter']) * 100
            enemy['pos'].y = 120 + math.sin(enemy['move_counter']) * 50
        elif pattern == 'sine':
            enemy['pos'].x = (self.width/2) + math.sin(enemy['move_counter']) * (self.width/2 - enemy['size'])
            enemy['pos'].y += speed * 0.3
            if enemy['pos'].y > self.height + enemy['size']:
                enemy['pos'].y = -enemy['size']
        elif pattern == 'boss':
            enemy['pos'].x = self.width / 2 + math.sin(enemy['move_counter'] * 0.5) * (self.width / 3)
            enemy['pos'].y = 100 + math.cos(enemy['move_counter']) * 40
            
    def _generate_stars(self):
        self.stars = []
        for i in range(200):
            x = self.np_random.random() * self.width
            y = self.np_random.random() * self.height
            size = self.np_random.random() * 1.5
            layer = self.np_random.integers(1, 4)
            self.stars.append({'pos': pygame.Vector2(x, y), 'size': size, 'layer': layer})
            
    def _create_explosion(self, pos, max_radius, speed=3):
        self.explosions.append({'pos': pygame.Vector2(pos), 'radius': 0, 'max_radius': max_radius, 'alpha': 255, 'speed': speed})
        
    def _get_min_dist_to(self, entities):
        if not entities:
            return float('inf')
        return min(self.player_pos.distance_to(e['pos']) for e in entities)

    # --- RENDER METHODS ---

    def _render_background(self):
        for star in self.stars:
            star['pos'].y += 0.1 * star['layer']
            if star['pos'].y > self.height:
                star['pos'].y = 0
                star['pos'].x = self.np_random.random() * self.width
            
            brightness = 50 + (star['layer'] - 1) * 50
            color = (brightness, brightness, brightness)
            pygame.draw.circle(self.screen, color, (int(star['pos'].x), int(star['pos'].y)), int(star['size']))

    def _render_player(self):
        if self.player_health <= 0: return

        # Ship body
        points = [
            (self.player_pos.x, self.player_pos.y - 15),
            (self.player_pos.x - 10, self.player_pos.y + 10),
            (self.player_pos.x + 10, self.player_pos.y + 10)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), 20, self.COLOR_PLAYER_GLOW)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            size = int(enemy['size'])
            color = self.ENEMY_COLORS[self.current_wave - 1]
            pattern = enemy['pattern']

            if pattern in ['horizontal', 'boss']: # Circle
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
            elif pattern == 'diagonal': # Square
                rect = pygame.Rect(pos[0] - size, pos[1] - size, size*2, size*2)
                pygame.draw.rect(self.screen, color, rect)
            elif pattern in ['circular', 'sine']: # Diamond
                points = [
                    (pos[0], pos[1] - size), (pos[0] + size, pos[1]),
                    (pos[0], pos[1] + size), (pos[0] - size, pos[1])
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)


    def _render_projectiles(self, projectiles, color):
        for proj in projectiles:
            start = (int(proj.x), int(proj.y))
            end = (int(proj.x), int(proj.y + 5 if color != self.COLOR_PLAYER_PROJECTILE else proj.y - 5))
            pygame.draw.line(self.screen, color, start, end, 3)

    def _render_explosions(self):
        for exp in self.explosions:
            for i, c in enumerate(self.COLOR_EXPLOSION):
                radius = int(exp['radius'] * (1 - i * 0.2))
                if radius > 0:
                    alpha = int(exp['alpha'] * (1 - (exp['radius'] / exp['max_radius'])))
                    color = (*c, max(0, min(255, alpha)))
                    surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(surf, color, (radius, radius), radius)
                    self.screen.blit(surf, (int(exp['pos'].x - radius), int(exp['pos'].y - radius)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Health
        health_text = self.font_ui.render(f"HEALTH: {self.player_health}", True, self.COLOR_PLAYER)
        self.screen.blit(health_text, (10, 10))
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, (255, 255, 0))
        self.screen.blit(score_text, (self.width - score_text.get_width() - 10, 10))
        
        # Wave
        wave_text = self.font_ui.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, (200, 200, 255))
        self.screen.blit(wave_text, (self.width/2 - wave_text.get_width()/2, 10))

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            title = self.font_title.render(message, True, color)
            title_rect = title.get_rect(center=(self.width/2, self.height/2 - 20))
            self.screen.blit(title, title_rect)
            
            final_score_text = self.font_subtitle.render(f"Final Score: {int(self.score)}", True, (255, 255, 255))
            score_rect = final_score_text.get_rect(center=(self.width/2, self.height/2 + 30))
            self.screen.blit(final_score_text, score_rect)
        elif self.wave_transition_timer > 0 and self.current_wave > 0:
            alpha = min(255, int(255 * (self.wave_transition_timer / (self.WAVE_TRANSITION_TIME/2))))
            wave_msg = self.font_title.render(f"WAVE {self.current_wave}", True, (*self.COLOR_WAVE_TEXT, alpha))
            wave_rect = wave_msg.get_rect(center=(self.width/2, self.height/2))
            wave_msg.set_alpha(alpha)
            self.screen.blit(wave_msg, wave_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")