
# Generated: 2025-08-27T23:22:11.683952
# Source Brief: brief_03436.md
# Brief Index: 3436

        
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
        "Controls: ↑ to jump, ↓ to slide. Time your moves to the beat to avoid obstacles."
    )

    game_description = (
        "A retro-futuristic rhythm runner. Jump and slide to the beat, avoid obstacles, and build your combo for a high score."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    BPS = 3  # Beats per second
    BEAT_INTERVAL = FPS // BPS

    # Colors
    COLOR_BG = (15, 15, 35)
    COLOR_GRID_1 = (25, 25, 55)
    COLOR_GRID_2 = (40, 40, 80)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_OBSTACLE = (255, 50, 100)
    COLOR_OBSTACLE_GLOW = (180, 20, 50)
    COLOR_BEAT = (50, 255, 150)
    COLOR_PARTICLE_SUCCESS = (255, 255, 0)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_MISS = (255, 0, 0)

    # Player
    PLAYER_X = 120
    PLAYER_SIZE = 20
    GROUND_Y = HEIGHT - 80
    JUMP_VELOCITY = -13
    GRAVITY = 0.7
    SLIDE_DURATION = 15  # frames

    # Obstacles
    OBSTACLE_WIDTH = 30
    OBSTACLE_HEIGHT_LOW = 25
    OBSTACLE_HEIGHT_HIGH = 50

    # Gameplay
    MAX_STEPS = 1000
    MAX_MISSES = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_combo = pygame.font.Font(None, 48)
        self.font_miss = pygame.font.Font(None, 60)
        
        # This will be initialized in reset()
        self.np_random = None

        self.reset()
        
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_y = self.GROUND_Y
        self.player_vy = 0
        self.player_state = "running"  # "running", "jumping", "sliding"
        self.slide_timer = 0
        self.action_initiated_frame = -100

        self.obstacles = []
        self.particles = []

        self.frame_count = 0
        self.steps = 0
        self.score = 0
        self.combo = 0
        self.misses = 0
        self.game_over = False

        self.obstacle_speed = 4.0
        self.next_obstacle_spawn_frame = self.BEAT_INTERVAL * 4
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        
        self._handle_input(movement)
        self._update_player()
        reward = self._update_game_state_and_get_reward()
        self._update_particles()
        self._update_difficulty()
        
        # Passive beat reward
        if self.frame_count % self.BEAT_INTERVAL == 0 and self.player_state == "running":
            reward += 1

        self.score += reward
        self.steps += 1
        self.frame_count += 1
        
        terminated = False
        if self.misses >= self.MAX_MISSES:
            terminated = True
            reward = -100 # Override reward on termination
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward = 100 # Override reward on termination
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # truncated
            self._get_info(),
        )

    def _handle_input(self, movement):
        if movement == 1 and self.player_state == "running":
            self.player_state = "jumping"
            self.player_vy = self.JUMP_VELOCITY
            self.action_initiated_frame = self.frame_count
            # sfx: jump_sound()
        elif movement == 2 and self.player_state == "running":
            self.player_state = "sliding"
            self.slide_timer = self.SLIDE_DURATION
            self.action_initiated_frame = self.frame_count
            # sfx: slide_sound()

    def _update_player(self):
        if self.player_state == "jumping":
            self.player_y += self.player_vy
            self.player_vy += self.GRAVITY
            if self.player_y >= self.GROUND_Y:
                self.player_y = self.GROUND_Y
                self.player_vy = 0
                self.player_state = "running"
        elif self.player_state == "sliding":
            self.slide_timer -= 1
            if self.slide_timer <= 0:
                self.player_state = "running"

    def _update_game_state_and_get_reward(self):
        reward = 0
        
        for obs in self.obstacles:
            obs['x'] -= self.obstacle_speed
        self.obstacles = [obs for obs in self.obstacles if obs['x'] > -self.OBSTACLE_WIDTH]

        if self.frame_count >= self.next_obstacle_spawn_frame:
            obs_type = self.np_random.choice(['low', 'high'])
            self.obstacles.append({'x': self.WIDTH, 'type': obs_type, 'cleared': False})
            self.next_obstacle_spawn_frame += self.BEAT_INTERVAL * self.np_random.integers(3, 6)

        player_rect = self._get_player_rect()
        for obs in self.obstacles:
            if obs['cleared']:
                continue

            obs_rect = self._get_obstacle_rect(obs)
            
            if player_rect.colliderect(obs_rect):
                self.misses += 1
                self.combo = 0
                obs['cleared'] = True
                self._spawn_particles(player_rect.centerx, player_rect.centery, 20, self.COLOR_OBSTACLE)
                # sfx: fail_sound()
                reward -= 20
                continue

            if obs['x'] + self.OBSTACLE_WIDTH < player_rect.left:
                obs['cleared'] = True
                
                was_in_correct_state = (obs['type'] == 'low' and self.player_state == 'jumping') or \
                                       (obs['type'] == 'high' and self.player_state == 'sliding')

                if was_in_correct_state:
                    reward += 5
                    if self.frame_count - self.action_initiated_frame < self.BEAT_INTERVAL * 1.5:
                        self.combo += 1
                        reward += (1 * self.combo)
                        self._spawn_particles(self.PLAYER_X, self.player_y, 15, self.COLOR_PARTICLE_SUCCESS)
                        # sfx: success_sound_pitched_up_with_combo()
                    else:
                        self.combo = 0
                else:
                    self.combo = 0
                    reward += 0.5
        
        if self.frame_count == self.action_initiated_frame:
            is_action_for_obstacle = False
            for obs in self.obstacles:
                time_to_impact = (obs['x'] - self.PLAYER_X) / self.obstacle_speed if self.obstacle_speed > 0 else float('inf')
                if 0 < time_to_impact < self.FPS * 1.5:
                    is_action_for_obstacle = True
                    break
            if not is_action_for_obstacle:
                reward -= 0.2
                self.combo = 0
        return reward

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.obstacle_speed = min(8.0, self.obstacle_speed + 0.05)
            
    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # particle gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_player_rect(self):
        if self.player_state == "sliding":
            w, h = self.PLAYER_SIZE * 2, self.PLAYER_SIZE / 2
            return pygame.Rect(self.PLAYER_X - w / 2, self.GROUND_Y - h, w, h)
        else:
            h = self.PLAYER_SIZE
            return pygame.Rect(self.PLAYER_X - h / 2, self.player_y - h, h, h)

    def _get_obstacle_rect(self, obs):
        if obs['type'] == 'low':
            return pygame.Rect(obs['x'], self.GROUND_Y - self.OBSTACLE_HEIGHT_LOW, self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT_LOW)
        else: # 'high'
            return pygame.Rect(obs['x'], self.GROUND_Y - self.PLAYER_SIZE - self.OBSTACLE_HEIGHT_HIGH, self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT_HIGH)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_obstacles()
        self._render_particles()
        self._render_player()
        self._render_beat_indicator()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "combo": self.combo, "misses": self.misses}

    def _render_background(self):
        for i in range(self.HEIGHT // 40 + 2):
            y = i * 40
            pygame.draw.line(self.screen, self.COLOR_GRID_1, (0, y), (self.WIDTH, y), 1)
        for i in range(self.WIDTH // 40 + 2):
            x = i * 40 - ((self.frame_count * self.obstacle_speed * 0.5) % 40)
            pygame.draw.line(self.screen, self.COLOR_GRID_1, (x, 0), (x, self.HEIGHT), 1)
            
        for i in range(self.HEIGHT // 80 + 2):
            y = i * 80
            pygame.draw.line(self.screen, self.COLOR_GRID_2, (0, y), (self.WIDTH, y), 2)
        for i in range(self.WIDTH // 80 + 2):
            x = i * 80 - ((self.frame_count * self.obstacle_speed) % 80)
            pygame.draw.line(self.screen, self.COLOR_GRID_2, (x, 0), (x, self.HEIGHT), 2)
            
        pygame.draw.line(self.screen, self.COLOR_BEAT, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 2)

    def _render_player(self):
        rect = self._get_player_rect()
        glow_rect = rect.inflate(10, 10)
        
        if self.player_state == "sliding":
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, glow_rect, border_radius=8)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=5)
        else:
            center = (int(rect.centerx), int(rect.centery))
            radius = int(rect.width / 2)
            glow_radius = int(glow_rect.width / 2)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], glow_radius, (*self.COLOR_PLAYER_GLOW, 100))
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, self.COLOR_PLAYER)
            
    def _render_obstacles(self):
        for obs in self.obstacles:
            rect = self._get_obstacle_rect(obs)
            glow_rect = rect.inflate(8, 8)
            
            if obs['type'] == 'low':
                p1 = (rect.left, rect.bottom)
                p2 = (rect.centerx, rect.top)
                p3 = (rect.right, rect.bottom)
                pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_OBSTACLE)
                pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_OBSTACLE)
            else:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, glow_rect, border_radius=5)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)

    def _render_particles(self):
        surf = pygame.display.get_surface()
        if surf is None: surf = self.screen # Headless check
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = p['color']
            pygame.draw.circle(surf, color, (int(p['x']), int(p['y'])), int(p['life'] * 0.2), 0)

    def _render_beat_indicator(self):
        beat_progress = (self.frame_count % self.BEAT_INTERVAL) / self.BEAT_INTERVAL
        pulse = (1 - beat_progress) ** 3
        radius = int(10 + 15 * pulse)
        alpha = int(50 + 200 * pulse)
        
        indicator_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(indicator_surf, radius, radius, radius, (*self.COLOR_BEAT, alpha))
        pygame.gfxdraw.aacircle(indicator_surf, radius, radius, radius, (*self.COLOR_BEAT, alpha))
        self.screen.blit(indicator_surf, (self.WIDTH / 2 - radius, self.HEIGHT - 40 - radius))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.combo > 1:
            combo_text = self.font_combo.render(f"x{self.combo}", True, self.COLOR_PARTICLE_SUCCESS)
            text_rect = combo_text.get_rect(topright=(self.WIDTH - 20, 10))
            self.screen.blit(combo_text, text_rect)

        for i in range(self.MAX_MISSES):
            color = self.COLOR_UI_TEXT if i >= self.misses else self.COLOR_MISS
            miss_text = self.font_miss.render("X", True, color)
            x_pos = self.WIDTH / 2 - (self.MAX_MISSES * 40 / 2) + i * 40 + 10
            self.screen.blit(miss_text, (x_pos, self.HEIGHT - 55))
            
    def _spawn_particles(self, x, y, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': life, 'max_life': life, 'color': color
            })

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        print("Beginning implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    obs, info = env.reset(seed=random.randint(0, 10000))
    terminated = False
    
    env.screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    
    action = [0, 0, 0]
    
    print(GameEnv.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        
        action = [movement, 0, 0]

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            # print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Combo: {info['combo']}")
            pass
        
        pygame.display.flip()
        env.clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Info: {info}")
    env.close()