import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the ninja. "
        "Collect numbers to reach the target sum."
    )

    game_description = (
        "A fast-paced arcade puzzle game. Control a ninja to collect falling numbers, "
        "aiming to reach a target sum before the timer runs out. Green numbers are "
        "positive, red are negative, and gold are high-value bonuses. Avoid the spikes!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_EPISODE_STEPS = 1200 # 40 seconds at 30fps

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_PLAYER = (230, 230, 255)
    COLOR_PLAYER_SCARF = (255, 60, 90)
    COLOR_OBSTACLE = (50, 60, 80)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_POSITIVE = (80, 255, 120)
    COLOR_NEGATIVE = (255, 80, 80)
    COLOR_BONUS = (255, 215, 0)
    COLOR_PROGRESS_BAR = (60, 180, 255)
    COLOR_PROGRESS_BAR_BG = (40, 50, 70)
    COLOR_WIN = (80, 255, 120, 200)
    COLOR_LOSE = (255, 80, 80, 200)

    # Player
    PLAYER_SPEED = 7
    PLAYER_SIZE = 12
    SCARF_LENGTH = 10

    # Numbers
    NUMBER_SPAWN_RATE = 8 # Lower is faster
    NUMBER_SIZE = 14

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

        try:
            self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
            self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_medium = pygame.font.Font(None, 30)
            self.font_large = pygame.font.Font(None, 60)
            
        self.game_over_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        
        # Fix: Initialize attributes before the first reset() call to avoid AttributeError.
        self.game_over = False
        self.win_condition = False
        self.stage = 0

        self.reset()
        # self.validate_implementation() # This is a non-standard helper, can be commented out

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.game_over and self.win_condition:
             self.stage += 1
        else:
             self.stage = 1

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False

        self.timer = max(10, 35 - self.stage * 2) * self.FPS
        self.target_sum = 100 + (self.stage - 1) * 25
        self.current_sum = 0
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50], dtype=np.float32)
        self.player_scarf_history = [self.player_pos.copy() for _ in range(self.SCARF_LENGTH)]

        self.numbers = []
        self.particles = []
        self._spawn_initial_numbers()
        
        self.obstacles = self._create_obstacles()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Time penalty

        if not self.game_over:
            self.steps += 1
            self.timer -= 1

            movement = action[0]
            self._update_player(movement)
            
            reward += self._update_numbers()
            self._update_particles()
            
            reward += self._handle_collisions()

            terminated, terminal_reward = self._check_termination()
            reward += terminal_reward
            if terminated:
                self.game_over = True
        
        terminated = self.game_over
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, movement):
        direction = np.array([0, 0], dtype=np.float32)
        if movement == 1: direction[1] = -1 # Up
        elif movement == 2: direction[1] = 1 # Down
        elif movement == 3: direction[0] = -1 # Left
        elif movement == 4: direction[0] = 1 # Right

        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)

        self.player_pos += direction * self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

        self.player_scarf_history.pop(0)
        self.player_scarf_history.append(self.player_pos.copy())

    def _update_numbers(self):
        collision_reward = 0
        if self.steps % self.NUMBER_SPAWN_RATE == 0:
            self._spawn_number()
            
        base_speed = 1.0 + self.stage * 0.2
        
        for number in self.numbers:
            number['pos'][1] += number['speed'] * base_speed
        
        # Check for collisions with player
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE, self.player_pos[1] - self.PLAYER_SIZE, self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2)
        
        numbers_to_remove = []
        for number in self.numbers:
            number_rect = pygame.Rect(number['pos'][0] - self.NUMBER_SIZE, number['pos'][1] - self.NUMBER_SIZE, self.NUMBER_SIZE*2, self.NUMBER_SIZE*2)
            if player_rect.colliderect(number_rect):
                # sfx: collect_number.wav
                self.current_sum += number['value']
                self.score += number['value']
                
                if number['type'] == 'bonus':
                    collision_reward += 1.0
                else:
                    collision_reward += 0.1
                
                self._create_particle_burst(number['pos'], number['color'])
                numbers_to_remove.append(number)
        
        # Remove collected and off-screen numbers
        self.numbers = [n for n in self.numbers if n not in numbers_to_remove and n['pos'][1] < self.SCREEN_HEIGHT + self.NUMBER_SIZE]

        return collision_reward

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE/2, self.player_pos[1] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for obs in self.obstacles:
            if player_rect.clipline(obs[0], obs[1]):
                # sfx: hit_obstacle.wav
                self.current_sum = max(0, self.current_sum - 10)
                reward -= 0.5
                self._create_particle_burst(self.player_pos, self.COLOR_OBSTACLE, 10, 2)
                # Small knockback
                center_x = self.SCREEN_WIDTH / 2
                if self.player_pos[0] < center_x:
                    self.player_pos[0] += 15
                else:
                    self.player_pos[0] -= 15
                break # only one collision per frame
        return reward

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] *= 0.95
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['size'] > 0.5]

    def _check_termination(self):
        if self.current_sum >= self.target_sum:
            self.win_condition = True
            # sfx: win_level.wav
            return True, 100.0
        if self.timer <= 0:
            # sfx: lose_level.wav
            return True, -100.0
        return False, 0.0
        
    def _spawn_initial_numbers(self):
        for _ in range(15):
            self._spawn_number(random_y=True)

    def _spawn_number(self, random_y=False):
        rand_val = self.np_random.random()
        pos_x = self.np_random.integers(self.NUMBER_SIZE, self.SCREEN_WIDTH - self.NUMBER_SIZE)
        pos_y = self.np_random.integers(0, self.SCREEN_HEIGHT - 100) if random_y else -self.NUMBER_SIZE
        
        if rand_val < 0.15: # Bonus
            value = self.np_random.integers(15, 25)
            color = self.COLOR_BONUS
            num_type = 'bonus'
        elif rand_val < 0.50: # Negative
            value = self.np_random.integers(-10, -1)
            color = self.COLOR_NEGATIVE
            num_type = 'neg'
        else: # Positive
            value = self.np_random.integers(1, 12)
            color = self.COLOR_POSITIVE
            num_type = 'pos'

        self.numbers.append({
            'pos': np.array([pos_x, pos_y], dtype=np.float32),
            'value': value,
            'color': color,
            'type': num_type,
            'speed': self.np_random.uniform(1.0, 1.5)
        })
        
    def _create_obstacles(self):
        obstacles = []
        num_spikes = 5
        spike_length = 20
        y_spacing = self.SCREEN_HEIGHT / (num_spikes + 1)
        for i in range(num_spikes):
            y = y_spacing * (i + 1)
            # Left side
            p1 = (0, y - 15)
            p2 = (spike_length, y)
            p3 = (0, y + 15)
            obstacles.append((p2, p1))
            obstacles.append((p2, p3))
            # Right side
            p1 = (self.SCREEN_WIDTH, y - 15)
            p2 = (self.SCREEN_WIDTH - spike_length, y)
            p3 = (self.SCREEN_WIDTH, y + 15)
            obstacles.append((p2, p1))
            obstacles.append((p2, p3))
        return obstacles

    def _create_particle_burst(self, pos, color, count=20, speed_mult=1):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * speed_mult
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'size': self.np_random.uniform(3, 7),
                'color': color,
                'lifespan': self.np_random.integers(15, 30)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Obstacles
        for obs in self.obstacles:
            pygame.draw.aaline(self.screen, self.COLOR_OBSTACLE, obs[0], obs[1], 3)
            
        # Numbers
        for number in self.numbers:
            pos = (int(number['pos'][0]), int(number['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.NUMBER_SIZE, number['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.NUMBER_SIZE, (0,0,0,50))
            
            text_surf = self.font_small.render(str(number['value']), True, self.COLOR_BG)
            text_rect = text_surf.get_rect(center=pos)
            self.screen.blit(text_surf, text_rect)

        # Particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = int(p['size'])
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, p['color'])

        # Player Scarf
        points = [tuple(p) for p in self.player_scarf_history]
        if len(points) > 2:
            pygame.draw.aalines(self.screen, self.COLOR_PLAYER_SCARF, False, points, 2)

        # Player
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        s = self.PLAYER_SIZE
        points = [(pos[0], pos[1] - s), (pos[0] - s/1.5, pos[1] + s/2), (pos[0] + s/1.5, pos[1] + s/2)]
        pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_PLAYER)
        pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_PLAYER)

    def _render_ui(self):
        def draw_text(text, font, x, y, color=self.COLOR_TEXT, shadow_color=self.COLOR_TEXT_SHADOW, center=False):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            text_rect = text_surf.get_rect()
            if center:
                text_rect.center = (x, y)
            else:
                text_rect.topleft = (x, y)
            self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
            self.screen.blit(text_surf, text_rect)
            return text_rect

        # Score and Stage
        draw_text(f"SCORE: {self.score}", self.font_medium, 10, 5)
        draw_text(f"STAGE: {self.stage}", self.font_medium, self.SCREEN_WIDTH - 150, 5)
        
        # Timer
        secs = max(0, self.timer // self.FPS)
        timer_text = f"{secs // 60:02d}:{secs % 60:02d}"
        draw_text(timer_text, self.font_medium, self.SCREEN_WIDTH / 2, 20, center=True)

        # Progress bar
        bar_width = self.SCREEN_WIDTH - 40
        bar_height = 20
        bar_x = 20
        bar_y = self.SCREEN_HEIGHT - 35

        progress = min(1.0, self.current_sum / self.target_sum if self.target_sum > 0 else 0)
        fill_width = int(bar_width * progress)

        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR, (bar_x, bar_y, fill_width, bar_height), border_radius=5)
        
        progress_text = f"{self.current_sum} / {self.target_sum}"
        draw_text(progress_text, self.font_medium, self.SCREEN_WIDTH / 2, bar_y + bar_height / 2, center=True)

    def _render_game_over(self):
        self.game_over_surface.fill((0,0,0,0))
        
        if self.win_condition:
            color = self.COLOR_WIN
            text = "LEVEL COMPLETE!"
        else:
            color = self.COLOR_LOSE
            text = "TIME'S UP!"
        
        pygame.draw.rect(self.game_over_surface, color, (0, 150, self.SCREEN_WIDTH, 100))
        
        text_surf = self.font_large.render(text, True, self.COLOR_TEXT)
        shadow_surf = self.font_large.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, 200))

        self.game_over_surface.blit(shadow_surf, (text_rect.x + 3, text_rect.y + 3))
        self.game_over_surface.blit(text_surf, text_rect)
        self.screen.blit(self.game_over_surface, (0,0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "current_sum": self.current_sum,
            "target_sum": self.target_sum
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not work in a headless environment, but is useful for local testing.
    # To run, you might need to comment out the `os.environ` line at the top.
    
    # Re-enable display for local play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Ninja Number Crunch")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Stage: {info['stage']}")
            pygame.time.wait(2000) # Pause for 2 seconds on game over
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()